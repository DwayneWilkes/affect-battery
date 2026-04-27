"""End-to-end analysis pipeline.

`analyze_results_dir(results_dir, model)` is the single production caller
that ties together:

1. Per-experiment corpus loading (Exp 1a/1b/2/3a/3b/3c JSON files)
2. Per-experiment analysis (analyze_*_corpus / analyze_*)
3. Per-experiment report rendering (render_*_report)
4. H4 cross-experiment aggregation + manipulation-check exclusions
5. Family-wise correction across the primary hypothesis family
   (H1, H1b-TOST, H2, H3a, H4)
6. Aggregate landing-page report tying all of the above together
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.exp1a import analyze_exp1a_corpus
from src.analysis.exp1b import analyze_exp1b
from src.analysis.exp2 import analyze_exp2_corpus
from src.analysis.exp3a import analyze_exp3a
from src.analysis.exp3b import analyze_exp3b_corpus
from src.analysis.exp3c import analyze_exp3c_corpus
from src.analysis.h4 import analyze_h4_corpus
from src.analysis.reports.aggregate import render_aggregate
from src.analysis.reports.exp1a import render_exp1a_report
from src.analysis.reports.exp1b import render_exp1b_report
from src.analysis.reports.exp2 import render_exp2_report
from src.analysis.reports.exp3a import render_exp3a_report
from src.analysis.reports.exp3b import render_exp3b_report
from src.analysis.reports.exp3c import render_exp3c_report
from src.analysis.reports.h4 import render_h4_report
from src.analysis.stats.corrections import apply_family_corrections


def _resolve_corpus_dir(results_dir: Path, experiment: str) -> Path:
    """Pick the data directory for `experiment` under `results_dir`.

    Layouts probed in priority order (newest → oldest):
      1. Single-experiment pilot dir: <results_dir>/data/<condition>/
         The pilot dir's name carries the experiment id (e.g.
         '2026-04-26_gpt-5.4-nano_exp1a/'), so the inner data/ dir
         doesn't need to repeat it. Each result file's `experiment_type`
         field is the canonical experiment identity.
      2. Multi-experiment pilot dir: <results_dir>/data/<experiment>/
         Older layout where one pilot dir held all 6 experiments under
         data/<exp>/ subdirs. Kept for backward compat with migrated
         baselines.
      3. Legacy flat layout: <results_dir>/<experiment>/
         Pre-pilot-root layout from before the data/ subdir was
         introduced. Still works; the analyzer rglobs to find files.
    """
    # Layout 1: single-experiment pilot dir. Result files at
    # data/<cond>/<NNNN>.json. We probe by checking for any *.json
    # file under data/ that matches the requested experiment_type.
    flat_data = results_dir / "data"
    if flat_data.exists() and not (flat_data / experiment).exists():
        # data/ exists but data/<exp>/ does NOT — this is the new layout.
        # Verify by sampling: at least one JSON file should declare the
        # requested experiment in its config / experiment_type.
        sample = next(flat_data.rglob("*.json"), None)
        if sample is not None:
            try:
                payload = json.loads(sample.read_text())
                exp_type = payload.get("experiment_type") or (
                    payload.get("config") or {}
                ).get("experiment_type")
                if exp_type == experiment:
                    return flat_data
            except (json.JSONDecodeError, OSError):
                pass

    # Layout 2: explicit per-experiment subdir under data/.
    pilot_root_layout = results_dir / "data" / experiment
    if pilot_root_layout.exists():
        return pilot_root_layout

    # Layout 3: legacy flat layout.
    legacy = results_dir / experiment
    if legacy.exists():
        return legacy

    # Layout 4 (sibling lookup): when results_dir is itself a single-
    # experiment pilot dir like `<pilot_root>/exp1b/` and we're asked
    # for a DIFFERENT experiment (e.g. exp1a for the three-way
    # comparison), look at the sibling pilot dir `<pilot_root>/exp1a/`
    # and resolve from there.
    sibling = results_dir.parent / experiment
    if sibling.is_dir() and sibling != results_dir:
        sibling_data = sibling / "data"
        if sibling_data.is_dir():
            return sibling_data

    # Last resort: return the (non-existent) legacy path; _load_corpus
    # checks .exists() and returns [] when nothing matches.
    return legacy


def _load_corpus(results_dir: Path, experiment: str) -> list[dict]:
    """Load all result JSONs under the resolved corpus dir as a flat list.

    Backfills `transfer_correct` from `transfer_responses` + `transfer_expected`
    via `score_factual_qa` when a legacy result file lacks the field (or
    has it as None / empty list). This rescues pilot data saved before
    the runner started grading transfer responses at write time.
    """
    from src.scoring.accuracy import score_factual_qa

    exp_dir = _resolve_corpus_dir(results_dir, experiment)
    if not exp_dir.exists():
        return []
    corpus: list[dict] = []
    for path in sorted(exp_dir.rglob("*.json")):
        if path.name == "events.jsonl":
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "condition" not in data and "config" in data:
            data["condition"] = data.get("config", {}).get("condition", "")
        existing = data.get("transfer_correct")
        if not existing:
            responses = data.get("transfer_responses") or []
            expected = data.get("transfer_expected") or []
            aliases = data.get("transfer_expected_aliases") or [
                [] for _ in expected
            ]
            if responses and len(responses) == len(expected):
                data["transfer_correct"] = [
                    score_factual_qa(r, e, aliases=a)
                    for r, e, a in zip(responses, expected, aliases)
                ]
            else:
                data["transfer_correct"] = []
        corpus.append(data)
    return corpus


def _extract_primary_p_values(
    exp1a_analysis: dict | None,
    exp1b_analysis: dict | None,
    exp2_analysis: dict | None,
    exp3a_analysis: dict | None,
    h4_analysis: dict | None,
) -> dict[str, float]:
    """Pull the primary-family p-values that exist out of the per-experiment
    analyses for family-wise correction.

    Hypotheses in the primary family per power-analysis spec:
      H1       -> Exp 1a per-condition tests; smallest Holm-corrected p
                  across non-baseline conditions.
      H1b      -> Exp 1b session-2 directional p (smallest across conditions)
      H1b_TOST -> Exp 1b session-2 TOST p (smallest across conditions)
      H2       -> Exp 2 asymmetry-ratio bootstrap p (one-sided H_a:
                  |neg_auc| / |pos_auc| > 1.0)
      H3a      -> Exp 3a beta_2 one-sided Student-t p
      H4       -> H4 cross-experiment asymmetry_delta_ratio bootstrap p
                  (one-sided H_a: ratio_instruct / ratio_base > 1.0)
    """
    from src.analysis.stats.bootstrap import bootstrap_ratio_p

    p: dict[str, float] = {}

    if exp1a_analysis and exp1a_analysis.get("verdict") == "complete":
        per_cond = exp1a_analysis.get("per_condition_vs_baseline", {})
        if per_cond:
            p["H1"] = min(c["p_holm_corrected"] for c in per_cond.values())

    if exp1b_analysis and exp1b_analysis.get("verdict") == "complete":
        comparison = exp1b_analysis.get("three_way_comparison", {})
        directional = [
            c["session_2_directional_p"] for c in comparison.values()
            if c.get("session_2_directional_p") is not None
        ]
        tost = [
            c["session_2_tost_p"] for c in comparison.values()
            if c.get("session_2_tost_p") is not None
        ]
        if directional:
            p["H1b"] = min(directional)
        if tost:
            p["H1b_TOST"] = min(tost)

    # H2: asymmetry-ratio bootstrap on the per-condition AUCs. The
    # analyze_exp2_corpus output exposes per-condition `recovery_metrics`
    # with the AUC values; we pull the strong-positive and strong-negative
    # AUCs and bootstrap-test |neg|/|pos| > 1.0.
    if exp2_analysis and exp2_analysis.get("verdict") == "complete":
        by_cond = exp2_analysis.get("by_condition", {})
        neg_cell = by_cond.get("strong_negative", {})
        pos_cell = by_cond.get("strong_positive", {})
        neg_auc = (neg_cell.get("recovery_metrics") or {}).get("auc")
        pos_auc = (pos_cell.get("recovery_metrics") or {}).get("auc")
        if neg_auc is not None and pos_auc is not None:
            p["H2"] = bootstrap_ratio_p(
                numerator=[neg_auc],
                denominator=[pos_auc],
                n_resamples=2000,
                seed=0,
            )

    # H3a: the analyze_exp3a output exposes a Student-t one-sided p for
    # beta_2 < 0; pull it directly.
    if exp3a_analysis and "beta_2_p_one_sided" in exp3a_analysis:
        p["H3a"] = float(exp3a_analysis["beta_2_p_one_sided"])

    # H4: bootstrap p-value on the cross-experiment asymmetry_delta_ratio
    # using the per-model aggregate ratios. Pre-registered test (a) is
    # `delta_ratio > 1.0` (one-sided).
    if h4_analysis and h4_analysis.get("verdict") == "complete":
        per_model = h4_analysis.get("per_model_aggregates") or {}
        base_model = h4_analysis.get("base_model")
        instruct_model = h4_analysis.get("instruct_model")
        base_ratio = (per_model.get(base_model) or {}).get("ratio_geomean")
        instruct_ratio = (per_model.get(instruct_model) or {}).get("ratio_geomean")
        if base_ratio is not None and instruct_ratio is not None and base_ratio > 0:
            p["H4"] = bootstrap_ratio_p(
                numerator=[instruct_ratio],
                denominator=[base_ratio],
                n_resamples=2000,
                seed=0,
            )

    return p


_VALID_EXPERIMENT_NAMES = frozenset(
    ["exp1a", "exp1b", "exp2", "exp3a", "exp3b", "exp3c"]
)


def analyze_results_dir(
    results_dir: Path,
    model: str = "aggregate",
    base_model: str = "Meta-Llama-3-8B",
    instruct_model: str = "Meta-Llama-3-8B-Instruct",
    only_experiment: str | None = None,
) -> dict[str, Path]:
    """Run analysis + report rendering across experiments present
    under results_dir.

    When `results_dir`'s basename is itself an experiment name (e.g.
    `<pilot_root>/exp1b/`), render is auto-scoped to only that
    experiment — sibling corpora are loaded for cross-references
    (exp1b's three-way comparison needs exp1a's corpus) but only the
    requested experiment's report is written. Pass `only_experiment`
    explicitly to override the auto-detection.

    Returns a dict mapping experiment_id (or 'h4', 'aggregate') to the
    path of the rendered report.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect single-experiment scope from dir name. If results_dir
    # is named after an experiment AND only_experiment isn't already set,
    # restrict report rendering to that experiment only.
    if only_experiment is None and results_dir.name in _VALID_EXPERIMENT_NAMES:
        only_experiment = results_dir.name

    def _should_render(exp: str) -> bool:
        return only_experiment is None or exp == only_experiment

    # Reports directory: pilot-root layouts (those with <root>/data/<exp>/)
    # get reports under <root>/reports/. Legacy flat layouts keep reports
    # at the top level so old pilot dirs still work without migration.
    pilot_root_layout = (results_dir / "data").is_dir()
    reports_dir = results_dir / "reports" if pilot_root_layout else results_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, Path] = {}
    aggregate_payload: dict = {}

    # ---- Exp 1a ----
    exp1a_corpus = _load_corpus(results_dir, "exp1a")
    exp1a_analysis: dict | None = None
    if exp1a_corpus:
        exp1a_analysis = analyze_exp1a_corpus(exp1a_corpus, model=model)
        if _should_render("exp1a"):
            path = reports_dir / "exp1a_report.md"
            render_exp1a_report(exp1a_analysis, output_path=path)
            rendered["exp1a"] = path
            aggregate_payload["exp1a"] = {
                "model": model, "verdict": exp1a_analysis["verdict"],
            }

    # ---- Exp 1b (needs both 1a and 1b corpora for three-way comparison) ----
    exp1b_corpus = _load_corpus(results_dir, "exp1b")
    exp1b_analysis: dict | None = None
    if exp1b_corpus and exp1a_corpus:
        exp1b_analysis = analyze_exp1b(
            exp1a_corpus=exp1a_corpus,
            exp1b_corpus=exp1b_corpus,
            model=model,
            h1b_dual_tests=True,
        )
        if _should_render("exp1b"):
            path = reports_dir / "exp1b_report.md"
            render_exp1b_report(exp1b_analysis, output_path=path)
            rendered["exp1b"] = path
            aggregate_payload["exp1b"] = {
                "model": model, "verdict": exp1b_analysis["verdict"],
            }

    # ---- Exp 2 (A1) ----
    exp2_corpus = _load_corpus(results_dir, "exp2")
    exp2_analysis: dict | None = None
    if exp2_corpus:
        exp2_analysis = analyze_exp2_corpus(exp2_corpus, model=model)
        if _should_render("exp2"):
            path = reports_dir / "exp2_report.md"
            render_exp2_report(exp2_analysis, output_path=path)
            rendered["exp2"] = path
        aggregate_payload["exp2"] = {
            "model": model, "verdict": exp2_analysis["verdict"],
        }

    # ---- Exp 3a ----
    # The Exp 3a corpus is loaded as a flat list and grouped by intensity
    # level. analyze_exp3a expects accuracy_by_level: {level: [accs]}
    # where each entry is a per-cell binary correctness (0 or 1) read
    # from Exp3aBody.binary_correct. analyze_exp3a's quadratic fit takes
    # the per-level mean over the list, so per-cell binaries aggregate
    # cleanly into per-level accuracies.
    exp3a_corpus = _load_corpus(results_dir, "exp3a")
    exp3a_analysis: dict | None = None
    if exp3a_corpus:
        accuracy_by_level: dict[int, list[float]] = {}
        for run in exp3a_corpus:
            body = run.get("body") or {}
            level = body.get("intensity_level")
            binary_correct = body.get("binary_correct")
            if level is None or binary_correct is None:
                continue
            accuracy_by_level.setdefault(level, []).append(float(binary_correct))
        if len(accuracy_by_level) >= 3:
            exp3a_analysis = analyze_exp3a(accuracy_by_level)
            exp3a_analysis["model"] = model
            if _should_render("exp3a"):
                path = reports_dir / "exp3a_report.md"
                render_exp3a_report(exp3a_analysis, output_path=path)
                rendered["exp3a"] = path
                aggregate_payload["exp3a"] = {
                    "model": model,
                    "verdict": "complete",
                }

    # ---- Exp 3b (A2) ----
    exp3b_corpus = _load_corpus(results_dir, "exp3b")
    if exp3b_corpus:
        exp3b_analysis = analyze_exp3b_corpus(exp3b_corpus, model=model)
        if _should_render("exp3b"):
            path = reports_dir / "exp3b_report.md"
            render_exp3b_report(exp3b_analysis, output_path=path)
            rendered["exp3b"] = path
            aggregate_payload["exp3b"] = {
                "model": model, "verdict": exp3b_analysis["verdict"],
            }

    # ---- Exp 3c (A3) ----
    exp3c_corpus = _load_corpus(results_dir, "exp3c")
    if exp3c_corpus:
        exp3c_analysis = analyze_exp3c_corpus(exp3c_corpus, model=model)
        if _should_render("exp3c"):
            path = reports_dir / "exp3c_report.md"
            render_exp3c_report(exp3c_analysis, output_path=path)
            rendered["exp3c"] = path
            aggregate_payload["exp3c"] = {
                "model": model, "verdict": exp3c_analysis["verdict"],
            }

    # ---- H4 cross-experiment + manipulation-check (A4 + A5) ----
    h4_analysis: dict | None = None
    # H4 is a cross-model contrast: it only makes sense when the corpus
    # contains BOTH the base and the instruct model. If <2 distinct models
    # are present, suppress the H4 render entirely so single-model pilots
    # don't produce nonsense reports with em-dashes for missing data.
    distinct_models = {
        r.get("model") or (r.get("config") or {}).get("model_name")
        for r in exp1a_corpus or []
    }
    distinct_models.discard(None)
    if exp1a_corpus and len(distinct_models) >= 2:
        h4_analysis = analyze_h4_corpus(
            corpus=exp1a_corpus,
            base_model=base_model,
            instruct_model=instruct_model,
        )
        if h4_analysis.get("verdict") == "complete":
            path = reports_dir / "h4_report.md"
            render_h4_report(h4_analysis, output_path=path)
            rendered["h4"] = path
            aggregate_payload["h4"] = {
                "model_family": h4_analysis.get("model_family"),
                "ratio_base": h4_analysis.get("ratio_base"),
                "ratio_instruct": h4_analysis.get("ratio_instruct"),
                "asymmetry_delta_ratio": h4_analysis.get("asymmetry_delta_ratio"),
            }

    # ---- Family-wise corrections (A6) ----
    p_values = _extract_primary_p_values(
        exp1a_analysis=exp1a_analysis,
        exp1b_analysis=exp1b_analysis,
        exp2_analysis=exp2_analysis,
        exp3a_analysis=exp3a_analysis,
        h4_analysis=h4_analysis,
    )
    if p_values:
        family_membership = {h: "primary" for h in p_values}
        corrected = apply_family_corrections(p_values, family_membership)
        aggregate_payload["primary_family_corrections"] = corrected

    # ---- Aggregate landing page ----
    aggregate_path = reports_dir / "AGGREGATE_REPORT.md"
    render_aggregate(aggregate_payload, output_path=aggregate_path)
    rendered["aggregate"] = aggregate_path

    return rendered
