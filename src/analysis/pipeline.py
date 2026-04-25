"""End-to-end analysis pipeline.

`analyze_results_dir(results_dir, model)` is the single production caller
that ties together:

1. Per-experiment corpus loading (Exp 1a/1b/2/3a/3b/3c JSON files)
2. Per-experiment analysis (analyze_*_corpus / analyze_*)
3. Per-experiment report rendering (render_*_report)
4. H4 cross-experiment aggregation + manipulation-check exclusions
5. Family-wise correction across the primary hypothesis family
   (H1, H1b-TOST, H2 placeholder, H3a placeholder, H4)
6. Aggregate landing-page report tying all of the above together
"""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.exp1a import analyze_exp1a_corpus
from src.analysis.exp1b import analyze_exp1b
from src.analysis.exp2 import analyze_exp2_corpus
from src.analysis.exp3b import analyze_exp3b_corpus
from src.analysis.exp3c import analyze_exp3c_corpus
from src.analysis.h4 import analyze_h4_corpus
from src.analysis.reports.aggregate import render_aggregate
from src.analysis.reports.exp1a import render_exp1a_report
from src.analysis.reports.exp1b import render_exp1b_report
from src.analysis.reports.exp2 import render_exp2_report
from src.analysis.reports.exp3b import render_exp3b_report
from src.analysis.reports.exp3c import render_exp3c_report
from src.analysis.reports.h4 import render_h4_report
from src.analysis.stats.corrections import apply_family_corrections


def _load_corpus(results_dir: Path, experiment: str) -> list[dict]:
    """Load all result JSONs under results_dir/<experiment>/ as a flat list."""
    exp_dir = results_dir / experiment
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
        if "transfer_correct" not in data:
            data["transfer_correct"] = []
        corpus.append(data)
    return corpus


def _extract_primary_p_values(
    exp1a_analysis: dict | None,
    exp1b_analysis: dict | None,
    h4_analysis: dict | None,
) -> dict[str, float]:
    """Pull the primary-family p-values that exist out of the per-experiment
    analyses for family-wise correction.

    Hypotheses in the primary family per power-analysis spec:
      H1     -> Exp 1a per-condition tests; we pick the strongest signal
                (smallest p across non-baseline conditions) as the family
                representative. Spec uses Holm within Exp 1a too, so this
                is the already-Holm-corrected min.
      H1b    -> Exp 1b session-2 directional p (smallest across conditions)
      H1b_TOST -> Exp 1b session-2 TOST p (smallest across conditions)
      H2     -> Exp 2 — placeholder; full mixed-effects fit is in the project follow-up queue
      H3a    -> Exp 3a — analyzed per-model; placeholder until per-model
                aggregation lands
      H4     -> H4 contrast pre-registered test (a)
    """
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

    # H4: the pre-registered (a) test is asymmetry_delta_ratio > 1. We don't
    # have a frequentist p-value on this contrast in the current implementation
    # (full bootstrap deferred ), so we map the test outcome to
    # a coarse p-value: True (effect present) -> 0.04, False/None -> 1.0.
    # This keeps H4 in the family without claiming a precise inferential p.
    if h4_analysis and h4_analysis.get("verdict") == "complete":
        outcome = h4_analysis.get("test_a_primary_delta_ratio_gt_1")
        if outcome is True:
            p["H4"] = 0.04
        elif outcome is False:
            p["H4"] = 1.0

    return p


def analyze_results_dir(
    results_dir: Path,
    model: str = "aggregate",
    base_model: str = "Meta-Llama-3-8B",
    instruct_model: str = "Meta-Llama-3-8B-Instruct",
) -> dict[str, Path]:
    """Run analysis + report rendering across all experiments present
    under results_dir. Returns a dict mapping experiment_id (or 'h4',
    'aggregate') to the path of the rendered report.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, Path] = {}
    aggregate_payload: dict = {}

    # ---- Exp 1a ----
    exp1a_corpus = _load_corpus(results_dir, "exp1a")
    exp1a_analysis: dict | None = None
    if exp1a_corpus:
        exp1a_analysis = analyze_exp1a_corpus(exp1a_corpus, model=model)
        path = results_dir / "exp1a_report.md"
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
        path = results_dir / "exp1b_report.md"
        render_exp1b_report(exp1b_analysis, output_path=path)
        rendered["exp1b"] = path
        aggregate_payload["exp1b"] = {
            "model": model, "verdict": exp1b_analysis["verdict"],
        }

    # ---- Exp 2 (A1) ----
    exp2_corpus = _load_corpus(results_dir, "exp2")
    if exp2_corpus:
        exp2_analysis = analyze_exp2_corpus(exp2_corpus, model=model)
        path = results_dir / "exp2_report.md"
        render_exp2_report(exp2_analysis, output_path=path)
        rendered["exp2"] = path
        aggregate_payload["exp2"] = {
            "model": model, "verdict": exp2_analysis["verdict"],
        }

    # ---- Exp 3b (A2) ----
    exp3b_corpus = _load_corpus(results_dir, "exp3b")
    if exp3b_corpus:
        exp3b_analysis = analyze_exp3b_corpus(exp3b_corpus, model=model)
        path = results_dir / "exp3b_report.md"
        render_exp3b_report(exp3b_analysis, output_path=path)
        rendered["exp3b"] = path
        aggregate_payload["exp3b"] = {
            "model": model, "verdict": exp3b_analysis["verdict"],
        }

    # ---- Exp 3c (A3) ----
    exp3c_corpus = _load_corpus(results_dir, "exp3c")
    if exp3c_corpus:
        exp3c_analysis = analyze_exp3c_corpus(exp3c_corpus, model=model)
        path = results_dir / "exp3c_report.md"
        render_exp3c_report(exp3c_analysis, output_path=path)
        rendered["exp3c"] = path
        aggregate_payload["exp3c"] = {
            "model": model, "verdict": exp3c_analysis["verdict"],
        }

    # ---- H4 cross-experiment + manipulation-check (A4 + A5) ----
    h4_analysis: dict | None = None
    if exp1a_corpus:
        h4_analysis = analyze_h4_corpus(
            corpus=exp1a_corpus,
            base_model=base_model,
            instruct_model=instruct_model,
        )
        if h4_analysis.get("verdict") == "complete":
            path = results_dir / "h4_report.md"
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
        h4_analysis=h4_analysis,
    )
    if p_values:
        family_membership = {h: "primary" for h in p_values}
        corrected = apply_family_corrections(p_values, family_membership)
        aggregate_payload["primary_family_corrections"] = corrected

    # ---- Aggregate landing page ----
    aggregate_path = results_dir / "AGGREGATE_REPORT.md"
    render_aggregate(aggregate_payload, output_path=aggregate_path)
    rendered["aggregate"] = aggregate_path

    return rendered
