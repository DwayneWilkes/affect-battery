"""H4 aggregation pipeline + manipulation-check stitching (A4 + A5).

`analyze_h4_corpus(corpus, base_model, instruct_model)` does the cross-
experiment H4 work that previously had no production caller:

1. Group Exp 1a runs by (model, condition).
2. For each model, compute manipulation_check (paper §3.2.1 + spec).
   FAIL verdict excludes the model from primary H4 analysis.
3. For each surviving model, compute Cohen's d for strong_positive
   and strong_negative vs no_conditioning baseline.
4. Bundle per-model pos/neg pairs through compute_pair → compute_aggregate.
5. Run contrast_base_vs_instruct on the (base_model, instruct_model)
   aggregates.
6. Return the dict shape that render_h4_report expects.

The manipulation-check call is the production caller for `manipulation_check`
that review-finding A5 flagged as missing.
"""

from __future__ import annotations

from statistics import mean

from src.analysis._effect_size import cohens_d, run_accuracy
from src.analysis.asymmetry import (
    compute_aggregate,
    compute_pair,
    contrast_base_vs_instruct,
    per_model_verdict,
)
from src.analysis.stats._manipulation_check import (
    ManipulationCheckResult,
    ManipulationVerdict,
    manipulation_check,
)
from src.conditioning.prompts import Condition


_BASELINE = Condition.NO_CONDITIONING.value
_POS = Condition.STRONG_POSITIVE.value
_NEG = Condition.STRONG_NEGATIVE.value


def manipulation_check_from_corpus(
    corpus: list[dict],
    min_effect_size_pp: float = 2.0,
) -> dict[str, ManipulationCheckResult]:
    """Run manipulation_check per model from Exp 1a corpus.

    Source signal: per-run conditioning accuracy (mean over the 5-turn
    conditioning_correct array). Manipulation-check then compares
    strong_positive / strong_negative / no_conditioning means against
    the threshold.
    """
    by_model_cond: dict[str, dict[str, list[float]]] = {}
    for run in corpus:
        model = run.get("model")
        cond = run.get("condition")
        cc = run.get("conditioning_correct", [])
        if model is None or cond is None or not cc:
            continue
        acc = sum(1 for x in cc if x) / len(cc)
        by_model_cond.setdefault(model, {}).setdefault(cond, []).append(acc)

    out: dict[str, ManipulationCheckResult] = {}
    for model, by_cond in by_model_cond.items():
        # manipulation_check requires the three direction probes; skip
        # models with incomplete coverage rather than crashing.
        if _POS not in by_cond or _NEG not in by_cond:
            continue
        out[model] = manipulation_check(
            accuracy_by_condition=by_cond,
            model=model,
            min_effect_size_pp=min_effect_size_pp,
        )
    return out


def _per_model_pair(corpus: list[dict], model: str) -> dict | None:
    """Compute the single (pos, neg) effect-size pair for one model from
    transfer-accuracy data. Returns None when any of the three required
    conditions is missing for this model.
    """
    by_cond_accs: dict[str, list[float]] = {}
    for run in corpus:
        if run.get("model") != model:
            continue
        cond = run.get("condition")
        if cond is None:
            continue
        by_cond_accs.setdefault(cond, []).append(run_accuracy(run))

    baseline = by_cond_accs.get(_BASELINE, [])
    pos = by_cond_accs.get(_POS, [])
    neg = by_cond_accs.get(_NEG, [])
    if not baseline or not pos or not neg:
        return None

    pos_d = cohens_d(pos, baseline)
    neg_d = cohens_d(neg, baseline)
    return compute_pair(pos_effect=pos_d, neg_effect=neg_d)


def analyze_h4_corpus(
    corpus: list[dict],
    base_model: str,
    instruct_model: str,
    apply_manipulation_check: bool = True,
) -> dict:
    """Full H4 aggregation across the model family + base-vs-instruct
    contrast.

    Per asymmetry-contrast spec: per-model aggregates feed both the
    cross-model contrast and the per-model verdict table.
    """
    if not corpus:
        return {
            "verdict": "unavailable_no_data",
            "per_model_aggregates": {},
            "per_model_verdicts": {},
            "manipulation_check_results": {},
            "asymmetry_delta_ratio": None,
        }

    # A5: manipulation-check first; FAIL excludes the model.
    mc_results: dict[str, ManipulationCheckResult] = (
        manipulation_check_from_corpus(corpus) if apply_manipulation_check else {}
    )
    excluded = {
        m for m, r in mc_results.items()
        if r.verdict == ManipulationVerdict.FAIL
    }

    models = sorted({r.get("model") for r in corpus if r.get("model")})
    per_model_aggregates: dict[str, dict] = {}
    per_model_verdicts: dict[str, str] = {}
    for model in models:
        if model in excluded:
            per_model_verdicts[model] = "excluded_manipulation_fail"
            continue
        pair = _per_model_pair(corpus, model)
        if pair is None:
            per_model_verdicts[model] = "degenerate"
            continue
        agg = compute_aggregate([pair])
        per_model_aggregates[model] = agg
        per_model_verdicts[model] = per_model_verdict(
            agg, p_value=None, mde=None,
        )

    # Cross-model base-vs-instruct contrast (only when both survive).
    contrast: dict | None = None
    if base_model in per_model_aggregates and instruct_model in per_model_aggregates:
        contrast = contrast_base_vs_instruct(
            per_model_aggregates,
            base_model=base_model,
            instruct_model=instruct_model,
        )

    return {
        "verdict": "complete",
        "model_family": _model_family(base_model),
        "base_model": base_model,
        "instruct_model": instruct_model,
        "per_model_aggregates": per_model_aggregates,
        "per_model_verdicts": per_model_verdicts,
        "manipulation_check_results": mc_results,
        "ratio_base": (contrast or {}).get("ratio_base"),
        "ratio_instruct": (contrast or {}).get("ratio_instruct"),
        "asymmetry_delta_ratio": (contrast or {}).get("asymmetry_delta_ratio"),
        "test_a_primary_delta_ratio_gt_1": (
            (contrast or {}).get("test_a_primary_delta_ratio_gt_1")
        ),
        "test_b_secondary_diff_instruct_gt_diff_base": (
            (contrast or {}).get("test_b_secondary_diff_instruct_gt_diff_base")
        ),
    }


def _model_family(model_name: str) -> str:
    """Strip variant suffixes to get a family label (e.g. 'Llama-3-8B'
    from 'Meta-Llama-3-8B-Instruct'). Best-effort; falls back to the
    full name."""
    base = model_name.replace("Meta-", "").replace("-Instruct", "")
    return base.split("/")[-1]
