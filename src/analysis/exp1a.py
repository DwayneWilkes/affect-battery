"""Exp 1a H1 analysis: per-condition Cohen's d vs no_conditioning baseline.

Per power-analysis spec (H1 primary test) and scoring-pipeline spec
(per-experiment views): for each non-baseline condition, compute mean
transfer accuracy, Cohen's d vs the NO_CONDITIONING baseline, and a raw
two-sample p-value, then apply Holm-Bonferroni correction across the
non-baseline family within Exp 1a.

Cohen's d uses pooled SD over the per-run accuracy series. The raw
p-value is from a two-sample Welch t-test (unequal variances). A
mixed-effects fit (lme4 / pymer4) is the planned upgrade for primary
analysis; the Welch test is a one-line drop-in for cross-validation.

UNAVAILABLE verdict: when the no_conditioning baseline is absent we
cannot compute deltas, so the function returns
verdict='unavailable_no_baseline' rather than silently substituting
NEUTRAL.
"""

from __future__ import annotations

from statistics import mean

from src.analysis._effect_size import cohens_d, run_accuracy, welch_p
from src.analysis_corrections import apply_holm_correction
from src.conditioning.prompts import Condition


def analyze_exp1a_corpus(corpus: list[dict], model: str) -> dict:
    """Run Exp 1a H1 analysis on a list of run records.

    Each run dict must have 'condition' (str) and 'transfer_correct' (list[bool]).
    Returns a structured dict suitable for downstream report rendering.
    """
    baseline_key = Condition.NO_CONDITIONING.value
    baseline_runs = [r for r in corpus if r["condition"] == baseline_key]

    if not baseline_runs:
        return {
            "model": model,
            "verdict": "unavailable_no_baseline",
            "per_condition_vs_baseline": {},
            "annotation": (
                "no_conditioning baseline runs absent; per-condition deltas "
                "cannot be computed. UNAVAILABLE is a measurement gap, not a "
                "null finding."
            ),
        }

    baseline_accs = [run_accuracy(r) for r in baseline_runs]
    baseline_mean = mean(baseline_accs)

    # Group non-baseline conditions
    by_cond: dict[str, list[float]] = {}
    for r in corpus:
        cond = r["condition"]
        if cond == baseline_key:
            continue
        by_cond.setdefault(cond, []).append(run_accuracy(r))

    conditions = list(by_cond.keys())
    raw_ps = [welch_p(by_cond[c], baseline_accs) for c in conditions]
    holm_qs = apply_holm_correction(raw_ps)

    per_condition: dict[str, dict] = {}
    for cond, raw_p, holm_q in zip(conditions, raw_ps, holm_qs):
        accs = by_cond[cond]
        per_condition[cond] = {
            "n_runs": len(accs),
            "mean_accuracy": mean(accs),
            "baseline_mean": baseline_mean,
            "cohens_d": cohens_d(accs, baseline_accs),
            "p_raw": raw_p,
            "p_holm_corrected": holm_q,
        }

    return {
        "model": model,
        "verdict": "complete",
        "per_condition_vs_baseline": per_condition,
        "baseline_n_runs": len(baseline_runs),
    }
