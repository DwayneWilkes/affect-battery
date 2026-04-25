"""Exp 1a H1 analysis: per-condition Cohen's d vs no_conditioning baseline.

Per power-analysis spec (H1 primary test) and scoring-pipeline spec
(per-experiment views): for each non-baseline condition, compute mean
transfer accuracy, Cohen's d vs the NO_CONDITIONING baseline, and a raw
two-sample p-value, then apply Holm-Bonferroni correction across the
non-baseline family within Exp 1a.

Cohen's d uses pooled SD over the per-run accuracy series. The raw p-value
is from a two-sample Welch t-test approximation; for the weekend ship this
is a quick proxy that does NOT assume equal variance — full lme4/pymer4
mixed-effects fit is logged in GAPS.md.

UNAVAILABLE verdict: when the no_conditioning baseline is absent we cannot
compute deltas, so the function returns verdict='unavailable_no_baseline'
rather than silently substituting NEUTRAL.
"""

from __future__ import annotations

import math
from statistics import mean

from src.analysis_corrections import apply_holm_correction
from src.conditioning.prompts import Condition


def _accuracy(run: dict) -> float:
    correct = run.get("transfer_correct", [])
    if not correct:
        return 0.0
    return sum(1 for c in correct if c) / len(correct)


def _pooled_sd(xs: list[float], ys: list[float]) -> float:
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sx2 = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    sy2 = sum((y - my) ** 2 for y in ys) / (ny - 1)
    pooled_var = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    return math.sqrt(pooled_var)


def _cohens_d(treatment: list[float], baseline: list[float]) -> float:
    sd = _pooled_sd(treatment, baseline)
    diff = mean(treatment) - mean(baseline)
    if sd == 0.0:
        if diff == 0.0:
            return 0.0
        return math.copysign(math.inf, diff)
    return diff / sd


def _welch_p(treatment: list[float], baseline: list[float]) -> float:
    """Two-sided Welch t-test p-value (normal approximation).

    Weekend-ship proxy: uses the standard-normal CDF rather than the
    Student-t with Welch-Satterthwaite df. Acceptable for the synthetic
    test path; primary analysis at submit-time replaces with mixed-effects.
    """
    nx, ny = len(treatment), len(baseline)
    if nx < 2 or ny < 2:
        return 1.0
    mx, my = mean(treatment), mean(baseline)
    sx2 = sum((x - mx) ** 2 for x in treatment) / (nx - 1)
    sy2 = sum((y - my) ** 2 for y in baseline) / (ny - 1)
    se = math.sqrt(sx2 / nx + sy2 / ny)
    if se == 0.0:
        return 0.0 if mx != my else 1.0
    z = abs(mx - my) / se
    # Two-sided p from standard-normal: 2 * (1 - Phi(z))
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


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

    baseline_accs = [_accuracy(r) for r in baseline_runs]
    baseline_mean = mean(baseline_accs)

    # Group non-baseline conditions
    by_cond: dict[str, list[float]] = {}
    for r in corpus:
        cond = r["condition"]
        if cond == baseline_key:
            continue
        by_cond.setdefault(cond, []).append(_accuracy(r))

    conditions = list(by_cond.keys())
    raw_ps = [_welch_p(by_cond[c], baseline_accs) for c in conditions]
    holm_qs = apply_holm_correction(raw_ps)

    per_condition: dict[str, dict] = {}
    for cond, raw_p, holm_q in zip(conditions, raw_ps, holm_qs):
        accs = by_cond[cond]
        per_condition[cond] = {
            "n_runs": len(accs),
            "mean_accuracy": mean(accs),
            "baseline_mean": baseline_mean,
            "cohens_d": _cohens_d(accs, baseline_accs),
            "p_raw": raw_p,
            "p_holm_corrected": holm_q,
        }

    return {
        "model": model,
        "verdict": "complete",
        "per_condition_vs_baseline": per_condition,
        "baseline_n_runs": len(baseline_runs),
    }
