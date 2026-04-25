"""Shared effect-size + accuracy primitives for per-experiment analyses.

Extracted from `src/analysis/exp1a.py` for reuse by `src/analysis/exp1b.py`
(Task 4.2 DRY check: effect-size computation shared with Exp 1a). Future
per-experiment analyses (Exp 2, Exp 3a-c) should import from here rather
than re-implementing.
"""

from __future__ import annotations

import math
from statistics import mean


def run_accuracy(run: dict) -> float:
    """Mean correctness over a run's transfer_correct list."""
    correct = run.get("transfer_correct", [])
    if not correct:
        return 0.0
    return sum(1 for c in correct if c) / len(correct)


def pooled_sd(xs: list[float], ys: list[float]) -> float:
    """Pooled-variance SD over two independent samples (length >= 2 each)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    sx2 = sum((x - mx) ** 2 for x in xs) / (nx - 1)
    sy2 = sum((y - my) ** 2 for y in ys) / (ny - 1)
    pooled_var = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    return math.sqrt(pooled_var)


def cohens_d(treatment: list[float], baseline: list[float]) -> float:
    """Cohen's d using pooled SD. Returns +/- inf when SD is zero but
    means differ (perfect separation), 0.0 when both means and SD tie."""
    sd = pooled_sd(treatment, baseline)
    diff = mean(treatment) - mean(baseline)
    if sd == 0.0:
        if diff == 0.0:
            return 0.0
        return math.copysign(math.inf, diff)
    return diff / sd


def welch_p(treatment: list[float], baseline: list[float]) -> float:
    """Two-sided Welch's t-test p-value via scipy.stats.ttest_ind.

    Per review-finding #10: the previous implementation used a standard-
    normal CDF approximation rather than Welch's t with Welch-Satterthwaite
    df, which is mildly anti-conservative for the small n we use in tests.
    This now delegates to scipy.stats.ttest_ind(equal_var=False).

    Returns 1.0 when either group has fewer than 2 samples (test cannot
    be computed); 0.0 when both means differ but SE is zero (perfect
    separation).
    """
    from scipy.stats import ttest_ind

    nx, ny = len(treatment), len(baseline)
    if nx < 2 or ny < 2:
        return 1.0
    # Quick perfect-separation short-circuit so callers don't see scipy
    # warnings on degenerate input.
    mx, my = mean(treatment), mean(baseline)
    sx2 = sum((x - mx) ** 2 for x in treatment) / (nx - 1)
    sy2 = sum((y - my) ** 2 for y in baseline) / (ny - 1)
    if sx2 == 0.0 and sy2 == 0.0:
        return 0.0 if mx != my else 1.0
    result = ttest_ind(treatment, baseline, equal_var=False)
    return float(result.pvalue)
