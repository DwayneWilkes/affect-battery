"""Shared effect-size + accuracy primitives for per-experiment analyses.

Cohen's d, pooled SD, Welch's t-test p-value, and per-run accuracy
helpers used across Exp 1a, Exp 1b, and the H4 aggregation.
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
    """Two-sided Welch's t-test p-value with degenerate-input handling.

    Branches:
      - n_x or n_y < 2: test undefined → return 1.0.
      - both variances zero: perfect separation if means differ → 0.0,
        otherwise no signal → 1.0.
      - one variance zero: Welch reduces to Student's t with
        df = (n_of_nonzero_group - 1). Compute directly to avoid
        scipy's catastrophic-cancellation RuntimeWarning, which fires
        when a group is constant. This is common in pilot data with
        small n where one cell happens to score all-identical.
      - both variances nonzero: delegate to scipy.stats.ttest_ind.

    Computing the one-zero-variance branch by hand isn't a workaround:
    it's the principled path. The scipy warning flags real precision
    loss, but the underlying math (one-sample t against a fixed value)
    is well-defined and we can compute it directly with full precision.
    """
    from scipy.stats import t as _t_dist
    from scipy.stats import ttest_ind

    nx, ny = len(treatment), len(baseline)
    if nx < 2 or ny < 2:
        return 1.0
    mx, my = mean(treatment), mean(baseline)
    sx2 = sum((x - mx) ** 2 for x in treatment) / (nx - 1)
    sy2 = sum((y - my) ** 2 for y in baseline) / (ny - 1)
    if sx2 == 0.0 and sy2 == 0.0:
        return 0.0 if mx != my else 1.0
    if sx2 == 0.0 or sy2 == 0.0:
        # One side is constant. Reduce to Student's t with df driven by
        # the side that has variance.
        if sx2 == 0.0:
            se = math.sqrt(sy2 / ny)
            df = ny - 1
        else:
            se = math.sqrt(sx2 / nx)
            df = nx - 1
        if se == 0.0 or df < 1:
            return 0.0 if mx != my else 1.0
        t_stat = (mx - my) / se
        return 2.0 * float(_t_dist.sf(abs(t_stat), df))
    result = ttest_ind(treatment, baseline, equal_var=False)
    return float(result.pvalue)
