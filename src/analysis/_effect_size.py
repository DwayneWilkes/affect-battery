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
    """Two-sided Welch t-test p-value (standard-normal CDF approximation).

    Weekend-ship proxy. Replace with full mixed-effects fit at submit-time
    (logged in GAPS.md). Returns 1.0 when either group has fewer than 2
    samples (test cannot be computed); 0.0 when both means differ but SE
    is zero (perfect separation).
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
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
