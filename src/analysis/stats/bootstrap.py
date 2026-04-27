"""Bootstrap p-values for cross-experiment statistics that don't have
a closed-form sampling distribution under the null.

Two helpers:

- `bootstrap_ratio_p`: tests H_a `numerator/denominator > 1.0` (one-
  sided) by resampling pairs with replacement and computing the
  fraction of bootstrap samples where the resampled ratio is <= 1.0.
  Used for the H4 cross-experiment asymmetry contrast and the H2
  per-experiment asymmetry ratio.

- `bootstrap_difference_p`: tests H_a `mean(treatment) > mean(baseline)`
  (one-sided) by resampling per-group with replacement.

Both default to 2000 resamples (paper §3.3 convention) with seed
control for reproducibility.
"""

from __future__ import annotations

import random
from statistics import mean


def _safe_ratio(numerator: list[float], denominator: list[float]) -> float | None:
    """Sum-based ratio used inside the bootstrap loop. Returns None when
    the denominator sums to zero (numerically degenerate)."""
    n_sum = sum(abs(x) for x in numerator)
    d_sum = sum(abs(x) for x in denominator)
    if d_sum == 0.0:
        return None
    return n_sum / d_sum


def bootstrap_ratio_p(
    numerator: list[float],
    denominator: list[float],
    n_resamples: int = 2000,
    seed: int = 0,
    null_value: float = 1.0,
) -> float:
    """One-sided p-value for H_a: ratio > null_value.

    Resamples both arrays with replacement (paired-bootstrap shape) and
    returns the fraction of resamples whose ratio falls at-or-below the
    null. Returns 1.0 when the observed ratio is below the null
    (test cannot reject).

    Both arrays must be non-empty. The denominator entries are taken in
    absolute value so the ratio operates in magnitude space (consistent
    with paper §3.3 asymmetry-ratio convention).
    """
    if not numerator or not denominator:
        raise ValueError("numerator and denominator must be non-empty")
    rng = random.Random(seed)
    n_a, n_b = len(numerator), len(denominator)
    n_below_or_at_null = 0
    n_valid = 0
    for _ in range(n_resamples):
        a = [numerator[rng.randrange(n_a)] for _ in range(n_a)]
        b = [denominator[rng.randrange(n_b)] for _ in range(n_b)]
        r = _safe_ratio(a, b)
        if r is None:
            continue
        n_valid += 1
        if r <= null_value:
            n_below_or_at_null += 1
    if n_valid == 0:
        return 1.0
    return n_below_or_at_null / n_valid


def bootstrap_difference_p(
    treatment: list[float],
    baseline: list[float],
    n_resamples: int = 2000,
    seed: int = 0,
) -> float:
    """One-sided p-value for H_a: mean(treatment) > mean(baseline).

    Resamples each group with replacement and counts the fraction of
    bootstrap means where treatment_mean <= baseline_mean. Returns 1.0
    on degenerate input.
    """
    if not treatment or not baseline:
        raise ValueError("treatment and baseline must be non-empty")
    rng = random.Random(seed)
    n_t, n_b = len(treatment), len(baseline)
    count_at_or_below = 0
    for _ in range(n_resamples):
        t = [treatment[rng.randrange(n_t)] for _ in range(n_t)]
        b = [baseline[rng.randrange(n_b)] for _ in range(n_b)]
        if mean(t) <= mean(b):
            count_at_or_below += 1
    return count_at_or_below / n_resamples
