"""Minimum-detectable-effect (MDE) for two-proportion comparisons.

Returns the smallest absolute difference between two proportions that a
two-sided binomial test can reliably detect at the given α and power,
using the normal approximation with baseline-variance doubled (the
conservative choice when the alternative-hypothesis proportion is not
yet known).

Formula:
    MDE = (z_{1-α/2} + z_{power}) · sqrt(2 · p(1-p) / n)

where p is the baseline proportion and n is the per-group sample size.
Output is a fraction in [0.0, 1.0] — callers that want percentage points
multiply by 100 themselves.

References: Fleiss (1981) Statistical Methods for Rates and Proportions,
equation 2.29; matches the formula behind
`statsmodels.stats.proportion.power_proportions_2indep` within the
normal-approximation regime. Hand-rolled here because statsmodels isn't
a project dependency and the math is ~6 lines.

Spec: affect-battery-task-difficulty-calibration::scoring-pipeline::
"Minimum-detectable-effect reporting". .
"""

from __future__ import annotations

from scipy.stats import norm


def compute_mde(
    baseline_acc: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum absolute difference detectable against `baseline_acc` with
    `n` observations per group at two-sided significance `alpha` and
    statistical `power`."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    if not (0.0 <= baseline_acc <= 1.0):
        raise ValueError(
            f"baseline_acc must be in [0.0, 1.0], got {baseline_acc!r}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha!r}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0.0, 1.0), got {power!r}")

    z_alpha_2 = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    pq = baseline_acc * (1.0 - baseline_acc)
    return float((z_alpha_2 + z_beta) * (2.0 * pq / n) ** 0.5)
