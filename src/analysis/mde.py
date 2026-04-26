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


# ---------------------------------------------------------------------------
# ICC fallback recommendation (power-analysis spec "ICC estimation with
# fallback prior")
# ---------------------------------------------------------------------------

ICC_FALLBACK_PRIOR = 0.20
ICC_SENSITIVITY = 0.35


def _design_effect(cluster_size: int, icc: float) -> float:
    """Variance inflation factor for clustered data: 1 + (m-1) * ICC."""
    return 1.0 + (cluster_size - 1) * icc


def n_required_for_mde(
    baseline_acc: float,
    target_mde: float,
    cluster_size: int,
    icc: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Smallest per-group n at which `compute_mde` returns <= target_mde
    after applying the design-effect inflation factor for ICC.

    Inverts the MDE formula: solve n_effective = (z_α/2 + z_power)² ·
    2·p(1-p) / target_mde², then n_required = n_effective ·
    design_effect.
    """
    if target_mde <= 0:
        raise ValueError(f"target_mde must be positive, got {target_mde!r}")
    z_alpha_2 = norm.ppf(1.0 - alpha / 2.0)
    z_beta = norm.ppf(power)
    pq = baseline_acc * (1.0 - baseline_acc)
    n_effective = ((z_alpha_2 + z_beta) ** 2) * 2.0 * pq / (target_mde ** 2)
    n_required = n_effective * _design_effect(cluster_size, icc)
    # Round up: ceil-to-int because partial samples don't help.
    return int(n_required) + (1 if n_required > int(n_required) else 0)


def icc_fallback_recommendation(
    baseline_acc: float,
    target_mde: float,
    cluster_size: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Recommended n-per-condition under the ICC fallback policy.

    Per the power-analysis spec scenario "Pilot is too thin for ICC
    estimation": when n_models < 3 in pilot data, ICC cannot be
    reliably estimated. We compute the recommended n at both the prior
    (ICC=0.20) and a sensitivity-check (ICC=0.35), then recommend the
    max as the conservative choice.

    Returns a dict suitable for serializing into the per-experiment
    power report.
    """
    n_at_prior = n_required_for_mde(
        baseline_acc, target_mde, cluster_size, ICC_FALLBACK_PRIOR,
        alpha=alpha, power=power,
    )
    n_at_sensitivity = n_required_for_mde(
        baseline_acc, target_mde, cluster_size, ICC_SENSITIVITY,
        alpha=alpha, power=power,
    )
    return {
        "recommended_n_per_condition": max(n_at_prior, n_at_sensitivity),
        "n_at_icc_prior": n_at_prior,
        "n_at_icc_sensitivity": n_at_sensitivity,
        "icc_prior_value": ICC_FALLBACK_PRIOR,
        "icc_sensitivity_value": ICC_SENSITIVITY,
        "icc_source": "fallback_0.20_prior_with_sensitivity_at_0.35",
        "rationale": (
            "Pilot data insufficient (n_models < 3) for reliable ICC "
            "estimation. Reporting recommended n at both the prior "
            "(0.20) and sensitivity (0.35); take the max."
        ),
    }
