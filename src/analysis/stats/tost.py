"""TOST (Two One-Sided Tests) equivalence test.

Per power-analysis spec "H1b null-equivalence test (TOST)" + Lakens
(2017): an effect is equivalent to zero within +/- epsilon if BOTH the
upper-bound test (effect < +epsilon) and the lower-bound test
(effect > -epsilon) reject at level alpha. The combined p-value is
max(p_lower, p_upper).

Distribution: standard-normal CDF approximation (matches
the welch_p proxy in src/analysis/_effect_size.py). Replace with full
Student-t at submit-time .

References
----------
Lakens, D. (2017). "Equivalence tests: A practical primer for t-tests,
    correlations, and meta-analyses." Social Psychological and
    Personality Science, 8(4), 355-362.
"""

from __future__ import annotations

import math

from src.analysis.stats._distributions import normal_cdf


def tost_equivalence(
    effect: float,
    se: float,
    epsilon: float,
    alpha: float = 0.05,
) -> dict:
    """Two one-sided tests for equivalence of `effect` to zero within
   /- epsilon.

    Parameters
    ----------
    effect : float
        Observed effect (e.g. Cohen's d, mean difference, standardized).
    se : float
        Standard error of the effect estimate.
    epsilon : float
        Equivalence band half-width (must be > 0). Test passes when
        |true effect| < epsilon at the chosen alpha.
    alpha : float
        Per-test significance level (default 0.05). The TOST decision uses
        the same alpha for both one-sided tests.

    Returns
    -------
    dict
        p_lower : p-value for H0_lower (effect <= -epsilon)
        p_upper : p-value for H0_upper (effect >= +epsilon)
        p_tost  : max(p_lower, p_upper) — the conservative combined p
        equivalent : True iff p_tost < alpha
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon!r}")
    if se <= 0.0:
        # Zero SE: deterministic equivalence iff |effect| < epsilon
        within = abs(effect) < epsilon
        return {
            "p_lower": 0.0 if within else 1.0,
            "p_upper": 0.0 if within else 1.0,
            "p_tost": 0.0 if within else 1.0,
            "equivalent": within,
        }
    z_lower = (effect - (-epsilon)) / se   # H0_lower: effect <= -eps; reject if z_lower large positive
    z_upper = ((+epsilon) - effect) / se   # H0_upper: effect >= +eps; reject if z_upper large positive
    # One-sided p for "test rejects when z is large positive" => p = 1 - Phi(z)
    p_lower = 1.0 - normal_cdf(z_lower)
    p_upper = 1.0 - normal_cdf(z_upper)
    p_tost = max(p_lower, p_upper)
    return {
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": p_tost < alpha,
    }
