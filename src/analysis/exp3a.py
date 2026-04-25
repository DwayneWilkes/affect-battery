"""Exp 3a analysis: quadratic fit on accuracy ~ intensity_level.

Per scoring-pipeline spec "Quadratic-model fit for H3a" + power-analysis
spec H3a formula: fit accuracy ~ beta_0 + beta_1 * L + beta_2 * L^2 across
the 7 intensity levels. H3a-confirming signature: beta_2 < 0 (inverted-U).
Report:
  beta_0, beta_1, beta_2 with one-sided p-value on beta_2 (H_a: beta_2 < 0)
  AIC + BIC for the quadratic vs linear-only models.

Closed-form OLS via numpy.linalg.lstsq when numpy is available; falls
back to a hand-rolled normal-equations solver. The one-sided p-value
uses the Welch-style standard-normal proxy from _effect_size; replace
with proper t / mixed-effects at submit-time per GAPS.md.
"""

from __future__ import annotations

import math

import numpy as np

from src.analysis.stats._distributions import student_t_cdf
from src.analysis.stats.decay import _aic_bic


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS fit returning (coefficients, residuals_vector, RSS)."""
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coeffs
    rss = float(residuals @ residuals)
    return coeffs, residuals, rss


def _se_of_coefficient(X: np.ndarray, rss: float, n: int, k: int) -> np.ndarray:
    """Sigma^2 * (X'X)^-1 diagonal -> SE per coefficient.

    Returns a vector of SEs aligned with the columns of X. Falls back to
    a vector of inf when (X'X) is singular.
    """
    if n - k <= 0:
        return np.full(X.shape[1], math.inf)
    sigma2 = rss / (n - k)
    try:
        cov = sigma2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], math.inf)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


def analyze_exp3a(accuracy_by_level: dict[int, list[float]]) -> dict:
    """Quadratic fit + β₂ one-sided test.

    `accuracy_by_level` maps intensity level (1..7 typically) to a list
    of per-run accuracies at that level. Need >= 3 distinct levels for
    a quadratic fit.
    """
    levels_present = sorted(accuracy_by_level.keys())
    if len(levels_present) < 3:
        raise ValueError(
            f"need >= 3 distinct intensity levels for quadratic fit; "
            f"got {len(levels_present)}"
        )

    # Flatten to (level, accuracy) pairs.
    xs: list[float] = []
    ys: list[float] = []
    for L in levels_present:
        for acc in accuracy_by_level[L]:
            xs.append(float(L))
            ys.append(float(acc))

    x_arr = np.array(xs)
    y_arr = np.array(ys)
    n = len(xs)

    # Quadratic model: y = b0 + b1*L + b2*L^2
    X_quad = np.column_stack([np.ones(n), x_arr, x_arr ** 2])
    coeffs_q, _, rss_q = _ols_fit(X_quad, y_arr)
    b0, b1, b2 = float(coeffs_q[0]), float(coeffs_q[1]), float(coeffs_q[2])
    se_q = _se_of_coefficient(X_quad, rss_q, n, k=3)
    se_b2 = float(se_q[2])

    # One-sided p-value for H_a: beta_2 < 0 using Student-t with df = n-k.
    # Per review-finding #9: standard-normal CDF is anti-conservative for
    # the small n we use; Student-t is the right reference distribution.
    if se_b2 == 0.0 or not math.isfinite(se_b2):
        p_one_sided = 1.0 if b2 >= 0 else 0.0
    else:
        t_stat = b2 / se_b2
        df = n - 3  # 3 free params: b0, b1, b2
        p_one_sided = student_t_cdf(t_stat, df=df)

    quad_aic, quad_bic = _aic_bic(rss_q, n, k=3)

    # Linear model for AIC/BIC comparison.
    X_lin = np.column_stack([np.ones(n), x_arr])
    _, _, rss_l = _ols_fit(X_lin, y_arr)
    lin_aic, lin_bic = _aic_bic(rss_l, n, k=2)

    return {
        "n": n,
        "levels": levels_present,
        "beta_0": b0,
        "beta_1": b1,
        "beta_2": b2,
        "beta_2_se": se_b2,
        "beta_2_p_one_sided": p_one_sided,
        "quadratic_rss": rss_q,
        "quadratic_aic": quad_aic,
        "quadratic_bic": quad_bic,
        "linear_rss": rss_l,
        "linear_aic": lin_aic,
        "linear_bic": lin_bic,
    }
