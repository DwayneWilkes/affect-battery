"""Exp 3a analysis: quadratic fit on accuracy ~ intensity_level.

Per scoring-pipeline spec "Quadratic-model fit for H3a" + power-analysis
spec H3a: fit accuracy ~ beta_0 + beta_1 * L + beta_2 * L^2 across
the 7 intensity levels. H3a-confirming signature: beta_2 < 0 (inverted-U).
Report:
  beta_0, beta_1, beta_2 with one-sided p-value on beta_2 (H_a: beta_2 < 0)
  AIC + BIC for the quadratic vs linear-only models.

Closed-form OLS via numpy.linalg.lstsq when numpy is available; falls
back to a hand-rolled normal-equations solver. The one-sided p-value
uses the Welch-style standard-normal proxy from _effect_size; replace
with proper t / mixed-effects at submit-time .
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
    # Student-t is the right reference distribution at the small n we use;
    # standard-normal would be anti-conservative.
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


def analyze_arousal_magnitude(accuracy_by_level: dict[int, list[float]]) -> dict:
    """Y-D under arousal-as-magnitude mapping (amendment 002).

    Recodes each cell's level to arousal = |level - 4|, mapping signed
    valence-intensity {1..7} to unsigned arousal {3, 2, 1, 0, 1, 2, 3}.
    Fits the OLS quadratic on accuracy ~ b0 + b1·arousal + b2·arousal²
    and reports β₂, SE, and one-sided p value.

    No directional pre-registration: this is a sensitivity analysis
    asking whether the canonical Y-D arousal-magnitude mapping recovers
    a curve from data the signed-axis test refuted.
    """
    levels_present = sorted(accuracy_by_level.keys())
    if len(levels_present) < 3:
        raise ValueError(
            f"need >= 3 distinct intensity levels for quadratic fit; "
            f"got {len(levels_present)}"
        )

    xs: list[float] = []
    ys: list[float] = []
    for L in levels_present:
        arousal = abs(L - 4)
        for acc in accuracy_by_level[L]:
            xs.append(float(arousal))
            ys.append(float(acc))

    x_arr = np.array(xs)
    y_arr = np.array(ys)
    n = len(xs)

    X = np.column_stack([np.ones(n), x_arr, x_arr ** 2])
    coeffs, _, rss = _ols_fit(X, y_arr)
    b0, b1, b2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    se = _se_of_coefficient(X, rss, n, k=3)
    se_b2 = float(se[2])

    if se_b2 == 0.0 or not math.isfinite(se_b2):
        p_one_sided = 1.0 if b2 >= 0 else 0.0
    else:
        t_stat = b2 / se_b2
        p_one_sided = student_t_cdf(t_stat, df=n - 3)

    if b2 != 0 and math.isfinite(b2):
        peak_arousal = -b1 / (2 * b2)
    else:
        peak_arousal = float("nan")

    return {
        "n": n,
        "arousal_values": sorted({int(abs(L - 4)) for L in levels_present}),
        "level_to_arousal": {L: int(abs(L - 4)) for L in levels_present},
        "beta_0": b0,
        "beta_1": b1,
        "beta_2": b2,
        "beta_2_se": se_b2,
        "beta_2_p_one_sided": p_one_sided,
        "peak_arousal": peak_arousal,
    }


def analyze_within_subjects(corpus: list[dict]) -> dict:
    """Within-subjects quadratic fit with per-item fixed effects (amendment 002).

    `corpus` is a list of run records, each with `body.intensity_level`,
    `body.binary_correct`, and `body.item_id`. Fits OLS with per-item
    dummy variables to partial out item-level variance, then reports the
    quadratic coefficient β₂ and SE on the level + level² regressors.

    Items with zero variance across levels (all-correct or all-incorrect)
    are dropped before fitting; their inclusion would be perfectly
    collinear with the per-item dummy and contribute no information to
    the level coefficients.
    """
    by_item: dict[str, list[tuple[int, int]]] = {}
    for record in corpus:
        body = record.get("body") or {}
        item_id = body.get("item_id") or ""
        level = body.get("intensity_level")
        binary = body.get("binary_correct")
        if not item_id or level is None or binary is None:
            continue
        by_item.setdefault(item_id, []).append((int(level), int(binary)))

    surviving_items = [
        item_id for item_id, obs in by_item.items()
        if len({b for _, b in obs}) > 1
    ]
    n_dropped = len(by_item) - len(surviving_items)
    if len(surviving_items) < 3:
        raise ValueError(
            f"need >= 3 items with non-zero variance across levels; "
            f"got {len(surviving_items)} (dropped {n_dropped} zero-variance items)"
        )

    item_index = {item_id: i for i, item_id in enumerate(surviving_items)}
    rows: list[tuple[float, float, int]] = []
    for item_id in surviving_items:
        for level, binary in by_item[item_id]:
            rows.append((float(level), float(binary), item_index[item_id]))

    n_obs = len(rows)
    n_items = len(surviving_items)
    X = np.zeros((n_obs, 3 + max(n_items - 1, 0)))
    y = np.zeros(n_obs)
    for r, (L, b, item_idx) in enumerate(rows):
        X[r, 0] = 1.0
        X[r, 1] = L
        X[r, 2] = L * L
        if item_idx > 0:
            X[r, 2 + item_idx] = 1.0
        y[r] = b

    coeffs, _, rss = _ols_fit(X, y)
    b0, b1, b2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    se = _se_of_coefficient(X, rss, n_obs, k=X.shape[1])
    se_b2 = float(se[2])

    if se_b2 == 0.0 or not math.isfinite(se_b2):
        p_one_sided = 1.0 if b2 >= 0 else 0.0
    else:
        t_stat = b2 / se_b2
        df = n_obs - X.shape[1]
        p_one_sided = student_t_cdf(t_stat, df=df)

    return {
        "n_observations": n_obs,
        "n_items_used": n_items,
        "n_items_dropped_zero_variance": n_dropped,
        "beta_0": b0,
        "beta_1": b1,
        "beta_2": b2,
        "beta_2_se": se_b2,
        "beta_2_p_one_sided": p_one_sided,
    }
