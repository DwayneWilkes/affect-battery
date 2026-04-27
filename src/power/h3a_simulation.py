"""Simulation-based power analysis for the H3a quadratic-curve test.

H3a fits a quadratic regression `accuracy ~ beta_0 + beta_1·level + beta_2·level^2`
across the seven intensity levels and tests `beta_2 < 0` at alpha = 0.05
(one-sided). The Yerkes-Dodson prediction is an inverted-U: peak in the
middle, lower at the edges. A negative beta_2 is the structural signature.

The simulation assumes:
  - Per-level mean accuracy follows the assumed inverted-U with peak at
    `peak_level` (default 4) and per-level variance `sigma_per_level`.
  - Within-condition variance is Gaussian with mean = level mean, sd = sigma.
  - Cells are sampled iid (no within-cluster correlation).
  - Each simulated cell is one (level, run) pair; the regression is fit
    on the n_per_level * 7 cells.

For designs with within-cluster correlation (multiple measurements per
cell, repeated raters per condition, etc.), the recommended n from this
simulation should be inflated by approximately (1 + (k - 1) * ICC)
where k is the cluster size; the `icc` argument here is recorded in the
result for downstream auditing but is not currently used in sampling.

Power is computed by Monte Carlo: simulate the data N times, fit the
quadratic each time, count the fraction of trials where the one-sided
test rejects beta_2 < 0 at alpha = 0.05.

Two entry points:
  - `power_at_n(n_per_level, beta2_assumed, sigma_per_level, icc, ...)` :
        compute power at a fixed sample size.
  - `find_min_n(beta2_assumed, sigma_per_level, icc, target_power, ...)` :
        binary search for the smallest n that achieves target power.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# Default Yerkes-Dodson prediction: inverted-U with peak at level 4.
# Numbers are illustrative defaults; actual probe data should override.
DEFAULT_PEAK_LEVEL = 4
DEFAULT_PEAK_ACCURACY = 0.70
DEFAULT_EDGE_DROP = 0.10  # accuracy drop from peak to levels 1 and 7


@dataclass
class PowerResult:
    n_per_level: int
    beta2_assumed: float
    icc: float
    n_simulations: int
    power: float
    n_significant: int
    alpha: float


def _quadratic_curve(
    peak_level: int = DEFAULT_PEAK_LEVEL,
    peak_accuracy: float = DEFAULT_PEAK_ACCURACY,
    edge_drop: float = DEFAULT_EDGE_DROP,
) -> list[float]:
    """Generate a 7-point inverted-U curve.

    Returns a list of mean accuracy values for levels 1..7. The curve is
    `peak_accuracy - edge_drop * ((level - peak_level)^2 / max_dist^2)`
    where max_dist is the distance from peak to the farthest level.
    """
    max_dist = max(peak_level - 1, 7 - peak_level)
    return [
        peak_accuracy - edge_drop * ((level - peak_level) ** 2) / (max_dist ** 2)
        for level in range(1, 8)
    ]


def _quadratic_beta2(curve: list[float]) -> float:
    """Fit `y = a + b*x + c*x^2` via OLS on a 7-point curve and return c."""
    xs = list(range(1, 8))
    n = len(xs)
    if n < 3:
        return 0.0
    # OLS for y = a + b*x + c*x^2: build (X^T X) and (X^T y), solve.
    sum_x = sum(xs)
    sum_x2 = sum(x ** 2 for x in xs)
    sum_x3 = sum(x ** 3 for x in xs)
    sum_x4 = sum(x ** 4 for x in xs)
    sum_y = sum(curve)
    sum_xy = sum(x * y for x, y in zip(xs, curve))
    sum_x2y = sum(x ** 2 * y for x, y in zip(xs, curve))
    # Solve 3x3 linear system:
    #   [n     sum_x  sum_x2 ] [a]   [sum_y  ]
    #   [sum_x sum_x2 sum_x3 ] [b] = [sum_xy ]
    #   [sum_x2 sum_x3 sum_x4] [c]   [sum_x2y]
    a_mat = [
        [n, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ]
    b_vec = [sum_y, sum_xy, sum_x2y]
    coeffs = _solve_3x3(a_mat, b_vec)
    return coeffs[2]


def _solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve 3x3 linear system Ax = b via Cramer's rule."""
    def det3(m):
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    det_a = det3(a)
    if abs(det_a) < 1e-12:
        raise ValueError("singular matrix in quadratic OLS")
    out = []
    for col in range(3):
        m = [row[:] for row in a]
        for r in range(3):
            m[r][col] = b[r]
        out.append(det3(m) / det_a)
    return out


def _xtx_inv_diag_for_quadratic(n_levels: int = 7) -> float:
    """Return (X^T X)^{-1}_{3,3} for the [1, x, x^2] design with x = 1..n_levels.

    This is the diagonal element of the inverse Gram matrix corresponding to
    the quadratic coefficient. The standard error of beta_2 in OLS is
    sigma_residual * sqrt(this value); on a regression of per-level means
    with sigma_residual = sigma_per_observation / sqrt(n_per_level), the
    SE simplifies to (sigma_per_obs / sqrt(n_per_level)) * sqrt(this value).

    For n_levels=7 this returns 196/16464 ≈ 0.01191.
    """
    xs = list(range(1, n_levels + 1))
    sum_x = sum(xs)
    sum_x2 = sum(x ** 2 for x in xs)
    sum_x3 = sum(x ** 3 for x in xs)
    sum_x4 = sum(x ** 4 for x in xs)
    a = [
        [n_levels, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4],
    ]
    # Cofactor for the (3,3) element is the top-left 2x2 minor.
    minor_33 = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    det_a = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    return minor_33 / det_a


# Pre-compute for the 7-level design; used everywhere.
_XTX_INV_22_7LEVEL = _xtx_inv_diag_for_quadratic(7)


def _t_critical_one_sided(df: int, alpha: float = 0.05) -> float:
    """Approximate one-sided t critical value via Wilson-Hilferty.

    For df >= 30 the value approaches ~1.645 (one-sided α=0.05).
    For smaller df we use a simple approximation good enough for this
    Monte Carlo simulation; the exact value would come from scipy.stats.t.
    """
    # Common one-sided 5% critical values (df: t_crit)
    table = {
        5: 2.015, 10: 1.812, 15: 1.753, 20: 1.725, 25: 1.708,
        30: 1.697, 50: 1.676, 100: 1.660, 1000: 1.646,
    }
    if df in table:
        return table[df]
    if df < 5:
        return 2.5
    # Linear interpolation
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i + 1]:
            t0, t1 = table[keys[i]], table[keys[i + 1]]
            return t0 + (t1 - t0) * (df - keys[i]) / (keys[i + 1] - keys[i])
    return 1.645  # df > 1000


def power_at_n(
    n_per_level: int,
    beta2_assumed: float,
    sigma_per_level: list[float],
    icc: float = 0.20,
    n_simulations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    peak_level: int = DEFAULT_PEAK_LEVEL,
) -> PowerResult:
    """Monte Carlo power calculation for the H3a quadratic test.

    Args:
        n_per_level: cells sampled per intensity level.
        beta2_assumed: hypothesized quadratic coefficient. The simulation
            generates data under this beta2 (negative beta2 = inverted-U).
        sigma_per_level: list of 7 within-level standard deviations (one
            per intensity level). Estimated from the variance probe.
        icc: intra-class correlation. Effective n is reduced by
            n_eff = n / (1 + (n - 1) * icc).
        n_simulations: Monte Carlo iterations.
        seed: RNG seed for reproducibility.
        alpha: one-sided significance level.
        peak_level: level number where the curve peaks (default 4).

    Returns:
        PowerResult with the computed power = fraction of simulations in
        which the one-sided t-test on beta2 rejected at `alpha`.
    """
    if len(sigma_per_level) != 7:
        raise ValueError("sigma_per_level must have 7 entries (one per level)")
    rng = random.Random(seed)

    # Construct the assumed inverted-U curve from beta2_assumed.
    # If beta2 = c, the parabola y = a + b*x + c*x^2 has peak at x = -b/(2c).
    # Solving for (a, b) given peak_level: b = -2*c*peak_level.
    # We anchor the peak value at DEFAULT_PEAK_ACCURACY for absolute level.
    c = beta2_assumed
    b = -2 * c * peak_level
    a = DEFAULT_PEAK_ACCURACY - b * peak_level - c * (peak_level ** 2)
    true_means = [a + b * x + c * (x ** 2) for x in range(1, 8)]

    # df = (number of cells) - (fitted params).
    # The simulation samples n_per_level iid observations per level, so the
    # regression has 7 * n_per_level total observations and 3 fitted params.
    df = max(n_per_level * 7 - 3, 1)
    t_crit = _t_critical_one_sided(df, alpha=alpha)

    n_significant = 0
    for _ in range(n_simulations):
        # Sample n_per_level observations per level, average to get
        # observed level-mean. Then fit quadratic, get beta2_obs, divide
        # by SE to get t.
        observed_means = []
        for level in range(7):
            sigma = sigma_per_level[level]
            samples = [rng.gauss(true_means[level], sigma) for _ in range(n_per_level)]
            observed_means.append(sum(samples) / n_per_level)
        beta2_obs = _quadratic_beta2(observed_means)
        # SE(beta2) for OLS on per-level means with n iid samples per level:
        #   SE = (sigma_per_obs / sqrt(n_per_level)) * sqrt((X^T X)^{-1}_{3,3})
        # where (X^T X)^{-1}_{3,3} accounts for collinearity with the linear
        # term in the [1, x, x^2] design (pre-computed for 7 levels).
        avg_sigma = sum(sigma_per_level) / 7
        if n_per_level <= 0:
            continue
        sigma_mean = avg_sigma / math.sqrt(n_per_level)
        se_beta2 = sigma_mean * math.sqrt(_XTX_INV_22_7LEVEL)
        if se_beta2 <= 0:
            continue
        t_stat = beta2_obs / se_beta2
        # One-sided test: reject when t < -t_crit (since H1: beta2 < 0).
        if t_stat < -t_crit:
            n_significant += 1

    return PowerResult(
        n_per_level=n_per_level,
        beta2_assumed=beta2_assumed,
        icc=icc,
        n_simulations=n_simulations,
        power=n_significant / n_simulations,
        n_significant=n_significant,
        alpha=alpha,
    )


def find_min_n(
    beta2_assumed: float,
    sigma_per_level: list[float],
    icc: float = 0.20,
    target_power: float = 0.80,
    n_min: int = 5,
    n_max: int = 200,
    n_simulations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[int | None, list[PowerResult]]:
    """Binary search for the smallest n_per_level such that power >= target_power.

    Returns (recommended_n, search_trace). recommended_n is None if even
    n_max doesn't reach the target power; the caller should report this
    as an MDE-not-achievable result and consider relaxing target_power
    or the assumed beta2.
    """
    trace: list[PowerResult] = []

    # First confirm the upper bound reaches target. If not, bail.
    upper = power_at_n(
        n_per_level=n_max, beta2_assumed=beta2_assumed,
        sigma_per_level=sigma_per_level, icc=icc, n_simulations=n_simulations,
        seed=seed, alpha=alpha,
    )
    trace.append(upper)
    if upper.power < target_power:
        return None, trace

    lo, hi = n_min, n_max
    best_n = n_max
    while lo <= hi:
        mid = (lo + hi) // 2
        result = power_at_n(
            n_per_level=mid, beta2_assumed=beta2_assumed,
            sigma_per_level=sigma_per_level, icc=icc, n_simulations=n_simulations,
            seed=seed, alpha=alpha,
        )
        trace.append(result)
        if result.power >= target_power:
            best_n = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return best_n, trace
