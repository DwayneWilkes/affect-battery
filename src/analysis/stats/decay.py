"""Decay-model fits for Exp 2 recovery curves.

Per persistence-dynamics spec "Decay model fit and comparison" + tasks.md
Task 5.4: fit both an exponential model and a linear-proportional model
to a turn-accuracy curve and report AIC + BIC for each so the consumer
can pick the winning shape.

Models
------
Exponential: acc(N) = baseline + amplitude * exp(-N / tau)
  amplitude carries sign (negative => below-baseline recovery, e.g.,
  strong-negative arm); tau > 0 is the decay timescale (in N units).

Linear-proportional: acc(N) = baseline + slope * N
  Closed-form OLS for slope; AIC/BIC from residual sum-of-squares.

Both models are fit to the residual `curve - baseline`. Information
criteria use the Gaussian-likelihood form: AIC = n*ln(RSS/n) + 2*k,
BIC = n*ln(RSS/n) + k*ln(n), with k = number of free parameters
(exponential: 2 [amplitude, tau], linear: 1 [slope]).
"""

from __future__ import annotations

import math

from scipy.optimize import curve_fit


def _aic_bic(rss: float, n: int, k: int) -> tuple[float, float]:
    """AIC + BIC under Gaussian residuals.

    rss = sum of squared residuals; n = sample size; k = free params.
    Returns (AIC, BIC). When rss == 0 (perfect fit) the log term is
    -inf; we return -inf rather than substituting a tiny floor because
    floor-based AIC inflates the difference between near-perfect and
    realistic-noise fits asymmetrically (review-finding #4).
    """
    if n <= 0 or k < 0:
        raise ValueError(f"n must be > 0 and k >= 0; got n={n}, k={k}")
    if rss <= 0.0:
        return -math.inf, -math.inf
    aic = n * math.log(rss / n) + 2 * k
    bic = n * math.log(rss / n) + k * math.log(n)
    return aic, bic


def fit_linear(
    n_values: list[int],
    curve: list[float],
    baseline: float,
) -> dict[str, float]:
    """OLS fit of (curve - baseline) ~ intercept + slope * N.

    Includes a free intercept so this competes fairly with
    fit_exponential's free amplitude on AIC/BIC. Without the intercept
    the line is forced through the origin in residual-space, inflating
    RSS and biasing model selection toward exponential
    (review-finding #3).
    """
    if len(n_values) != len(curve):
        raise ValueError(
            f"n_values ({len(n_values)}) and curve ({len(curve)}) must "
            f"have the same length"
        )
    n = len(n_values)
    if n < 2:
        raise ValueError(f"need >= 2 points to fit; got {n}")
    xs = [float(x) for x in n_values]
    ys = [c - baseline for c in curve]
    # OLS with intercept via closed-form normal equations.
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    slope = cov_xy / var_x if var_x != 0.0 else 0.0
    intercept = mean_y - slope * mean_x
    rss = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    aic, bic = _aic_bic(rss, n, k=2)  # intercept + slope = 2 free params
    return {"slope": slope, "intercept": intercept, "rss": rss, "aic": aic, "bic": bic}


def _exp_model(n, amplitude, tau):
    return amplitude * math.exp(-n / tau)


def fit_exponential(
    n_values: list[int],
    curve: list[float],
    baseline: float,
) -> dict[str, float]:
    """Nonlinear LS fit of (curve - baseline) ~ amplitude * exp(-N / tau)."""
    if len(n_values) != len(curve):
        raise ValueError(
            f"n_values ({len(n_values)}) and curve ({len(curve)}) must "
            f"have the same length"
        )
    n = len(n_values)
    if n < 3:
        raise ValueError(f"need >= 3 points to fit exponential; got {n}")

    xs = [float(x) for x in n_values]
    ys = [c - baseline for c in curve]

    # Initial guesses. amplitude ~ first residual; tau ~ midrange of N.
    amp0 = ys[0] if ys[0] != 0 else (-0.5 if curve[0] < baseline else 0.5)
    tau0 = max(1.0, (xs[-1] - xs[0]) / 2.0)

    def model(n_arr, amplitude, tau):
        # SciPy passes a numpy array; we don't import numpy here, so use
        # a list comprehension that works for both array-like inputs.
        return [_exp_model(nv, amplitude, tau) for nv in n_arr]

    try:
        popt, _ = curve_fit(
            model, xs, ys, p0=[amp0, tau0], maxfev=2000,
            bounds=([-1.0, 1e-3], [1.0, 1000.0]),
        )
        amplitude, tau = float(popt[0]), float(popt[1])
    except Exception:
        # Heuristic-seed fallback when curve_fit diverges. We do NOT
        # claim this is a refined fit — it's the initial guess, kept
        # so the pipeline doesn't crash on adversarial curves. Caller
        # should treat a fit with these exact bounds-edge values as
        # suspect (review-finding #15).
        amplitude, tau = amp0, tau0

    rss = sum((y - amplitude * math.exp(-x / tau)) ** 2 for x, y in zip(xs, ys))
    aic, bic = _aic_bic(rss, n, k=2)
    return {"amplitude": amplitude, "tau": tau, "rss": rss, "aic": aic, "bic": bic}


def compare_decay_models(
    n_values: list[int],
    curve: list[float],
    baseline: float,
) -> dict[str, dict[str, float]]:
    """Fit both exponential and linear models; return a dict keyed by
    'exponential' and 'linear', each value is the per-fit dict (with
    amplitude/tau or slope, plus rss/aic/bic).
    """
    return {
        "exponential": fit_exponential(n_values, curve, baseline),
        "linear": fit_linear(n_values, curve, baseline),
    }
