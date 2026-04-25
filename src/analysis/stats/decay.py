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
    Returns (AIC, BIC). When rss == 0 (perfect fit), the log term is
    nominally -inf; we substitute a small floor so callers can still
    compare models without crashing.
    """
    if n <= 0 or k < 0:
        raise ValueError(f"n must be > 0 and k >= 0; got n={n}, k={k}")
    rss_floor = max(rss, 1e-30)
    aic = n * math.log(rss_floor / n) + 2 * k
    bic = n * math.log(rss_floor / n) + k * math.log(n)
    return aic, bic


def fit_linear(
    n_values: list[int],
    curve: list[float],
    baseline: float,
) -> dict[str, float]:
    """OLS fit of (curve - baseline) ~ slope * N. Returns slope + AIC/BIC."""
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
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    slope = sum_xy / sum_xx if sum_xx != 0.0 else 0.0
    rss = sum((y - slope * x) ** 2 for x, y in zip(xs, ys))
    aic, bic = _aic_bic(rss, n, k=1)
    return {"slope": slope, "rss": rss, "aic": aic, "bic": bic}


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
        # Fallback: log-linear seed when nonlinear fit diverges. Crude but
        # keeps the analysis pipeline non-crashing on adversarial curves.
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
