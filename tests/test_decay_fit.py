"""Task 5.4 Red — Exp 2 decay-model fits + AIC/BIC comparison.

Per persistence-dynamics spec "Decay model fit and comparison" +
scoring-pipeline spec "Decay-model fitting for H2": for an Exp 2 turn-
accuracy curve, fit BOTH an exponential model (acc(N) = baseline +
amplitude * exp(-N / tau)) and a linear-proportional model (acc(N) =
baseline + slope * N). Report AIC + BIC for each and let the consumer
choose the winning shape.
"""

from __future__ import annotations

import math


class TestExponentialFit:
    def test_fits_known_exponential(self):
        from src.analysis.stats.decay import fit_exponential

        # Curve generated from known parameters: baseline=0.0, A=-0.6, tau=4
        n_values = [1, 3, 5, 10]
        baseline = 0.8
        true_amp = -0.6
        true_tau = 4.0
        curve = [baseline + true_amp * math.exp(-n / true_tau) for n in n_values]

        fit = fit_exponential(n_values, curve, baseline=baseline)
        assert "amplitude" in fit
        assert "tau" in fit
        assert "aic" in fit
        assert "bic" in fit
        # Recovered amplitude + tau within tolerance for a clean curve
        assert fit["amplitude"] == math.fabs(true_amp) * (-1 if true_amp < 0 else 1) or abs(fit["amplitude"] - true_amp) < 0.05
        assert abs(fit["tau"] - true_tau) < 1.0


class TestLinearFit:
    def test_fits_linear_proportional(self):
        from src.analysis.stats.decay import fit_linear

        n_values = [1, 3, 5, 10]
        baseline = 0.8
        # acc(N) = 0.8 + 0.02 * N
        curve = [baseline + 0.02 * n for n in n_values]

        fit = fit_linear(n_values, curve, baseline=baseline)
        assert "slope" in fit
        assert "aic" in fit
        assert "bic" in fit
        assert abs(fit["slope"] - 0.02) < 1e-6


class TestExponentialAndLinearBothFit:
    def test_exponential_and_linear_both_fit(self):
        """compare_decay_models returns both fits + their AIC/BIC values."""
        from src.analysis.stats.decay import compare_decay_models

        n_values = [1, 3, 5, 10]
        baseline = 0.8
        # Slightly nonlinear curve (exponential decay back to baseline)
        curve = [0.2, 0.5, 0.65, 0.78]

        comparison = compare_decay_models(n_values, curve, baseline=baseline)
        assert "exponential" in comparison
        assert "linear" in comparison
        assert "aic" in comparison["exponential"]
        assert "aic" in comparison["linear"]
        assert "bic" in comparison["exponential"]
        assert "bic" in comparison["linear"]
