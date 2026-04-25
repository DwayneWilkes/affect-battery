"""Exp 3a quadratic fit + β₂ test.

Per scoring-pipeline spec "Quadratic-model fit for H3a" + power-analysis
spec H3a: fit accuracy ~ intensity_level + I(intensity_level^2).
β₂ < 0 (one-sided test) is the H3a-confirming inverted-U signature.
Report β₂ p-value + AIC/BIC vs the linear-only model so the consumer
can compare shapes.
"""

from __future__ import annotations

import pytest


class TestQuadraticFit:
    def test_inverted_u_recovers_negative_beta2(self):
        from src.analysis.exp3a import analyze_exp3a

        # Inverted-U at peak intensity 4: acc(L) = 0.5 - 0.05 * (L - 4)^2
        levels = [1, 2, 3, 4, 5, 6, 7]
        # accuracy_by_level: list of per-run accuracies (synthesize 4 per level)
        accuracy_by_level = {
            L: [0.5 - 0.05 * (L - 4) ** 2] * 4 for L in levels
        }
        result = analyze_exp3a(accuracy_by_level)

        assert "beta_2" in result
        assert "beta_2_p_one_sided" in result
        assert "beta_1" in result
        assert "beta_0" in result
        # Inverted-U => β₂ < 0
        assert result["beta_2"] < 0

    def test_linear_curve_has_near_zero_beta2(self):
        from src.analysis.exp3a import analyze_exp3a

        levels = [1, 2, 3, 4, 5, 6, 7]
        accuracy_by_level = {L: [0.3 + 0.05 * L] * 4 for L in levels}
        result = analyze_exp3a(accuracy_by_level)
        assert abs(result["beta_2"]) < 0.01

    def test_aic_bic_for_quadratic_and_linear(self):
        """Both AIC + BIC for quadratic model + AIC + BIC for linear model
        are reported so consumers can compare."""
        from src.analysis.exp3a import analyze_exp3a

        levels = [1, 2, 3, 4, 5, 6, 7]
        accuracy_by_level = {
            L: [0.5 - 0.05 * (L - 4) ** 2] * 4 for L in levels
        }
        result = analyze_exp3a(accuracy_by_level)

        assert "quadratic_aic" in result
        assert "quadratic_bic" in result
        assert "linear_aic" in result
        assert "linear_bic" in result


class TestQuadraticFitInputValidation:
    def test_too_few_levels_raises(self):
        from src.analysis.exp3a import analyze_exp3a

        with pytest.raises(ValueError, match="levels"):
            analyze_exp3a({1: [0.5], 2: [0.5]})  # need >= 3 distinct levels for quadratic
