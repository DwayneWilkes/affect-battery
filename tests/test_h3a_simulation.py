"""Tests for the H3a simulation-based power analysis."""
from __future__ import annotations

import pytest

from src.power.h3a_simulation import (
    _XTX_INV_22_7LEVEL,
    _quadratic_beta2,
    _xtx_inv_diag_for_quadratic,
    find_min_n,
    power_at_n,
)


class TestQuadraticOLS:
    def test_recovers_known_quadratic(self):
        """A perfect inverted-U y = -x^2 + 8x has c = -1; OLS recovers it."""
        curve = [-(x ** 2) + 8 * x for x in range(1, 8)]
        c = _quadratic_beta2(curve)
        assert c == pytest.approx(-1.0, abs=1e-6)

    def test_constant_curve_gives_zero_quadratic(self):
        """A flat curve has zero quadratic coefficient."""
        curve = [0.5] * 7
        c = _quadratic_beta2(curve)
        assert c == pytest.approx(0.0, abs=1e-6)


class TestDesignMatrixConstant:
    def test_matches_hand_computation(self):
        """For x = 1..7, (X^T X)^{-1}_{3,3} = 196/16464."""
        assert _XTX_INV_22_7LEVEL == pytest.approx(196 / 16464, rel=1e-6)

    def test_helper_matches_constant(self):
        assert _xtx_inv_diag_for_quadratic(7) == pytest.approx(_XTX_INV_22_7LEVEL)


class TestPowerAtN:
    def test_null_effect_type_one_error_at_alpha(self):
        """Under H0 (beta2=0), one-sided test should reject ~5% at alpha=0.05.

        Tolerance is generous because Monte Carlo with 2000 sims has SE
        around sqrt(0.05 * 0.95 / 2000) ≈ 0.005, so 95% CI is ±0.01.
        """
        result = power_at_n(
            n_per_level=30,
            beta2_assumed=0.0,
            sigma_per_level=[0.10] * 7,
            n_simulations=2000,
            alpha=0.05,
        )
        assert 0.02 < result.power < 0.08, (
            f"Type I error {result.power} outside [0.02, 0.08]; "
            "the one-sided test may be miscalibrated."
        )

    def test_real_effect_increases_power_with_n(self):
        """At the same effect size and variance, larger n gives more power."""
        small = power_at_n(
            n_per_level=10, beta2_assumed=-0.005,
            sigma_per_level=[0.10] * 7, n_simulations=300, seed=1,
        )
        large = power_at_n(
            n_per_level=60, beta2_assumed=-0.005,
            sigma_per_level=[0.10] * 7, n_simulations=300, seed=1,
        )
        assert large.power > small.power, (
            f"power non-monotonic in n: n=10 -> {small.power}, n=60 -> {large.power}"
        )

    def test_more_variance_reduces_power(self):
        """At the same effect and n, larger variance gives less power."""
        low_var = power_at_n(
            n_per_level=30, beta2_assumed=-0.005,
            sigma_per_level=[0.10] * 7, n_simulations=300, seed=1,
        )
        high_var = power_at_n(
            n_per_level=30, beta2_assumed=-0.005,
            sigma_per_level=[0.20] * 7, n_simulations=300, seed=1,
        )
        assert low_var.power > high_var.power


class TestFindMinN:
    def test_finds_n_for_tractable_effect(self):
        """A moderate effect size should be detectable at modest n."""
        n, trace = find_min_n(
            beta2_assumed=-0.011,
            sigma_per_level=[0.10] * 7,
            target_power=0.80,
            n_simulations=300,
            n_min=5, n_max=200,
        )
        assert n is not None
        assert 5 <= n <= 200

    def test_returns_none_for_undetectable_effect(self):
        """A very small effect with small n_max should return None."""
        n, _ = find_min_n(
            beta2_assumed=-0.0001,  # extremely small
            sigma_per_level=[0.20] * 7,
            target_power=0.80,
            n_simulations=200,
            n_min=5, n_max=20,  # constrained search ceiling
        )
        assert n is None
