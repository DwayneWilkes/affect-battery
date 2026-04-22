"""Tests for multiple-comparisons correction functions.

Covers the policy laid out in the scoring-pipeline spec "Multiple-comparisons
correction policy" Requirement and design D8:

- Holm-Bonferroni (Holm 1979) for the manipulation-check family (FWER control)
- Benjamini-Hochberg (Benjamini & Hochberg 1995) for exploratory pairwise
  contrasts (FDR control)
"""

from __future__ import annotations

import math

import pytest

from src.analysis_corrections import apply_bh_correction, apply_holm_correction


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


class TestHolmCorrection:
    def test_hand_verified_four_pvals(self):
        """Hand-verified: Holm on [0.01, 0.04, 0.03, 0.005].

        Sort ascending: 0.005, 0.01, 0.03, 0.04 with m=4.
        Step q-values (cumulative max of (m-k+1)*p_(k)):
          k=1: 4 * 0.005 = 0.020
          k=2: max(0.020, 3 * 0.01) = 0.030
          k=3: max(0.030, 2 * 0.03) = 0.060
          k=4: max(0.060, 1 * 0.04) = 0.060
        Mapped back to input order [0.01, 0.04, 0.03, 0.005]:
          [0.030, 0.060, 0.060, 0.020]
        """
        pvals = [0.01, 0.04, 0.03, 0.005]
        expected = [0.030, 0.060, 0.060, 0.020]
        result = apply_holm_correction(pvals)
        assert len(result) == len(pvals)
        for got, want in zip(result, expected):
            assert math.isclose(got, want, abs_tol=1e-9)

    def test_preserves_input_order(self):
        pvals = [0.5, 0.01, 0.2, 0.001]
        result = apply_holm_correction(pvals)
        # smallest raw p (0.001, idx 3) should get smallest q
        assert result[3] == min(result)

    def test_q_values_cap_at_one(self):
        pvals = [0.6, 0.7, 0.8]
        result = apply_holm_correction(pvals)
        for q in result:
            assert q <= 1.0

    def test_q_values_gte_raw_p(self):
        pvals = [0.001, 0.01, 0.03, 0.04, 0.25]
        result = apply_holm_correction(pvals)
        for raw, q in zip(pvals, result):
            # Holm corrected q-value is always >= raw p
            assert q >= raw - 1e-12

    def test_single_pval_unchanged(self):
        assert apply_holm_correction([0.03]) == [0.03]

    def test_empty_list(self):
        assert apply_holm_correction([]) == []

    def test_all_equal_pvals(self):
        # All equal p = 0.02, m = 4 -> step values 4p, max(prev, 3p), etc.
        # Largest multiplier wins for all: 4 * 0.02 = 0.08 across the board.
        pvals = [0.02, 0.02, 0.02, 0.02]
        result = apply_holm_correction(pvals)
        for q in result:
            assert math.isclose(q, 0.08, abs_tol=1e-9)

    def test_invalid_pval_above_one_raises(self):
        with pytest.raises(ValueError):
            apply_holm_correction([0.5, 1.5])

    def test_invalid_negative_pval_raises(self):
        with pytest.raises(ValueError):
            apply_holm_correction([0.01, -0.01])

    def test_invalid_nan_raises(self):
        with pytest.raises(ValueError):
            apply_holm_correction([0.01, float("nan")])

    def test_output_length_matches_input(self):
        pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        assert len(apply_holm_correction(pvals)) == len(pvals)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


class TestBHCorrection:
    def test_hand_verified_four_pvals(self):
        """Hand-verified: BH on [0.01, 0.04, 0.03, 0.005].

        Sort ascending: 0.005, 0.01, 0.03, 0.04 with m=4.
        Raw step q_(k) = p_(k) * m / k:
          k=1: 0.005 * 4/1 = 0.020
          k=2: 0.010 * 4/2 = 0.020
          k=3: 0.030 * 4/3 = 0.040
          k=4: 0.040 * 4/4 = 0.040
        Enforce monotone non-decrease from highest rank down (BH step-up):
          [0.020, 0.020, 0.040, 0.040]
        Mapped back to input order [0.01, 0.04, 0.03, 0.005]:
          [0.020, 0.040, 0.040, 0.020]
        """
        pvals = [0.01, 0.04, 0.03, 0.005]
        expected = [0.020, 0.040, 0.040, 0.020]
        result = apply_bh_correction(pvals)
        for got, want in zip(result, expected):
            assert math.isclose(got, want, abs_tol=1e-9)

    def test_preserves_input_order(self):
        pvals = [0.5, 0.01, 0.2, 0.001]
        result = apply_bh_correction(pvals)
        # input idx 3 (0.001) should produce smallest q
        assert result[3] == min(result)

    def test_q_values_cap_at_one(self):
        pvals = [0.6, 0.7, 0.8, 0.95]
        result = apply_bh_correction(pvals)
        for q in result:
            assert q <= 1.0

    def test_q_values_gte_raw_p(self):
        pvals = [0.001, 0.01, 0.03, 0.04, 0.25]
        result = apply_bh_correction(pvals)
        for raw, q in zip(pvals, result):
            assert q >= raw - 1e-12

    def test_single_pval_unchanged(self):
        assert apply_bh_correction([0.03]) == [0.03]

    def test_empty_list(self):
        assert apply_bh_correction([]) == []

    def test_all_equal_pvals(self):
        # All equal p=0.02, m=4 -> q_(k) = 0.02 * 4 / k
        # raw: [0.08, 0.04, 0.0267, 0.02]
        # monotone non-decreasing from top down: take running min from rank m
        # running min: [0.02, 0.02, 0.02, 0.02]
        pvals = [0.02, 0.02, 0.02, 0.02]
        result = apply_bh_correction(pvals)
        for q in result:
            assert math.isclose(q, 0.02, abs_tol=1e-9)

    def test_invalid_pval_above_one_raises(self):
        with pytest.raises(ValueError):
            apply_bh_correction([0.5, 1.5])

    def test_invalid_negative_pval_raises(self):
        with pytest.raises(ValueError):
            apply_bh_correction([0.01, -0.01])

    def test_invalid_nan_raises(self):
        with pytest.raises(ValueError):
            apply_bh_correction([0.01, float("nan")])

    def test_output_length_matches_input(self):
        pvals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        assert len(apply_bh_correction(pvals)) == len(pvals)


# ---------------------------------------------------------------------------
# Cross-function sanity: both should give the same answer for a single pval
# ---------------------------------------------------------------------------


def test_holm_and_bh_agree_on_singleton():
    assert apply_holm_correction([0.04]) == apply_bh_correction([0.04])


def test_holm_more_conservative_than_bh_on_spec_six_family():
    """Per D8, Holm (FWER) is more conservative than BH (FDR) on the same family."""
    pvals = [0.001, 0.008, 0.02, 0.03, 0.04, 0.05]
    holm = apply_holm_correction(pvals)
    bh = apply_bh_correction(pvals)
    for h, b in zip(holm, bh):
        assert h >= b - 1e-12
