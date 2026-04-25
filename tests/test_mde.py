"""Tests for the minimum-detectable-effect (MDE) computation.

Spec: affect-battery-task-difficulty-calibration::scoring-pipeline::
"Minimum-detectable-effect reporting".

Tasks 9.1 + 9.2 from
`specs`:
`compute_mde(baseline_acc, n, alpha=0.05, power=0.80)` returns the
smallest manipulation-check delta (as a fraction 0.0-1.0) detectable
under a two-sided binomial proportion test at the given α and power.

Sanity anchor: when baseline=0.5 and n=50, MDE should be around 0.28
(from textbook two-proportion power tables / G*Power). When baseline
moves to 0.75, MDE drops because variance p(1-p) is smaller.
"""

import math

import pytest

from src.analysis.mde import compute_mde


class TestComputeMdeSanity:
    """MDE should land in the neighborhood of published two-proportion
    tables. Exact agreement with any specific tool is not required —
    normal-approximation formulas vary slightly in whether they use the
    baseline variance, pooled variance, or worst-case p=0.5."""

    def test_baseline_half_n_fifty(self):
        # p=0.5 maximizes variance; MDE should be around 0.28 for n=50.
        mde = compute_mde(baseline_acc=0.5, n=50)
        # Wide tolerance band — the point is the order of magnitude and
        # the structure, not any one tool's exact figure.
        assert 0.22 <= mde <= 0.32, f"MDE={mde} outside expected band [0.22, 0.32]"

    def test_baseline_quarter_or_three_quarters_symmetric(self):
        # Variance p(1-p) is symmetric about p=0.5, so MDE at 0.25 ≈ MDE at 0.75.
        mde_low = compute_mde(baseline_acc=0.25, n=50)
        mde_high = compute_mde(baseline_acc=0.75, n=50)
        assert math.isclose(mde_low, mde_high, abs_tol=1e-9)

    def test_larger_n_shrinks_mde(self):
        small_n = compute_mde(baseline_acc=0.75, n=25)
        large_n = compute_mde(baseline_acc=0.75, n=200)
        # MDE falls as 1/sqrt(n); 8x n should cut MDE to roughly sqrt(8)x smaller.
        assert large_n < small_n
        assert large_n * 2 < small_n  # at least 2x reduction for 8x n

    def test_stricter_alpha_raises_mde(self):
        mde_05 = compute_mde(baseline_acc=0.75, n=50, alpha=0.05)
        mde_01 = compute_mde(baseline_acc=0.75, n=50, alpha=0.01)
        assert mde_01 > mde_05  # stricter => harder to detect small effects

    def test_higher_power_raises_mde(self):
        mde_80 = compute_mde(baseline_acc=0.75, n=50, power=0.80)
        mde_95 = compute_mde(baseline_acc=0.75, n=50, power=0.95)
        assert mde_95 > mde_80  # higher power => need bigger effect to reliably detect


class TestComputeMdeEdgeCases:
    def test_n_must_be_positive(self):
        with pytest.raises(ValueError, match="n"):
            compute_mde(baseline_acc=0.5, n=0)
        with pytest.raises(ValueError, match="n"):
            compute_mde(baseline_acc=0.5, n=-10)

    def test_baseline_out_of_range(self):
        with pytest.raises(ValueError, match="baseline_acc"):
            compute_mde(baseline_acc=-0.01, n=50)
        with pytest.raises(ValueError, match="baseline_acc"):
            compute_mde(baseline_acc=1.01, n=50)

    def test_baseline_at_boundary_returns_zero_or_tiny(self):
        # p=0 or p=1 gives zero variance; MDE formula yields 0.
        # The function should return 0.0 (mathematically defined), not raise —
        # the calling code can then annotate that n is insufficient.
        mde_zero = compute_mde(baseline_acc=0.0, n=50)
        mde_one = compute_mde(baseline_acc=1.0, n=50)
        assert mde_zero == 0.0
        assert mde_one == 0.0

    def test_alpha_must_be_valid(self):
        with pytest.raises(ValueError, match="alpha"):
            compute_mde(baseline_acc=0.5, n=50, alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            compute_mde(baseline_acc=0.5, n=50, alpha=1.0)

    def test_power_must_be_valid(self):
        with pytest.raises(ValueError, match="power"):
            compute_mde(baseline_acc=0.5, n=50, power=0.0)
        with pytest.raises(ValueError, match="power"):
            compute_mde(baseline_acc=0.5, n=50, power=1.0)
