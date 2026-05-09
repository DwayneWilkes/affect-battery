"""Tests for the H3b precision simulation.

The H3b prereg's interpretive thresholds are CI-width-based:
- `c_ci95_hi < 0.05` is the "bounded small" strong-claim region
- `c_ci95_hi >= 0.10` is the "uninformative" region

So the meaningful sample-size question is precision, not power: at what
`n_items` does the bootstrap CI half-width reliably stay below 0.05 (or
0.10)? `simulate_h3b_precision` answers that under a realistic per-item
p̂ distribution and the prereg's per-cell rep budget. `find_min_n_for_precision`
binary-searches for the smallest n meeting a chosen target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.power.h3b_simulation import (
    H3bPrecisionResult,
    find_min_n_for_precision,
    simulate_h3b_precision,
)


# Use a small, fast sim budget for tests; production runs should use
# larger n_simulations + n_bootstrap.
SMALL_SIMS = 80
SMALL_BOOTSTRAP = 500


def test_simulate_returns_result_with_expected_fields():
    """Smoke: the result object exposes the metrics the dashboard / runbook
    consume. CI-half-width metrics + reliability percentages."""
    p_hats = [0.4, 0.5, 0.5, 0.5, 0.6] * 4  # 20 items
    result = simulate_h3b_precision(
        n_items=20,
        p_hat_per_item=p_hats,
        n_reps_per_cell=20,
        n_simulations=SMALL_SIMS,
        n_bootstrap=SMALL_BOOTSTRAP,
        seed=42,
    )
    assert isinstance(result, H3bPrecisionResult)
    assert result.n_items == 20
    assert result.median_ci_half_width > 0
    assert 0.0 <= result.pct_below_0_05 <= 100.0
    assert 0.0 <= result.pct_below_0_10 <= 100.0
    # CI HW < 0.10 is a strict superset of CI HW < 0.05 → pct_below_0_10 >= pct_below_0_05.
    assert result.pct_below_0_10 >= result.pct_below_0_05


def test_larger_n_gives_tighter_ci():
    """Doubling n_items roughly halves the CI half-width (Bernoulli √n
    scaling). The simulation should reflect that monotonically."""
    p_hats = [0.5] * 100
    small = simulate_h3b_precision(
        n_items=10, p_hat_per_item=p_hats[:10], n_reps_per_cell=20,
        n_simulations=SMALL_SIMS, n_bootstrap=SMALL_BOOTSTRAP, seed=42,
    )
    big = simulate_h3b_precision(
        n_items=80, p_hat_per_item=p_hats[:80], n_reps_per_cell=20,
        n_simulations=SMALL_SIMS, n_bootstrap=SMALL_BOOTSTRAP, seed=42,
    )
    assert big.median_ci_half_width < small.median_ci_half_width
    # Should be roughly √(80/10) ≈ 2.83x tighter; allow generous tolerance.
    ratio = small.median_ci_half_width / big.median_ci_half_width
    assert 1.8 < ratio < 4.5, f"unexpected scaling ratio {ratio}"


def test_under_h0_median_c_estimate_near_zero():
    """When c_assumed=0, the median observed c across simulations should
    be near 0 (signal-free baseline)."""
    p_hats = [0.5] * 50
    result = simulate_h3b_precision(
        n_items=30, p_hat_per_item=p_hats, n_reps_per_cell=20,
        c_assumed=0.0,
        n_simulations=SMALL_SIMS, n_bootstrap=SMALL_BOOTSTRAP, seed=42,
    )
    # Within the per-sim noise floor; for n=30 items, |median c| < 0.02
    # is comfortably wide.
    assert abs(result.median_c_estimate) < 0.02


def test_under_h1_median_c_estimate_recovers_assumed():
    """When c_assumed=0.05, the median observed c should be near 0.05.
    This validates the simulation actually injects the assumed effect."""
    p_hats = [0.5] * 80
    result = simulate_h3b_precision(
        n_items=50, p_hat_per_item=p_hats, n_reps_per_cell=20,
        c_assumed=0.05,
        n_simulations=SMALL_SIMS, n_bootstrap=SMALL_BOOTSTRAP, seed=42,
    )
    # Should recover the assumed contrast within Monte Carlo noise.
    assert abs(result.median_c_estimate - 0.05) < 0.02


def test_p_hat_resampling_used_when_pool_smaller_than_n_items():
    """With n_items > len(p_hat pool), the simulation samples with
    replacement from the calibrated pool, so n can exceed the pool size
    without errors."""
    p_hats = [0.45, 0.50, 0.55]  # only 3 calibrated items
    result = simulate_h3b_precision(
        n_items=20, p_hat_per_item=p_hats, n_reps_per_cell=20,
        n_simulations=SMALL_SIMS, n_bootstrap=SMALL_BOOTSTRAP, seed=42,
    )
    # Should run without error and produce sensible results.
    assert result.n_items == 20
    assert 0 < result.median_ci_half_width < 1


def test_find_min_n_returns_smallest_n_meeting_threshold():
    """At the prereg's strong-claim threshold (CI HW < 0.05) with 80%
    reliability and n_reps=20, the sim should converge on a recommended
    n. Analytical back-of-envelope: ~14-18 items."""
    p_hats = [0.5] * 100
    n_required, trace = find_min_n_for_precision(
        p_hat_per_item=p_hats,
        target_ci_half_width=0.05,
        target_reliability=80.0,
        n_min=5,
        n_max=60,
        n_reps_per_cell=20,
        n_simulations=SMALL_SIMS,
        n_bootstrap=SMALL_BOOTSTRAP,
        seed=42,
    )
    assert n_required is not None
    assert 5 <= n_required <= 60
    # Trace must contain the chosen n's result for audit.
    chosen_match = [r for r in trace if r.n_items == n_required]
    assert chosen_match, f"chosen n={n_required} not in trace"
    assert chosen_match[0].pct_below_0_05 >= 80.0


def test_find_min_n_returns_none_when_n_max_too_small():
    """If even n_max can't meet a too-tight precision target, return None."""
    p_hats = [0.5] * 50
    n_required, trace = find_min_n_for_precision(
        p_hat_per_item=p_hats,
        target_ci_half_width=0.001,  # absurdly tight
        target_reliability=99.0,
        n_min=5,
        n_max=10,
        n_reps_per_cell=20,
        n_simulations=SMALL_SIMS,
        n_bootstrap=SMALL_BOOTSTRAP,
        seed=42,
    )
    assert n_required is None
    assert len(trace) >= 1


# ---------------------------------------------------------------------------
# BCa bootstrap (bias-corrected and accelerated)
# ---------------------------------------------------------------------------


from src.power.h3b_simulation import _bootstrap_ci_bca  # noqa: E402


def test_bca_reduces_to_percentile_when_symmetric():
    """When the per-item statistic is symmetric and centered, the bias
    correction z0 ≈ 0 and acceleration â ≈ 0, so BCa's adjusted quantile
    lookups collapse to plain percentile lookups."""
    rng = np.random.default_rng(42)
    statistic = rng.normal(loc=0.0, scale=1.0, size=200)
    point, lo, hi, z0, a_hat = _bootstrap_ci_bca(
        statistic, n_bootstrap=2000, rng=np.random.default_rng(7),
    )
    # Symmetric data → small bias correction and acceleration.
    assert abs(z0) < 0.10
    assert abs(a_hat) < 0.05
    # Point estimate is the sample mean.
    assert abs(point - statistic.mean()) < 1e-9
    # CI brackets the mean.
    assert lo < point < hi
    # Width is in the right ballpark for n=200, σ=1: SE ≈ 0.07, 95% CI ≈ ±0.14.
    assert 0.20 < (hi - lo) < 0.40


def test_bca_handles_constant_data_without_crashing():
    """All-identical statistic → degenerate jackknife (var=0). Must not
    raise ZeroDivisionError or produce NaN bounds; should return a
    degenerate CI at the constant value."""
    statistic = np.full(50, 0.5)
    point, lo, hi, z0, a_hat = _bootstrap_ci_bca(
        statistic, n_bootstrap=500, rng=np.random.default_rng(7),
    )
    assert point == 0.5
    assert lo == 0.5
    assert hi == 0.5
    # a_hat is undefined for constant data; convention is 0.
    assert a_hat == 0.0


def test_bca_skewed_data_produces_nontrivial_acceleration():
    """For a right-skewed distribution (log-normal here), the jackknife
    acceleration term is non-trivial — the whole point of BCa over plain
    percentile is to handle skew."""
    rng = np.random.default_rng(42)
    statistic = np.exp(rng.normal(0, 1, size=100))  # log-normal: right-skewed
    point, lo, hi, z0, a_hat = _bootstrap_ci_bca(
        statistic, n_bootstrap=2000, rng=np.random.default_rng(7),
    )
    assert abs(a_hat) > 0.01, f"acceleration should be non-trivial for log-normal, got {a_hat}"
    # CI should still bracket the sample mean.
    assert lo < point < hi


def test_simulation_uses_bca_internally():
    """Smoke: simulate_h3b_precision still produces sensible CI half-widths
    after switching the inner bootstrap to BCa."""
    p_hats = [0.5] * 50
    r = simulate_h3b_precision(
        n_items=30, p_hat_per_item=p_hats,
        n_reps_per_cell=20, n_simulations=SMALL_SIMS,
        n_bootstrap=SMALL_BOOTSTRAP, seed=42,
    )
    assert 0 < r.median_ci_half_width < 0.10
    # Under H0 with n=30, BCa half-width should be near percentile (~0.035).
    assert r.median_ci_half_width < 0.06
