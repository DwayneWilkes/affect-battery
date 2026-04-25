"""Asymmetry-compute primitives.

Per asymmetry-contrast spec "Asymmetry conventions" + "Paired asymmetry
computation" + "Aggregation rule": for each (positive, negative) effect-
size pair, compute:
  - ratio = |neg_effect| / |pos_effect|  (None if pos near zero)
  - difference = |neg_effect| - |pos_effect|
Aggregation across pairs:
  - ratios: geometric mean
  - differences: arithmetic mean
"""

from __future__ import annotations

import math


class TestComputePair:
    def test_paired_ratio_and_difference(self):
        from src.analysis.asymmetry import compute_pair

        result = compute_pair(pos_effect=0.3, neg_effect=-0.6)
        assert result["ratio"] == 2.0
        assert math.isclose(result["difference"], 0.3, abs_tol=1e-9)

    def test_near_zero_pos_returns_null_ratio(self):
        from src.analysis.asymmetry import compute_pair

        # |pos| < epsilon -> ratio is None
        result = compute_pair(pos_effect=0.001, neg_effect=-0.6)
        assert result["ratio"] is None
        # Difference still computed
        assert result["difference"] > 0.5

    def test_handles_signs_via_abs(self):
        from src.analysis.asymmetry import compute_pair

        # neg_effect provided as positive number => still computed via abs
        result = compute_pair(pos_effect=0.5, neg_effect=0.5)
        assert result["ratio"] == 1.0
        assert result["difference"] == 0.0


class TestComputeAggregate:
    def test_geometric_mean_ratios(self):
        from src.analysis.asymmetry import compute_aggregate

        pairs = [
            {"ratio": 2.0, "difference": 0.3},
            {"ratio": 4.0, "difference": 0.4},
            {"ratio": 8.0, "difference": 0.5},
        ]
        agg = compute_aggregate(pairs)
        # geometric mean of [2, 4, 8] = 4
        assert math.isclose(agg["ratio_geomean"], 4.0, rel_tol=1e-6)
        # arithmetic mean of differences = 0.4
        assert math.isclose(agg["difference_mean"], 0.4, rel_tol=1e-6)

    def test_geometric_mean_skips_null_ratios(self):
        from src.analysis.asymmetry import compute_aggregate

        pairs = [
            {"ratio": 2.0, "difference": 0.3},
            {"ratio": None, "difference": 0.5},
            {"ratio": 8.0, "difference": 0.4},
        ]
        agg = compute_aggregate(pairs)
        # Geomean of [2, 8] = 4
        assert math.isclose(agg["ratio_geomean"], 4.0, rel_tol=1e-6)
        # Differences average all three
        assert math.isclose(agg["difference_mean"], 0.4, rel_tol=1e-6)

    def test_empty_pairs_returns_null_geomean(self):
        from src.analysis.asymmetry import compute_aggregate

        agg = compute_aggregate([])
        assert agg["ratio_geomean"] is None
        assert agg["difference_mean"] is None
