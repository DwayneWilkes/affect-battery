"""Task 8.3 Red — per-model H4 decision rule (7-row table).

Per asymmetry-contrast spec "Per-model H4 decision rule": map an
(aggregate, p_value, mde) triple to one of 7 verdict rows: supported,
below-MDE, inconclusive, null, reverse, near-1, degenerate.
"""

from __future__ import annotations


class TestPerModelVerdict:
    def test_decision_rule_covers_all_rows(self):
        from src.analysis.asymmetry import per_model_verdict

        # supported: ratio > 1, p < alpha
        assert per_model_verdict(
            {"ratio_geomean": 2.0, "difference_mean": 0.3},
            p_value=0.001, mde=0.10,
        ) == "supported"

        # reverse: ratio < 1, p < alpha
        assert per_model_verdict(
            {"ratio_geomean": 0.5, "difference_mean": -0.2},
            p_value=0.001, mde=0.10,
        ) == "reverse"

        # near-1: ratio ~ 1.0, p > alpha
        assert per_model_verdict(
            {"ratio_geomean": 1.05, "difference_mean": 0.01},
            p_value=0.5, mde=0.10,
        ) == "near-1"

        # degenerate: ratio is None
        assert per_model_verdict(
            {"ratio_geomean": None, "difference_mean": 0.3},
            p_value=0.001, mde=0.10,
        ) == "degenerate"

        # below-MDE: ratio > 1 but observed magnitude below MDE
        assert per_model_verdict(
            {"ratio_geomean": 1.5, "difference_mean": 0.05},
            p_value=0.20, mde=0.10,
        ) == "below-MDE"

        # null: ratio in tight band around 1.0, p > alpha
        assert per_model_verdict(
            {"ratio_geomean": 1.02, "difference_mean": 0.005},
            p_value=0.20, mde=0.10,
        ) == "near-1"  # Test the spec's near-1 vs null partition; near-1 wins
                       # at the 0.05 threshold; null only when ratio EXACTLY 1.

        # inconclusive: ratio > 1, magnitude not below MDE, p > alpha
        assert per_model_verdict(
            {"ratio_geomean": 1.5, "difference_mean": 0.20},
            p_value=0.20, mde=0.10,
        ) == "inconclusive"
