"""Per-model H4 decision rule (7-row table).

Per asymmetry-contrast spec "Per-model H4 decision rule": map an
(aggregate, p_value, mde) triple to one of 7 verdict rows: supported,
below-MDE, inconclusive, null, reverse, near-1, degenerate.
"""

from __future__ import annotations


class TestPerModelVerdict:
    def test_decision_rule_covers_all_seven_rows(self):
        """Per previously near-1 and null had overlapping
        bands and no equivalence-test signal, so null was unreachable. Now
        they form a real partition driven by p_equivalence_under_alpha."""
        from src.analysis.asymmetry import per_model_verdict

        # supported: p < alpha and ratio > 1
        assert per_model_verdict(
            {"ratio_geomean": 2.0, "difference_mean": 0.3},
            p_value=0.001, mde=0.10,
        ) == "supported"

        # reverse: p < alpha and ratio < 1
        assert per_model_verdict(
            {"ratio_geomean": 0.5, "difference_mean": -0.2},
            p_value=0.001, mde=0.10,
        ) == "reverse"

        # degenerate: ratio is None
        assert per_model_verdict(
            {"ratio_geomean": None, "difference_mean": 0.3},
            p_value=0.001, mde=0.10,
        ) == "degenerate"

        # null: equivalence test under alpha AND ratio in tight band
        assert per_model_verdict(
            {"ratio_geomean": 1.02, "difference_mean": 0.005,
             "p_equivalence_under_alpha": True},
            p_value=0.20, mde=0.10,
        ) == "null"

        # near-1: ratio in wider band around 1.0, no equivalence signal
        assert per_model_verdict(
            {"ratio_geomean": 1.05, "difference_mean": 0.01},
            p_value=0.5, mde=0.10,
        ) == "near-1"

        # below-MDE: ratio > 1, observed magnitude < MDE, p > alpha
        assert per_model_verdict(
            {"ratio_geomean": 1.5, "difference_mean": 0.05},
            p_value=0.20, mde=0.10,
        ) == "below-MDE"

        # inconclusive: ratio > 1, magnitude not below MDE, p > alpha
        assert per_model_verdict(
            {"ratio_geomean": 1.5, "difference_mean": 0.20},
            p_value=0.20, mde=0.10,
        ) == "inconclusive"

    def test_null_unreachable_without_equivalence_signal(self):
        """A ratio sitting at exactly 1.0 with non-significant p but NO
        equivalence signal should fall to near-1, not null."""
        from src.analysis.asymmetry import per_model_verdict

        assert per_model_verdict(
            {"ratio_geomean": 1.00, "difference_mean": 0.0},
            p_value=0.50, mde=0.10,
        ) == "near-1"
