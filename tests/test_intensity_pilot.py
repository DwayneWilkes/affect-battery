"""Intensity-axis pilot protocol (Krippendorff validation).

Per conditioning-protocol spec "Two intensity axes for Experiment 3a" +
"Intensity-axis pilot-as-gate for Exp 3a": run a pilot collecting
ratings from 3 raters across N items at 7 intensity levels. Compute
Krippendorff α per pair + overall. If α < 0.6 for any adjacent-level
pair, trigger the collapse scenario (recommend axis collapse to fewer
levels).
"""

from __future__ import annotations

import pytest


class TestIntensityPilot:
    def test_pilot_requires_3_raters(self):
        from src.probes.intensity_pilot import run_intensity_pilot

        # 7 levels x 5 items, 3 raters with high agreement
        ratings = {
            "rater_1": [1, 2, 3, 4, 5, 6, 7] * 5,
            "rater_2": [1, 2, 3, 4, 5, 6, 7] * 5,
            "rater_3": [1, 2, 2, 4, 5, 6, 7] * 5,
        }
        result = run_intensity_pilot(ratings)
        assert result["n_raters"] == 3
        assert "alpha_overall" in result
        assert result["alpha_overall"] >= 0.6
        assert result["decision"] in {"proceed", "collapse", "restructure"}

    def test_low_alpha_triggers_collapse(self):
        from src.probes.intensity_pilot import run_intensity_pilot

        # Disagreement: rater 3 inverted on adjacent levels
        ratings = {
            "rater_1": [1, 2, 3, 4, 5, 6, 7] * 5,
            "rater_2": [3, 1, 6, 2, 7, 4, 5] * 5,
            "rater_3": [7, 6, 5, 4, 3, 2, 1] * 5,
        }
        result = run_intensity_pilot(ratings)
        assert result["decision"] == "collapse"
        assert result["alpha_overall"] < 0.6

    def test_fewer_than_3_raters_raises(self):
        from src.probes.intensity_pilot import run_intensity_pilot

        with pytest.raises(ValueError, match="3 raters"):
            run_intensity_pilot({"rater_1": [1, 2, 3], "rater_2": [1, 2, 3]})
