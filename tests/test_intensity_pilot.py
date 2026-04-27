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


class TestSoloSeed:
    """`emit_solo_seed` writes a single-rater seed without Krippendorff α."""

    def test_writes_seed_with_irr_flag_false(self, tmp_path):
        import json
        from src.probes.intensity_pilot import emit_solo_seed

        seed_path = tmp_path / "seed.json"
        emit_solo_seed(
            rater_id="rater_PI",
            ratings=[1, 2, 3, 4, 5, 6, 7],
            axis_id="intensity_axis_v1",
            n_levels=7,
            pilot_date="2026-04-27",
            output_path=seed_path,
        )
        data = json.loads(seed_path.read_text())
        assert data["irr_validated"] is False
        assert data["solo_rater"] is True
        assert data["rater_id"] == "rater_PI"
        assert data["ratings"] == [1, 2, 3, 4, 5, 6, 7]
        assert data["axis_id"] == "intensity_axis_v1"
        assert data["n_levels"] == 7
        assert "sha256" in data

    def test_rejects_wrong_rating_count(self, tmp_path):
        from src.probes.intensity_pilot import emit_solo_seed

        with pytest.raises(ValueError, match="entries; expected 7"):
            emit_solo_seed(
                rater_id="x", ratings=[1, 2, 3],
                axis_id="x", n_levels=7, pilot_date="2026-04-27",
                output_path=tmp_path / "seed.json",
            )

    def test_rejects_out_of_range_rating(self, tmp_path):
        from src.probes.intensity_pilot import emit_solo_seed

        with pytest.raises(ValueError, match="must be integers in"):
            emit_solo_seed(
                rater_id="x", ratings=[1, 2, 3, 4, 5, 6, 9],  # 9 is out of [1,7]
                axis_id="x", n_levels=7, pilot_date="2026-04-27",
                output_path=tmp_path / "seed.json",
            )

    def test_multi_rater_seed_marks_validated(self, tmp_path):
        """The existing emit_seed (multi-rater) must set irr_validated=True."""
        import json
        from src.probes.intensity_pilot import emit_seed

        seed_path = tmp_path / "multi_seed.json"
        pilot_result = {
            "decision": "proceed",
            "alpha_overall": 0.85,
            "alpha_pairwise": {"r1__r2": 0.88, "r1__r3": 0.84, "r2__r3": 0.83},
        }
        emit_seed(
            pilot_result=pilot_result,
            axis_id="intensity_axis_v1",
            n_levels=7,
            pilot_date="2026-04-27",
            output_path=seed_path,
        )
        data = json.loads(seed_path.read_text())
        assert data["irr_validated"] is True
        assert data["solo_rater"] is False
        assert data["alpha_overall"] == 0.85
