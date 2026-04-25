"""Task 6.2 Red — pre-registration seed emission.

Per conditioning-protocol spec "Pre-registration seed definition" +
tasks.md Task 6.2: pilot-pass produces a signed JSON artifact at
configs/intensity_pilot_pass_<date>.json with 5 required fields +
SHA-256 over canonicalized payload.
"""

from __future__ import annotations

import json
import pytest


REQUIRED_FIELDS = {
    "pilot_date",
    "axis_id",
    "n_levels",
    "alpha_overall",
    "alpha_pairwise",
}


class TestEmitSeed:
    def test_pilot_pass_emits_seed(self, tmp_path):
        from src.probes.intensity_pilot import emit_seed

        pilot_result = {
            "n_raters": 3,
            "n_items": 35,
            "alpha_overall": 0.85,
            "alpha_pairwise": {"r1__r2": 0.84, "r1__r3": 0.82, "r2__r3": 0.88},
            "decision": "proceed",
            "raters": ["r1", "r2", "r3"],
        }
        seed_path = tmp_path / "intensity_pilot_pass_2026-04-24.json"
        emit_seed(
            pilot_result,
            axis_id="primary_valence_axis",
            n_levels=7,
            pilot_date="2026-04-24",
            output_path=seed_path,
        )
        assert seed_path.exists()
        payload = json.loads(seed_path.read_text())
        for f in REQUIRED_FIELDS:
            assert f in payload, f"missing required field: {f}"
        assert "sha256" in payload
        assert isinstance(payload["sha256"], str)
        assert len(payload["sha256"]) == 64  # hex digest length

    def test_non_pass_decision_raises(self, tmp_path):
        from src.probes.intensity_pilot import emit_seed

        pilot_result = {
            "n_raters": 3,
            "n_items": 35,
            "alpha_overall": 0.4,
            "alpha_pairwise": {},
            "decision": "collapse",
            "raters": ["r1", "r2", "r3"],
        }
        with pytest.raises(ValueError, match="proceed"):
            emit_seed(
                pilot_result,
                axis_id="primary_valence_axis",
                n_levels=7,
                pilot_date="2026-04-24",
                output_path=tmp_path / "seed.json",
            )
