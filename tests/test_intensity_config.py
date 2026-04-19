"""Tests for configs/intensity_levels.yaml loading.

Enforces two ordered intensity axes for Experiment 3a (per spec + GAPS.md):
- primary_valence_axis: rater-validated 7 levels, length-matched, signed valence
- replication_arousal_axis: EA16-EA22 from Li et al. 2024 (arXiv:2312.11111)
  in authors' published arousal rank order (Table 19).
"""

from pathlib import Path

import pytest
import yaml


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "intensity_levels.yaml"


@pytest.fixture(scope="module")
def config():
    assert CONFIG_PATH.exists(), f"Missing config: {CONFIG_PATH}"
    return yaml.safe_load(CONFIG_PATH.read_text())


class TestPrimaryAxis:
    def test_primary_axis_present(self, config):
        assert "primary_valence_axis" in config

    def test_primary_has_seven_levels(self, config):
        axis = config["primary_valence_axis"]
        assert len(axis) == 7

    def test_primary_levels_ordered_one_to_seven(self, config):
        levels = [entry["level"] for entry in config["primary_valence_axis"]]
        assert levels == [1, 2, 3, 4, 5, 6, 7]

    def test_primary_every_entry_has_required_fields(self, config):
        for entry in config["primary_valence_axis"]:
            for field in ("level", "label", "feedback_text", "provenance"):
                assert field in entry, f"Missing {field} in {entry}"
                assert entry[field], f"Empty {field} in {entry}"

    def test_primary_length_matched(self, config):
        """Primary axis texts must be length-matched within 20% (GAPS spec rule)."""
        lengths = [
            len(entry["feedback_text"].split())
            for entry in config["primary_valence_axis"]
        ]
        ratio = max(lengths) / min(lengths)
        assert ratio <= 1.20, (
            f"Primary axis length ratio {ratio:.2f} exceeds 20%. Lengths: {lengths}"
        )


class TestReplicationAxis:
    """EA16-EA22 from Li et al. 2024 Figure 6(c) / Table 19, verbatim."""

    EA_VERBATIM = {
        "EA22": "A musician is playing passionately.",
        "EA17": "A child is laughing joyfully.",
        "EA20": "A bird is singing cheerfully.",
        "EA18": "A dog is barking angrily.",
        "EA21": "A girl is humming dreamily.",
        "EA16": "A baby is crying sadly.",
        "EA19": "A cat is purring contentedly.",
    }
    # Authors' arousal rank order (highest -> lowest arousal)
    EA_RANK_ORDER = ["EA22", "EA17", "EA20", "EA18", "EA21", "EA16", "EA19"]

    def test_replication_axis_present(self, config):
        assert "replication_arousal_axis" in config

    def test_replication_has_seven_levels(self, config):
        assert len(config["replication_arousal_axis"]) == 7

    def test_replication_levels_ordered_one_to_seven(self, config):
        levels = [entry["level"] for entry in config["replication_arousal_axis"]]
        assert levels == [1, 2, 3, 4, 5, 6, 7]

    def test_replication_stimuli_verbatim(self, config):
        """Stimuli must match Li et al. 2024 verbatim (no paraphrasing)."""
        for entry in config["replication_arousal_axis"]:
            ea_id = entry.get("source_id")
            assert ea_id in self.EA_VERBATIM, f"Unknown EA id: {ea_id}"
            assert entry["feedback_text"] == self.EA_VERBATIM[ea_id], (
                f"{ea_id} text mismatch; must be verbatim per spec"
            )

    def test_replication_ranked_by_authors_arousal_order(self, config):
        """Level 1 = highest-arousal per Table 19; Level 7 = lowest."""
        observed = [entry["source_id"] for entry in config["replication_arousal_axis"]]
        assert observed == self.EA_RANK_ORDER

    def test_replication_provenance_cites_li_2024(self, config):
        for entry in config["replication_arousal_axis"]:
            prov = entry["provenance"]
            assert "2312.11111" in prov or "Li et al." in prov, (
                f"Replication axis provenance must cite Li et al. 2024: {prov}"
            )


class TestAxisIndependence:
    """The two axes test different claims (signed valence vs unsigned arousal).
    They must be independently addressable by ExperimentConfig."""

    def test_two_distinct_axes(self, config):
        assert "primary_valence_axis" in config
        assert "replication_arousal_axis" in config

    def test_neither_axis_shares_stimuli_with_the_other(self, config):
        """Sanity: no copy-paste between axes."""
        primary = {e["feedback_text"] for e in config["primary_valence_axis"]}
        replication = {e["feedback_text"] for e in config["replication_arousal_axis"]}
        assert primary.isdisjoint(replication)
