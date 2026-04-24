"""Task 0.4 Red — OSF pre-registration YAML skeleton.

Per power-analysis/spec.md "OSF pre-registration top-level gate" +
design.md D10 (single versioned YAML for OSF pre-registration):
- configs/osf_prereg_v1.yaml exists.
- Has required top-level keys: hypotheses, correction_families,
  stopping_rules, preregistration_narrative, amendment_chain, osf_url.
- 7 hypotheses: H1, H1b-directional, H1b-TOST, H2, H3a, H3b, H3c, H4
  (paper §2 + design.md D3 grounded MDEs + power-analysis spec).
- Each hypothesis has: id, formula, mde, alpha, target_power,
  correction_family.
- Correction families list primary {H1, H2, H3a, H4, H1b-TOST} and
  secondary {H1b-directional, H3b, H3c}.
- pre_registration_sha is SHA-256 of canonicalized YAML.
"""

from pathlib import Path
import hashlib

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO / "configs" / "osf_prereg_v1.yaml"


def _load() -> dict:
    return yaml.safe_load(PREREG_PATH.read_text())


class TestOsfPreregFileExists:
    def test_yaml_file_exists(self):
        assert PREREG_PATH.exists(), (
            f"OSF pre-registration skeleton missing at {PREREG_PATH}"
        )


class TestRequiredTopLevelKeys:
    def test_has_hypotheses_key(self):
        data = _load()
        assert "hypotheses" in data
        assert isinstance(data["hypotheses"], list)

    def test_has_correction_families_key(self):
        data = _load()
        assert "correction_families" in data

    def test_has_stopping_rules_key(self):
        data = _load()
        assert "stopping_rules" in data

    def test_has_preregistration_narrative_key(self):
        data = _load()
        assert "preregistration_narrative" in data
        assert isinstance(data["preregistration_narrative"], str)

    def test_has_amendment_chain_key(self):
        data = _load()
        assert "amendment_chain" in data
        assert isinstance(data["amendment_chain"], list)

    def test_has_osf_url_key(self):
        data = _load()
        assert "osf_url" in data  # may be None until upload

    def test_has_pre_registration_sha_key(self):
        data = _load()
        assert "pre_registration_sha" in data


class TestHypothesisCoverage:
    REQUIRED_HYPOTHESES = {
        "H1", "H1b-directional", "H1b-TOST", "H2", "H3a", "H3b", "H3c", "H4"
    }

    def test_all_hypotheses_present(self):
        data = _load()
        ids = {h["id"] for h in data["hypotheses"]}
        assert ids == self.REQUIRED_HYPOTHESES, (
            f"Missing or extra hypotheses. Got {ids}, "
            f"expected {self.REQUIRED_HYPOTHESES}"
        )

    @pytest.mark.parametrize("field", [
        "id", "formula", "mde", "alpha", "target_power",
        "correction_family", "test_direction",
    ])
    def test_each_hypothesis_has_required_field(self, field):
        data = _load()
        for h in data["hypotheses"]:
            assert field in h, (
                f"Hypothesis {h.get('id', '?')!r} missing required field {field!r}"
            )


class TestCorrectionFamilies:
    PRIMARY = {"H1", "H2", "H3a", "H4", "H1b-TOST"}
    SECONDARY = {"H1b-directional", "H3b", "H3c"}

    def test_primary_family_membership(self):
        data = _load()
        primary = set(data["correction_families"].get("primary", []))
        assert primary == self.PRIMARY

    def test_secondary_family_membership(self):
        data = _load()
        secondary = set(data["correction_families"].get("secondary", []))
        assert secondary == self.SECONDARY

    def test_correction_methods_named(self):
        data = _load()
        cf = data["correction_families"]
        assert cf.get("primary_method") == "holm-bonferroni"
        assert cf.get("secondary_method") == "benjamini-hochberg"
