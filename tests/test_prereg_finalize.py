"""Tasks 2.0a + 2.1 + 2.2 Red — OSF pre-reg lifecycle helper.

Per design.md D4 + power-analysis spec OSF gate:
- compute_prereg_sha(yaml_path): SHA-256 over canonicalized YAML.
- finalize_v1(v0_path, variance_probe, base_decision, output_path):
  reads v0 YAML + probe outputs + base-feasibility decision; emits
  v1 YAML with probe-grounded MDEs + base-decision-aware H4 status +
  amendment_chain entry + computed SHA.
- prepare_for_submit(yaml_path, osf_url): records osf_url + final SHA;
  writes amendment_chain entry.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.prereg.finalize import (
    compute_prereg_sha,
    finalize_v1,
    prepare_for_submit,
)


def _v0_yaml(path: Path) -> Path:
    """Minimal v0 YAML mirroring configs/osf_prereg_v1.yaml structure."""
    path.write_text(yaml.safe_dump({
        "osf_url": None,
        "pre_registration_sha": None,
        "preregistration_narrative": "test narrative",
        "hypotheses": [
            {"id": "H1", "mde": 0.40, "mde_source": "paper-prior-default",
             "alpha": 0.01, "target_power": 0.80,
             "correction_family": "primary", "test_direction": "two-sided",
             "formula": "..."},
            {"id": "H4", "mde": 1.45, "mde_source": "paper-prior-default",
             "alpha": 0.01, "target_power": 0.80,
             "correction_family": "primary", "test_direction": "one-sided",
             "formula": "..."},
        ],
        "correction_families": {
            "primary": ["H1", "H4"],
            "secondary": [],
            "primary_method": "holm-bonferroni",
            "secondary_method": "benjamini-hochberg",
        },
        "stopping_rules": [],
        "amendment_chain": [],
    }, sort_keys=True))
    return path


class TestComputePreregSha:
    def test_sha_is_64_hex_chars(self, tmp_path):
        path = _v0_yaml(tmp_path / "v0.yaml")
        sha = compute_prereg_sha(path)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_sha_stable_across_runs(self, tmp_path):
        path = _v0_yaml(tmp_path / "v0.yaml")
        a = compute_prereg_sha(path)
        b = compute_prereg_sha(path)
        assert a == b


class TestFinalizeV1:
    def test_updates_mde_from_observed(self, tmp_path):
        v0 = _v0_yaml(tmp_path / "v0.yaml")
        v1 = tmp_path / "v1.yaml"
        finalize_v1(
            v0_path=v0,
            output_path=v1,
            observed_effect_sizes={"H1": 0.55},  # observed > default 0.40
            base_feasibility_verdict="pass",
        )
        data = yaml.safe_load(v1.read_text())
        h1 = next(h for h in data["hypotheses"] if h["id"] == "H1")
        assert h1["mde"] == 0.55
        assert h1["mde_source"] == "pilot_observed"

    def test_keeps_default_when_observed_below(self, tmp_path):
        v0 = _v0_yaml(tmp_path / "v0.yaml")
        v1 = tmp_path / "v1.yaml"
        finalize_v1(
            v0_path=v0,
            output_path=v1,
            observed_effect_sizes={"H1": 0.20},  # below default 0.40
            base_feasibility_verdict="pass",
        )
        data = yaml.safe_load(v1.read_text())
        h1 = next(h for h in data["hypotheses"] if h["id"] == "H1")
        assert h1["mde"] == 0.40
        assert h1["mde_source"] == "default"

    def test_base_feasibility_fail_demotes_h4(self, tmp_path):
        v0 = _v0_yaml(tmp_path / "v0.yaml")
        v1 = tmp_path / "v1.yaml"
        finalize_v1(
            v0_path=v0,
            output_path=v1,
            observed_effect_sizes={},
            base_feasibility_verdict="fail",
        )
        data = yaml.safe_load(v1.read_text())
        h4 = next(h for h in data["hypotheses"] if h["id"] == "H4")
        assert h4["correction_family"] == "exploratory"

    def test_amendment_chain_appends(self, tmp_path):
        v0 = _v0_yaml(tmp_path / "v0.yaml")
        v1 = tmp_path / "v1.yaml"
        finalize_v1(
            v0_path=v0,
            output_path=v1,
            observed_effect_sizes={"H1": 0.55},
            base_feasibility_verdict="pass",
        )
        data = yaml.safe_load(v1.read_text())
        assert len(data["amendment_chain"]) == 1
        entry = data["amendment_chain"][0]
        assert "sha" in entry
        assert "rationale" in entry
        assert "timestamp" in entry

    def test_pre_registration_sha_set(self, tmp_path):
        v0 = _v0_yaml(tmp_path / "v0.yaml")
        v1 = tmp_path / "v1.yaml"
        finalize_v1(
            v0_path=v0,
            output_path=v1,
            observed_effect_sizes={},
            base_feasibility_verdict="pass",
        )
        data = yaml.safe_load(v1.read_text())
        assert data["pre_registration_sha"] is not None
        assert len(data["pre_registration_sha"]) == 64


class TestPrepareForSubmit:
    def test_records_osf_url(self, tmp_path):
        v1 = tmp_path / "v1.yaml"
        v1.write_text(yaml.safe_dump({
            "osf_url": None,
            "pre_registration_sha": None,
            "amendment_chain": [],
        }, sort_keys=True))
        prepare_for_submit(v1, osf_url="https://osf.io/abcde")
        data = yaml.safe_load(v1.read_text())
        assert data["osf_url"] == "https://osf.io/abcde"
        assert data["pre_registration_sha"] is not None
