"""Tests for the sweep launcher preflight checks.

Spec: affect-battery-task-difficulty-calibration::task-difficulty-calibration::
"Pre-flight go/no-go gate" + sweep-launcher gate-snapshot check +
seed-disjointness check + pre-registration verification.

Group 12: cmd_run (sweep launcher) MUST refuse to launch when:
    - no calibration report exists for the selected bank
    - gate config hash doesn't match the calibration report's recorded hash
      (drift detection: someone edited the gate config after calibration ran)
    - sweep seed range overlaps with the calibration report's seed range
    - gate config still carries the `REPLACE_BEFORE_CALIBRATION` sentinel in
      pre_registration_tag or pre_registration_sha

These are PURE refusal checks — they do not modify state. Tests exercise
the `preflight_checks` helper directly rather than driving cmd_run, since
cmd_run also spins up async clients.
"""

import json
from pathlib import Path

import pytest

from src.cli_preflight import (
    PreflightError,
    preflight_checks,
)


def _make_calibration_report(
    tmp_path: Path,
    bank_id: str = "arithmetic_hard_v1",
    config_hash: str = "abc123",
    seed_ranges: list[list[int]] | None = None,
    report_date: str = "2026-04-22",
) -> Path:
    """Write a minimal calibration report JSON to tmp_path."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    report_path = artifacts_dir / f"calibration-{bank_id}-{report_date}.json"
    report = {
        "bank_id": bank_id,
        "report_date": report_date,
        "gate_verdict": {
            "status": "PASS",
            "justification": "stub",
            "config_hash": config_hash,
        },
        "seed_ranges_used": seed_ranges or [[0, 4]],
    }
    report_path.write_text(json.dumps(report))
    return report_path


def _make_gate_config_file(
    tmp_path: Path,
    config_hash: str = "abc123",
    pre_reg_tag: str = "gate-prereg-arithmetic_hard_v1-2026-04-22",
    pre_reg_sha: str = "a" * 40,
) -> Path:
    """Write a mock gate-config file whose computed SHA-256 matches
    `config_hash`. Since the preflight check reads the hash from the file
    contents, we inject the hash by writing a content whose SHA matches.
    In practice, tests control the hash via the `current_config_hash` arg."""
    p = tmp_path / "calibration-gate.yaml"
    p.write_text(
        f"pre_registration_tag: {pre_reg_tag}\n"
        f"pre_registration_sha: {pre_reg_sha}\n"
    )
    return p


class TestMissingCalibrationReport:
    def test_no_report_for_bank_raises(self, tmp_path):
        gate_path = _make_gate_config_file(tmp_path)
        with pytest.raises(PreflightError, match="no calibration"):
            preflight_checks(
                bank_id="arithmetic_hard_v1",
                artifacts_dir=tmp_path / "artifacts",  # doesn't exist
                gate_config_path=gate_path,
                current_config_hash="abc123",
                current_pre_reg_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
                current_pre_reg_sha="a" * 40,
                sweep_seeds=range(100, 200),
            )


class TestGateConfigHashMismatch:
    def test_mismatch_raises(self, tmp_path):
        _make_calibration_report(tmp_path, config_hash="abc123")
        gate_path = _make_gate_config_file(tmp_path)
        with pytest.raises(PreflightError, match="gate config"):
            preflight_checks(
                bank_id="arithmetic_hard_v1",
                artifacts_dir=tmp_path / "artifacts",
                gate_config_path=gate_path,
                current_config_hash="DIFFERENT_HASH",  # gate was edited since calibration
                current_pre_reg_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
                current_pre_reg_sha="a" * 40,
                sweep_seeds=range(100, 200),
            )


class TestSeedOverlap:
    def test_overlap_raises(self, tmp_path):
        _make_calibration_report(
            tmp_path,
            config_hash="abc123",
            seed_ranges=[[0, 50]],
        )
        gate_path = _make_gate_config_file(tmp_path)
        with pytest.raises(PreflightError, match="seed"):
            preflight_checks(
                bank_id="arithmetic_hard_v1",
                artifacts_dir=tmp_path / "artifacts",
                gate_config_path=gate_path,
                current_config_hash="abc123",
                current_pre_reg_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
                current_pre_reg_sha="a" * 40,
                sweep_seeds=range(25, 75),  # overlaps [0, 50]
            )

    def test_disjoint_seeds_pass(self, tmp_path):
        _make_calibration_report(
            tmp_path,
            config_hash="abc123",
            seed_ranges=[[0, 50]],
        )
        gate_path = _make_gate_config_file(tmp_path)
        # No exception on disjoint seeds.
        preflight_checks(
            bank_id="arithmetic_hard_v1",
            artifacts_dir=tmp_path / "artifacts",
            gate_config_path=gate_path,
            current_config_hash="abc123",
            current_pre_reg_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
            current_pre_reg_sha="a" * 40,
            sweep_seeds=range(100, 200),
        )


class TestPreRegistrationSentinel:
    def test_placeholder_tag_refused(self, tmp_path):
        _make_calibration_report(tmp_path, config_hash="abc123")
        gate_path = _make_gate_config_file(tmp_path)
        with pytest.raises(PreflightError, match="pre.?registration"):
            preflight_checks(
                bank_id="arithmetic_hard_v1",
                artifacts_dir=tmp_path / "artifacts",
                gate_config_path=gate_path,
                current_config_hash="abc123",
                current_pre_reg_tag="REPLACE_BEFORE_CALIBRATION",
                current_pre_reg_sha="a" * 40,
                sweep_seeds=range(100, 200),
            )

    def test_placeholder_sha_refused(self, tmp_path):
        _make_calibration_report(tmp_path, config_hash="abc123")
        gate_path = _make_gate_config_file(tmp_path)
        with pytest.raises(PreflightError, match="pre.?registration"):
            preflight_checks(
                bank_id="arithmetic_hard_v1",
                artifacts_dir=tmp_path / "artifacts",
                gate_config_path=gate_path,
                current_config_hash="abc123",
                current_pre_reg_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
                current_pre_reg_sha="REPLACE_BEFORE_CALIBRATION",
                sweep_seeds=range(100, 200),
            )


class TestHappyPath:
    def test_all_checks_pass(self, tmp_path):
        _make_calibration_report(
            tmp_path,
            config_hash="abc123",
            seed_ranges=[[0, 50]],
        )
        gate_path = _make_gate_config_file(tmp_path)
        # Should not raise.
        preflight_checks(
            bank_id="arithmetic_hard_v1",
            artifacts_dir=tmp_path / "artifacts",
            gate_config_path=gate_path,
            current_config_hash="abc123",
            current_pre_reg_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
            current_pre_reg_sha="a" * 40,
            sweep_seeds=range(100, 200),
        )
