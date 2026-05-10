"""Tests for scripts/calibration/build_h3b_bank.py.

Subprocess-based tests against a synthetic calibration JSON. Verifies
the bank YAML produced has the correct schema (matches what the
affect-battery loader expects), preserves item ordering by closeness to
p̂=0.5, includes all qualifiers (no truncation), and refuses runs where
the calibration's yield falls below --min-items.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "calibration" / "build_h3b_bank.py"


def _make_calibration(tmp_path: Path, calibrated_subset: list[dict],
                     **extra) -> Path:
    """Write a synthetic calibration JSON; return its path."""
    calib = {
        "model": "gpt-5.4-nano",
        "provider": "openai",
        "temperature": 0.7,
        "n_candidates": 500,
        "n_reps": 100,
        "target_lo": 0.40,
        "target_hi": 0.60,
        "calibrated_subset": calibrated_subset,
        "n_calibrated": len(calibrated_subset),
        **extra,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calib))
    return path


def _build_item(item_id: str, p_hat: float) -> dict:
    return {
        "item_id": item_id,
        "question": f"Question for {item_id}?",
        "expected": "42.0",
        "n_reps": 100,
        "n_correct": int(p_hat * 100),
        "p_hat": p_hat,
    }


def _run_builder(calib_path: Path, output_path: Path,
                 *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT),
         "--calibration", str(calib_path),
         "--output", str(output_path),
         *extra_args],
        capture_output=True, text=True, timeout=30,
    )


def test_builds_bank_with_all_qualifiers_no_truncation(tmp_path: Path):
    """The bank must include every item from calibrated_subset; no
    truncation, no top-N limit. All-qualifiers selection minimizes the
    bootstrap CI half-width on the contrast at fixed n_calibrated."""
    items = [_build_item(f"gsm8k_{i:04d}", 0.40 + 0.01 * i) for i in range(35)]
    calib = _make_calibration(tmp_path, items)
    out = tmp_path / "bank.yaml"
    r = _run_builder(calib, out)
    assert r.returncode == 0, f"stderr: {r.stderr}"
    bank = yaml.safe_load(out.read_text())
    assert len(bank["items"]) == 35


def test_builder_items_sorted_by_closeness_to_0_5(tmp_path: Path):
    """Items in the bank are sorted by |p_hat - 0.5| ascending so the
    bank reads with the most-central items first (operationally
    informative for review)."""
    items = [
        _build_item("gsm8k_far_lo", 0.40),
        _build_item("gsm8k_central", 0.50),
        _build_item("gsm8k_far_hi", 0.60),
        _build_item("gsm8k_near_lo", 0.48),
    ]
    calib = _make_calibration(tmp_path, items)
    out = tmp_path / "bank.yaml"
    r = _run_builder(calib, out, "--min-items", "4")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    bank = yaml.safe_load(out.read_text())
    ids = [it["id"] for it in bank["items"]]
    assert ids == ["gsm8k_central", "gsm8k_near_lo",
                   "gsm8k_far_lo", "gsm8k_far_hi"], (
        f"items not sorted by |p_hat-0.5|: {ids}"
    )


def test_builder_preserves_required_per_item_fields(tmp_path: Path):
    """The affect-battery loader requires id/question/expected per item.
    The bank generator must emit those plus difficulty and
    answer_aliases (read by the runtime task loader) and p_hat_calib
    (used by selection-bias audit tooling)."""
    items = [_build_item("gsm8k_0001", 0.50)]
    calib = _make_calibration(tmp_path, items)
    out = tmp_path / "bank.yaml"
    assert _run_builder(calib, out, "--min-items", "1").returncode == 0
    bank = yaml.safe_load(out.read_text())
    item = bank["items"][0]
    assert item["id"] == "gsm8k_0001"
    assert item["question"] == "Question for gsm8k_0001?"
    assert item["expected"] == "42.0"
    assert item["difficulty"] == "hard"
    assert item["answer_aliases"] == []
    assert item["p_hat_calib"] == 0.5


def test_builder_emits_bank_level_metadata(tmp_path: Path):
    """Bank-level fields: bank_id, bank_version, bank_type, status,
    parent_bank, alignment_review with a SHA-pinned reference to the
    calibration JSON. The loader uses bank_type to dispatch and
    parent_bank for provenance."""
    items = [_build_item(f"gsm8k_{i:04d}", 0.5) for i in range(2)]
    calib = _make_calibration(tmp_path, items)
    out = tmp_path / "bank.yaml"
    assert _run_builder(calib, out, "--min-items", "2").returncode == 0
    bank = yaml.safe_load(out.read_text())
    assert bank["bank_id"] == "h3b_calibrated_v2"
    assert bank["bank_version"] == 2
    assert bank["bank_type"] == "task"
    assert bank["status"] == "active"
    assert bank["parent_bank"] == "gsm_hard_full_v1"
    review = bank["alignment_review"]
    assert review["verdict"] == "pass"
    assert "sha256" in review["rationale"]
    # The rationale must reference the actual calibration file's SHA;
    # a hardcoded or stale SHA would mean any audit comparing the
    # rationale to the on-disk file fails.
    import hashlib
    expected_sha = hashlib.sha256(calib.read_bytes()).hexdigest()
    assert expected_sha in review["rationale"]


def test_builder_rejects_below_min_items_floor(tmp_path: Path):
    """Sanity check: if calibration yielded fewer items than
    --min-items, the builder errors rather than silently producing an
    underpowered bank."""
    items = [_build_item(f"gsm8k_{i:04d}", 0.5) for i in range(10)]
    calib = _make_calibration(tmp_path, items)
    out = tmp_path / "bank.yaml"
    r = _run_builder(calib, out, "--min-items", "25")
    assert r.returncode == 1
    assert "below --min-items" in r.stderr or "below" in r.stderr
    assert not out.exists(), "bank file should not be created when run errors"


def test_builder_reads_target_band_from_json_not_cli(tmp_path: Path):
    """The target band is recorded in the calibration JSON. The
    builder's rationale string must use those values, not hardcoded CLI
    defaults — otherwise an audit comparing the rationale's claim to
    the actual run config would silently disagree."""
    items = [_build_item(f"gsm8k_{i:04d}", 0.5) for i in range(2)]
    calib = _make_calibration(tmp_path, items, target_lo=0.42, target_hi=0.58)
    out = tmp_path / "bank.yaml"
    assert _run_builder(calib, out, "--min-items", "2").returncode == 0
    bank = yaml.safe_load(out.read_text())
    rationale = bank["alignment_review"]["rationale"]
    assert "0.42" in rationale and "0.58" in rationale, (
        f"target band from JSON not honored in rationale: {rationale}"
    )
