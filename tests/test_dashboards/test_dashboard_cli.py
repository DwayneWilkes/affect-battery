"""Tests for the dashboard CLI: mode auto-detection and one-frame
render of both source types against synthetic input."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dashboards.dashboard import (
    detect_mode, render_calibration, render_pilot,
)
from scripts.dashboards.sources.calibration import CalibrationSource
from scripts.dashboards.sources.pilot import PilotSource


def test_detect_mode_picks_calibration_for_json_path(tmp_path: Path):
    """A `.json` path always means calibration, even before the file
    exists."""
    assert detect_mode(tmp_path / "calib.json") == "calibration"


def test_detect_mode_picks_pilot_for_directory_with_manifest(tmp_path: Path):
    (tmp_path / "run_manifest.txt").write_text("n_passes: 1\n")
    assert detect_mode(tmp_path) == "pilot"


def test_detect_mode_picks_pilot_for_existing_directory(tmp_path: Path):
    """An existing directory without `.json` suffix is treated as
    pilot output base, matching the wrapper's CLI shape."""
    (tmp_path / "subdir").mkdir()
    assert detect_mode(tmp_path / "subdir") == "pilot"


def test_render_calibration_produces_a_layout(tmp_path: Path):
    """One frame of calibration rendering against synthetic tracker
    data must complete without raising."""
    output_path = tmp_path / "calib.json"
    tracker_root = output_path.with_suffix(output_path.suffix + ".tracker")
    bank_dir = tracker_root / "bank_abc123"
    (bank_dir / "cache").mkdir(parents=True)
    (bank_dir / "run_metadata.json").write_text(json.dumps({
        "params": {"model": "gpt-x", "n_candidates": 10,
                   "target_lo": 0.40, "target_hi": 0.60},
        "metrics": {"usage_n_calls": 50, "usage_estimated_usd": 0.123},
        "stages": {"pre_screen": {"duration_seconds": 12.5}},
    }))
    (bank_dir / "cache" / "x1.json").write_text(json.dumps({
        "kind": "scored", "p_hat": 0.5, "item_id": "x1",
    }))
    snap = CalibrationSource().load(output_path)
    layout = render_calibration(snap)
    assert layout is not None  # rendered without raising


def test_render_pilot_produces_a_layout(tmp_path: Path):
    """One frame of pilot rendering against synthetic pass dirs must
    complete without raising."""
    (tmp_path / "run_manifest.txt").write_text(
        "n_passes:              2\n"
        "n_bank_items:          3\n"
        "n_levels:              2\n"
        "expected_cells/pass:   6\n"
        "max_parallel:          2\n"
    )
    pass_dir = tmp_path / "pass_01"
    pass_dir.mkdir()
    (pass_dir / "manifest.yaml").write_text(
        "model: gpt-x\nprovider: openai\nseed: 42\n"
    )
    for level in (1, 2):
        d = pass_dir / "data" / f"level_{level}" / "neutral"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i:04d}.json").write_text("{}")
    snap = PilotSource().load(tmp_path)
    layout = render_pilot(snap)
    assert layout is not None


def test_render_calibration_completion_shows_done_panel(tmp_path: Path):
    """Final-state JSON makes the source's `final` non-None, and the
    renderer dispatches to the COMPLETE branch."""
    output_path = tmp_path / "calib.json"
    output_path.write_text(json.dumps({
        "n_calibrated": 35, "n_candidates": 1319, "n_blocked": 2,
        "target_lo": 0.40, "target_hi": 0.60,
    }))
    snap = CalibrationSource().load(output_path)
    assert snap.is_done
    layout = render_calibration(snap)
    assert layout is not None
