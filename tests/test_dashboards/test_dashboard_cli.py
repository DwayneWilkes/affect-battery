"""Tests for the dashboard CLI: mode auto-detection and one-frame
render of both source types against synthetic input."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from scripts.dashboards.dashboard import (
    detect_mode, render_calibration, render_pilot,
)
from scripts.dashboards.sources.calibration import CalibrationSource
from scripts.dashboards.sources.pilot import PilotSource


def _capture(layout) -> str:
    """Render a Layout to a string so tests can assert on the actual
    visible content rather than just 'didn't raise'. Layouts need
    both width and height to render contents (without height,
    rich draws borders only); 200x60 covers any reasonable panel
    arrangement."""
    import io
    console = Console(width=200, height=60, record=True, file=io.StringIO())
    with console.capture() as cap:
        console.print(layout)
    return cap.get()


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


def test_render_calibration_renders_config_progress_and_cells(tmp_path: Path):
    """Verify the calibration frame surfaces the config params, the
    progress bar, the usage panel, and the cells summary — not just
    'didn't raise'."""
    output_path = tmp_path / "calib.json"
    tracker_root = output_path.with_suffix(output_path.suffix + ".tracker")
    bank_dir = tracker_root / "bank_abc123"
    (bank_dir / "cache").mkdir(parents=True)
    (bank_dir / "run_metadata.json").write_text(json.dumps({
        "params": {"model": "gpt-5.4-nano", "n_candidates": 10,
                   "target_lo": 0.40, "target_hi": 0.60},
        "metrics": {"usage_n_calls": 50, "usage_estimated_usd": 0.123},
        "stages": {"pre_screen": {"duration_seconds": 12.5}},
    }))
    (bank_dir / "cache" / "x1.json").write_text(json.dumps({
        "kind": "scored", "p_hat": 0.50, "item_id": "x1",
    }))
    (bank_dir / "cache" / "x2.json").write_text(json.dumps({
        "kind": "scored", "p_hat": 0.20, "item_id": "x2",
    }))
    snap = CalibrationSource(title="Test Calibration").load(output_path)
    out = _capture(render_calibration(snap))
    assert "Test Calibration" in out, "title not rendered"
    assert "gpt-5.4-nano" in out, "config panel missing model"
    assert "config" in out and "progress" in out and "usage" in out
    assert "cells" in out, "right-column cells panel missing"
    # 2 scored cells, 1 in band ([0.40, 0.60]), 1 out of band.
    assert "scored cells" in out
    assert "in band" in out
    # Progress is 2/10 = 20%.
    assert "20.0%" in out


def test_render_pilot_renders_config_progress_and_passes(tmp_path: Path):
    """Verify the pilot frame surfaces the config params, the
    progress bar, and the per-pass status grid — not just 'didn't
    raise'."""
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
        "model: gpt-5.4-nano\nprovider: openai\nseed: 42\n"
    )
    for level in (1, 2):
        d = pass_dir / "data" / f"level_{level}" / "neutral"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"{i:04d}.json").write_text("{}")
    snap = PilotSource(title="Test Pilot").load(tmp_path)
    out = _capture(render_pilot(snap))
    assert "Test Pilot" in out, "title not rendered"
    assert "gpt-5.4-nano" in out, "model from first pass manifest missing"
    assert "passes" in out and "pass_01" in out and "pass_02" in out
    # pass_01 is 6/6 complete, pass_02 is 0/6 waiting.
    assert "complete" in out, "complete pass status missing"
    assert "waiting" in out, "waiting pass status missing"


def test_render_calibration_completion_shows_done_panel(tmp_path: Path):
    """Final-state JSON makes the source's `final` non-None, and the
    renderer dispatches to the COMPLETE branch with the headline
    counts visible."""
    output_path = tmp_path / "calib.json"
    output_path.write_text(json.dumps({
        "n_calibrated": 35, "n_candidates": 1319, "n_blocked": 2,
        "target_lo": 0.40, "target_hi": 0.60,
    }))
    snap = CalibrationSource(title="Test Calibration").load(output_path)
    assert snap.is_done
    out = _capture(render_calibration(snap))
    assert "COMPLETE" in out
    assert "35" in out  # n_calibrated
    assert "1319" in out  # n_candidates
    assert "0.4" in out and "0.6" in out  # target band
