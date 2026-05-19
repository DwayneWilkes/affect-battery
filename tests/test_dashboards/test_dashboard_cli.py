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
    """Render a Layout to a string so tests can assert on its visible
    content. Rich's Layout renders both borders and bodies only when
    given an explicit width and height; 200x60 covers the panel
    arrangement used by both renderers with room to spare."""
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


def test_progress_panel_renders_elapsed_and_eta():
    """When `metadata.params.started_utc` is set and the run has
    written cells, the progress panel surfaces both elapsed
    wall-clock and a projected ETA."""
    from datetime import datetime, timedelta, timezone
    from scripts.dashboards.panels import progress_panel
    from scripts.dashboards.snapshot import RunSnapshot
    started = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    snap = RunSnapshot(
        title="test", cells_done=100, cells_total=400,
        metadata={"params": {"started_utc": started}, "metrics": {},
                  "stages": {}},
    )
    out = _capture(progress_panel(snap))
    assert "elapsed" in out
    assert "ETA" in out


def test_progress_panel_omits_eta_when_no_cells_yet():
    """Before the first cell is written, ETA cannot be projected.
    Elapsed alone renders so the operator can see how long they've
    been waiting."""
    from datetime import datetime, timedelta, timezone
    from scripts.dashboards.panels import progress_panel
    from scripts.dashboards.snapshot import RunSnapshot
    started = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    snap = RunSnapshot(
        title="test", cells_done=0, cells_total=400,
        metadata={"params": {"started_utc": started}, "metrics": {},
                  "stages": {}},
    )
    out = _capture(progress_panel(snap))
    assert "elapsed" in out
    assert "ETA" not in out


def test_usage_panel_renders_unavailable_when_flagged():
    """Pilot mode signals via `extras["usage_unavailable"]` that
    token tracking isn't possible from the cell schema. The panel
    points operators to the OpenAI dashboard instead of implying
    usage data will arrive."""
    from scripts.dashboards.panels import usage_panel
    from scripts.dashboards.snapshot import RunSnapshot
    snap = RunSnapshot(
        title="test", cells_done=0, cells_total=10,
        metadata={"params": {}, "metrics": {}, "stages": {}},
        extras={"usage_unavailable": True},
    )
    out = _capture(usage_panel(snap))
    assert "not tracked" in out
    assert "platform.openai.com" in out


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
