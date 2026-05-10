"""Tests for CalibrationSource — wraps the ExperimentTracker tracker
root in a RunSnapshot. Reuses src.lib.tracker_io for layout reads so
the convention stays in one place."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.dashboards.sources.calibration import CalibrationSource


def _make_calibration_tracker(
    tmp_path: Path, *,
    n_candidates: int = 10,
    target_lo: float = 0.40,
    target_hi: float = 0.60,
    cells: list[dict] | None = None,
) -> Path:
    """Build an ExperimentTracker-style tracker root at
    `<output>.tracker/bank_<sha>/`. Returns the calibration JSON
    output path that callers pass to CalibrationSource.load."""
    output_path = tmp_path / "calib.json"
    tracker_root = output_path.with_suffix(output_path.suffix + ".tracker")
    bank_dir = tracker_root / "bank_abc123def456"
    cache_dir = bank_dir / "cache"
    cache_dir.mkdir(parents=True)
    metadata = {
        "experiment_name": "h3b_calibration_test",
        "params": {
            "model": "gpt-5.4-nano",
            "provider": "openai",
            "n_candidates": n_candidates,
            "n_reps": 100,
            "target_lo": target_lo,
            "target_hi": target_hi,
        },
        "metrics": {},
        "stages": {},
    }
    (bank_dir / "run_metadata.json").write_text(json.dumps(metadata))
    if cells:
        for i, c in enumerate(cells):
            p = cache_dir / f"item_{i:04d}.json"
            p.write_text(json.dumps(c))
    return output_path


def test_calibration_source_reads_tracker_layout(tmp_path: Path):
    cells = [
        {"kind": "scored", "item_id": "x1", "p_hat": 0.42, "n_reps": 100},
        {"kind": "scored", "item_id": "x2", "p_hat": 0.58, "n_reps": 100},
        {"kind": "scored", "item_id": "x3", "p_hat": 0.20, "n_reps": 100},
    ]
    output_path = _make_calibration_tracker(
        tmp_path, n_candidates=10, cells=cells,
    )
    snap = CalibrationSource().load(output_path)
    assert snap.cells_done == 3
    assert snap.cells_total == 10
    assert len(snap.cells) == 3
    assert snap.metadata["params"]["model"] == "gpt-5.4-nano"
    assert snap.is_done is False


def test_calibration_source_extras_carry_target_band(tmp_path: Path):
    output_path = _make_calibration_tracker(
        tmp_path, target_lo=0.42, target_hi=0.58,
    )
    snap = CalibrationSource().load(output_path)
    assert snap.extras["target_lo"] == 0.42
    assert snap.extras["target_hi"] == 0.58


def test_calibration_source_reports_done_when_final_json_exists(tmp_path: Path):
    """When the final calibration JSON has been written, the source
    sets `final` so dashboards render the completion panel."""
    output_path = _make_calibration_tracker(tmp_path)
    output_path.write_text(json.dumps({
        "n_calibrated": 35, "n_candidates": 1319, "n_blocked": 2,
        "target_lo": 0.40, "target_hi": 0.60,
    }))
    snap = CalibrationSource().load(output_path)
    assert snap.is_done is True
    assert snap.final["n_calibrated"] == 35


def test_calibration_source_handles_missing_tracker(tmp_path: Path):
    """During the first seconds of a run, the tracker dir doesn't
    exist yet. Source returns a 'waiting' snapshot, not an error."""
    snap = CalibrationSource().load(tmp_path / "calib.json")
    assert snap.cells_done == 0
    assert snap.cells_total == 0
    assert snap.is_done is False
