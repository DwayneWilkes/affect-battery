"""Dashboard source for ExperimentTracker-style calibration runs.
Reads `<output>.tracker/bank_<sha>/` (the layout written by
`scripts/calibration/h3b_calibration.py`)."""
from __future__ import annotations

import json
from pathlib import Path

from src.lib.tracker_io import (
    find_bank_subdir,
    load_cache_items,
    load_run_metadata,
    tracker_root_for,
)

from scripts.dashboards.snapshot import RunSnapshot


class CalibrationSource:
    """Constructs a `RunSnapshot` from a calibration tracker root.

    `title` is the dashboard header text. Callers set it per
    experiment (e.g., 'H3b Calibration', 'H4 Calibration')."""

    def __init__(self, title: str = "Calibration") -> None:
        self._title = title

    def load(self, output_path: Path) -> RunSnapshot:
        title = self._title
        if output_path.is_file():
            try:
                final = json.loads(output_path.read_text())
                return RunSnapshot(
                    title=title,
                    cells_done=int(final.get("n_calibrated", 0)),
                    cells_total=int(final.get("n_candidates", 0)),
                    metadata={"params": {
                        "model": final.get("model"),
                        "provider": final.get("provider"),
                        "target_lo": final.get("target_lo"),
                        "target_hi": final.get("target_hi"),
                    }, "metrics": {}, "stages": {}},
                    final=final,
                )
            except (json.JSONDecodeError, OSError):
                pass

        tracker_root = tracker_root_for(output_path)
        bank_dir = find_bank_subdir(tracker_root)
        if bank_dir is None:
            return RunSnapshot(
                title=title,
                cells_done=0, cells_total=0,
                metadata={"params": {}, "metrics": {}, "stages": {}},
            )

        md = load_run_metadata(bank_dir)
        params = md.get("params", {})
        cells = load_cache_items(bank_dir)

        return RunSnapshot(
            title=title,
            cells_done=len(cells),
            cells_total=int(params.get("n_candidates", 0)),
            metadata=md,
            cells=cells,
            extras={"bank_dir_name": bank_dir.name},
        )
