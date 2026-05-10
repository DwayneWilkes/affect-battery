"""Dashboard source for `scripts/calibration/h3b_calibration.py`-style
calibration runs. Reads the ExperimentTracker layout at
`<output>.tracker/bank_<sha>/` via `src.lib.tracker_io`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lib.tracker_io import (  # noqa: E402
    find_bank_subdir,
    load_cache_items,
    load_run_metadata,
    tracker_root_for,
)

from scripts.dashboards.snapshot import RunSnapshot  # noqa: E402


class CalibrationSource:
    """Constructs a `RunSnapshot` from a calibration tracker root."""

    def load(self, output_path: Path) -> RunSnapshot:
        title = "H3b Calibration"
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
                    cells=[],
                    extras={
                        "target_lo": final.get("target_lo", 0.40),
                        "target_hi": final.get("target_hi", 0.60),
                    },
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
        target_lo = float(params.get("target_lo", 0.40))
        target_hi = float(params.get("target_hi", 0.60))

        return RunSnapshot(
            title=title,
            cells_done=len(cells),
            cells_total=int(params.get("n_candidates", 0)),
            metadata=md,
            cells=cells,
            extras={
                "target_lo": target_lo,
                "target_hi": target_hi,
                "bank_dir_name": bank_dir.name,
            },
        )
