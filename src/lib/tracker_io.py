"""Shared helpers for reading from `ExperimentTracker` output directories.

Multiple scripts (the live dashboard, status snapshot, cache cleanup,
selection-bias re-screen, bank-overlap inspector, carry-over verifier)
all need to (a) resolve the per-bank subdir under a tracker root and
(b) read the cells in its cache directory. Centralizing the conventions
here means a single place updates if the tracker layout evolves.

Conventions (mirrored from `scripts/calibration/h3b_calibration.py`):
- The tracker root is `<output>.tracker/` where `<output>` is the
  calibration JSON path passed to the script.
- Bank subdirs are named `bank_<sha256[:12]>`. A run uses exactly one;
  multiple subdirs may co-exist if successive runs used different banks.
- The cache subdir is `<bank_subdir>/cache/`; cells are JSON files named
  `<item_id>.json`.
"""
from __future__ import annotations

import json
from pathlib import Path


def tracker_root_for(output_path: Path) -> Path:
    """Tracker root for a calibration output: `<output>.tracker/`."""
    return output_path.with_suffix(output_path.suffix + ".tracker")


def find_bank_subdir(tracker_root: Path) -> Path | None:
    """Most-recently-touched `bank_*` subdir under the tracker root,
    or None if the root doesn't exist or holds no bank dirs."""
    if not tracker_root.is_dir():
        return None
    candidates = [
        p for p in tracker_root.iterdir()
        if p.is_dir() and p.name.startswith("bank_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def cache_dir_for(bank_subdir: Path) -> Path:
    """Per-item cache dir within a bank subdir: `<bank_subdir>/cache/`."""
    return bank_subdir / "cache"


def metadata_path_for(bank_subdir: Path) -> Path:
    """Run-metadata file within a bank subdir."""
    return bank_subdir / "run_metadata.json"


def load_run_metadata(bank_subdir: Path) -> dict:
    """Load `run_metadata.json` from a bank subdir; returns `{}` if the
    file is missing or unparseable so callers can degrade gracefully
    during early-startup polling."""
    path = metadata_path_for(bank_subdir)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_cache_items(bank_subdir: Path) -> list[dict]:
    """Read every JSON cell in the bank's cache dir into a list, sorted
    by mtime ascending. Each cell is augmented with `_mtime` so callers
    can compute time-windowed stats. Returns `[]` if the cache dir
    doesn't exist yet (e.g., the run is still in setup)."""
    cache = cache_dir_for(bank_subdir)
    if not cache.is_dir():
        return []
    items: list[dict] = []
    for p in cache.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        data["_mtime"] = p.stat().st_mtime
        items.append(data)
    items.sort(key=lambda d: d["_mtime"])
    return items
