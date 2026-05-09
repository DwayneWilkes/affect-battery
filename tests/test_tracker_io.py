"""Tests for src.lib.tracker_io helpers used by the dashboard, the
status snapshot, and the cache-cleanup utilities.

The interesting contract: `find_bank_subdir` must pick the bank that's
actively being written to, even if the directory's own mtime hasn't
updated. ExperimentTracker's atomic-write pattern (tmp+rename) and
in-place log_metric/log_params updates do not always touch the parent
dir mtime — the dashboard would otherwise read a stale bank.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.lib.tracker_io import (
    cache_dir_for,
    find_bank_subdir,
    load_cache_items,
    load_run_metadata,
    metadata_path_for,
    tracker_root_for,
)


def _make_bank_dir(root: Path, sha_prefix: str, *, dir_mtime: float,
                   metadata_mtime: float | None = None,
                   cache_dir_mtime: float | None = None) -> Path:
    """Create a bank subdir with controlled mtimes on the dir itself,
    its run_metadata.json, and its cache/ subdir."""
    p = root / f"bank_{sha_prefix}"
    cache = p / "cache"
    cache.mkdir(parents=True)
    if metadata_mtime is not None:
        md = p / "run_metadata.json"
        md.write_text("{}")
        os.utime(md, (metadata_mtime, metadata_mtime))
    if cache_dir_mtime is not None:
        os.utime(cache, (cache_dir_mtime, cache_dir_mtime))
    os.utime(p, (dir_mtime, dir_mtime))
    return p


def test_find_bank_subdir_prefers_recent_metadata_when_dir_mtimes_tied(
    tmp_path: Path,
):
    """When two banks share the same dir mtime (created together),
    the one with the more-recent run_metadata.json must win.
    Directory mtime alone is insufficient because rewriting
    run_metadata.json in place doesn't always bump the parent dir."""
    now = time.time()
    stale = _make_bank_dir(
        tmp_path, "aaaaaaaaaaaa",
        dir_mtime=now - 3600,
        metadata_mtime=now - 3600,
        cache_dir_mtime=now - 3600,
    )
    active = _make_bank_dir(
        tmp_path, "bbbbbbbbbbbb",
        dir_mtime=now - 3600,
        metadata_mtime=now,
        cache_dir_mtime=now - 3600,
    )
    assert find_bank_subdir(tmp_path) == active, (
        "find_bank_subdir didn't honor recent run_metadata.json: "
        f"got {find_bank_subdir(tmp_path)}, expected {active}"
    )
    assert stale.exists()  # silence unused-name


def test_find_bank_subdir_prefers_recent_cache_when_dir_mtimes_tied(
    tmp_path: Path,
):
    """log_item writes go through the cache/ subdir; adding a new
    cell bumps cache/'s mtime. Active bank wins on cache mtime."""
    now = time.time()
    _make_bank_dir(
        tmp_path, "cccccccccccc",
        dir_mtime=now - 3600,
        metadata_mtime=now - 3600,
        cache_dir_mtime=now - 3600,
    )
    active = _make_bank_dir(
        tmp_path, "dddddddddddd",
        dir_mtime=now - 3600,
        metadata_mtime=now - 3600,
        cache_dir_mtime=now,
    )
    assert find_bank_subdir(tmp_path) == active


def test_find_bank_subdir_returns_none_for_missing_or_empty(tmp_path: Path):
    assert find_bank_subdir(tmp_path / "nonexistent") is None
    (tmp_path / "empty_root").mkdir()
    assert find_bank_subdir(tmp_path / "empty_root") is None


def test_find_bank_subdir_ignores_non_bank_entries(tmp_path: Path):
    """Anything that isn't a `bank_*` directory is invisible to the
    selector. Otherwise stray files (e.g., logs) would compete."""
    (tmp_path / "log.txt").write_text("noise")
    (tmp_path / "bank_real").mkdir()
    (tmp_path / "bank_real" / "cache").mkdir()
    assert find_bank_subdir(tmp_path) == tmp_path / "bank_real"
