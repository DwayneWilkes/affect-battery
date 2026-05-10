"""Tests for PilotSource — reads the wrapper's `pass_NN/` layout and
builds a RunSnapshot. The pilot wrapper writes a run-level manifest
text file with item/level counts, plus per-pass `manifest.yaml` files
and per-cell JSONs under `pass_NN/data/level_M/neutral/`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dashboards.sources.pilot import PilotSource


def _write_run_manifest(output_base: Path, *, n_passes: int = 20,
                        n_items: int = 93, n_levels: int = 7,
                        max_parallel: int = 5) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    expected = n_items * n_levels
    (output_base / "run_manifest.txt").write_text(
        "H3b Phase 1A: single-turn calibrated replication "
        f"({n_passes} within-subjects passes)\n"
        "prereg:                docs/preregistrations/h3b_2026-05-07.md\n"
        "prereg_commit:         owner/repo@sha\n"
        "bank:                  configs/banks/h3b_calibrated_v2.yaml\n"
        f"n_bank_items:          {n_items}\n"
        f"n_levels:              {n_levels}\n"
        f"expected_cells/pass:   {expected}\n"
        f"n_passes:              {n_passes}\n"
        f"max_parallel:          {max_parallel}\n"
        "seed:                  42\n"
        "started_utc:           2026-05-09T18:00:00Z\n"
    )


def _write_pass(output_base: Path, pass_num: int, *,
                n_items: int = 93, n_levels: int = 7,
                items_per_level: int | None = None) -> None:
    """Build pass_NN/ with synthetic cells. items_per_level lets a
    test simulate partial passes."""
    pass_dir = output_base / f"pass_{pass_num:02d}"
    pass_dir.mkdir(parents=True, exist_ok=True)
    (pass_dir / "manifest.yaml").write_text(
        "pilot_kind: exp3a_pilot\n"
        "model: gpt-5.4-nano\n"
        "provider: openai\n"
        f"num_runs: {n_items}\n"
        "seed: 42\n"
    )
    n = n_items if items_per_level is None else items_per_level
    for level in range(1, n_levels + 1):
        d = pass_dir / "data" / f"level_{level}" / "neutral"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"{i:04d}.json").write_text(json.dumps({
                "run_number": i,
                "experiment_type": "exp3a",
                "model": "gpt-5.4-nano",
                "condition": "neutral",
            }))


def test_pilot_source_counts_cells_across_passes(tmp_path: Path):
    """Total cells_done equals the sum of cell JSONs across every
    pass_NN/data/level_M/neutral/. Total cells_total = n_passes ×
    n_items × n_levels from the run manifest."""
    _write_run_manifest(tmp_path, n_passes=3, n_items=4, n_levels=2)
    _write_pass(tmp_path, 1, n_items=4, n_levels=2)
    _write_pass(tmp_path, 2, n_items=4, n_levels=2)
    # Pass 3 not yet started.
    snap = PilotSource().load(tmp_path)
    assert snap.cells_total == 24  # 3 passes × 4 items × 2 levels
    assert snap.cells_done == 16  # 2 complete passes × 4 × 2
    assert snap.is_done is False


def test_pilot_source_reports_done_when_all_passes_complete(tmp_path: Path):
    _write_run_manifest(tmp_path, n_passes=2, n_items=3, n_levels=2)
    _write_pass(tmp_path, 1, n_items=3, n_levels=2)
    _write_pass(tmp_path, 2, n_items=3, n_levels=2)
    snap = PilotSource().load(tmp_path)
    assert snap.cells_done == 12
    assert snap.cells_total == 12
    assert snap.is_done is True


def test_pilot_source_handles_missing_run_manifest(tmp_path: Path):
    """During the first seconds of a run, `run_manifest.txt` does
    not yet exist. The source returns a snapshot with
    `cells_total=0` so the dashboard renders a 'waiting' panel."""
    snap = PilotSource().load(tmp_path)
    assert snap.cells_done == 0
    assert snap.cells_total == 0
    assert snap.is_done is False


def test_pilot_source_extras_carry_pass_breakdown(tmp_path: Path):
    """The pilot extras include per-pass cell counts so the
    pilot-specific panel can render a per-pass progress grid without
    re-walking the directory tree."""
    _write_run_manifest(tmp_path, n_passes=3, n_items=4, n_levels=2)
    _write_pass(tmp_path, 1, n_items=4, n_levels=2)  # complete
    _write_pass(tmp_path, 2, n_items=2, n_levels=2)  # partial: 2*2=4 cells
    snap = PilotSource().load(tmp_path)
    breakdown = snap.extras["passes"]
    assert breakdown[0]["pass_num"] == 1
    assert breakdown[0]["cells_done"] == 8
    assert breakdown[0]["expected"] == 8
    assert breakdown[1]["pass_num"] == 2
    assert breakdown[1]["cells_done"] == 4
    assert breakdown[1]["expected"] == 8
    assert breakdown[2]["pass_num"] == 3
    assert breakdown[2]["cells_done"] == 0


def test_pilot_source_skips_title_line_in_manifest(tmp_path: Path):
    """`run_manifest.txt` opens with a free-form title line ('H3b
    Phase 1A: single-turn ...') whose 'key' position contains
    whitespace. Canonical params keys are word-only, so the parser
    accepts only those and the title row contributes no entry."""
    _write_run_manifest(tmp_path, n_passes=1, n_items=1, n_levels=1)
    snap = PilotSource().load(tmp_path)
    for key in snap.metadata.get("params", {}):
        assert "H3b_Phase" not in str(key), (
            f"manifest title line leaked into params as key {key!r}"
        )


def test_pilot_source_carries_run_metadata(tmp_path: Path):
    """The first pass's manifest.yaml carries the run's model,
    provider, and seed; the dashboard config panel renders these."""
    _write_run_manifest(tmp_path, n_passes=2, n_items=3, n_levels=2)
    _write_pass(tmp_path, 1, n_items=3, n_levels=2)
    snap = PilotSource().load(tmp_path)
    md = snap.metadata
    assert md["params"]["model"] == "gpt-5.4-nano"
    assert md["params"]["provider"] == "openai"
    assert md["params"]["seed"] == 42
    assert md["params"]["n_passes"] == 2
    assert md["params"]["n_items"] == 3
    assert md["params"]["n_levels"] == 2
