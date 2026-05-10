"""Contract tests for RunSnapshot — the layout-agnostic data carrier
that feeds shared dashboard panels. Sources (calibration tracker,
pilot pass dirs, future experiments) construct a RunSnapshot from
their on-disk layout; panels read it without knowing which source
produced it."""
from __future__ import annotations

from scripts.dashboards.snapshot import RunSnapshot


def test_snapshot_minimal_construction():
    snap = RunSnapshot(
        title="Test Run",
        cells_done=10,
        cells_total=100,
        metadata={},
    )
    assert snap.title == "Test Run"
    assert snap.cells_done == 10
    assert snap.cells_total == 100
    assert snap.progress_pct == 10.0
    assert snap.metadata == {}
    assert snap.cells == []
    assert snap.extras == {}
    assert snap.is_done is False


def test_snapshot_progress_pct_zero_total_returns_zero():
    snap = RunSnapshot(
        title="Pre-run", cells_done=0, cells_total=0, metadata={},
    )
    assert snap.progress_pct == 0.0


def test_snapshot_is_done_when_cells_meet_target():
    snap = RunSnapshot(
        title="Complete",
        cells_done=100, cells_total=100,
        metadata={},
    )
    assert snap.is_done is True


def test_snapshot_extras_carry_source_specific_data():
    """Sources put their experiment-specific data in `extras` so shared
    panels stay layout-agnostic and source-contributed extras can pull
    from a known location."""
    snap = RunSnapshot(
        title="H3b Calibration",
        cells_done=50, cells_total=1319,
        metadata={"params": {"target_lo": 0.40, "target_hi": 0.60}},
        extras={"target_lo": 0.40, "target_hi": 0.60, "min_items": 32},
    )
    assert snap.extras["target_lo"] == 0.40
    assert snap.extras["min_items"] == 32
