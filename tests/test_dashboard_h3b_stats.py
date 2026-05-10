"""Tests for the dashboard's band-stats helper.

The dashboard live-renders rich panels that are expensive to test
visually. The `compute_band_stats` helper isolates the pure-data math
(in-band count, yield, projection, blocked count, median p̂, items-
needed-to-floor) so we can assert on it without driving rich.

Reasoning for some of the math:
- `recent_yield_pct` is over the last `recent_window` scored cells,
  ordered by cache mtime. It tells the operator if yield has dried up
  late in the run (e.g., easier items got front-loaded) — overall yield
  alone hides that.
- `candidates_needed_for_min` is how many MORE candidates we'd need to
  reach `min_items` in-band, assuming the overall yield holds. None when
  the floor is already met or when no scored cells exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibration.dashboard_h3b import compute_band_stats


def _scored(item_id: str, p_hat: float, mtime: float = 0.0) -> dict:
    return {
        "kind": "scored",
        "item_id": item_id,
        "p_hat": p_hat,
        "n_reps": 100,
        "n_correct": int(p_hat * 100),
        "_mtime": mtime,
    }


def _blocked(item_id: str, mtime: float = 0.0) -> dict:
    return {
        "kind": "blocked",
        "item_id": item_id,
        "reason": "rate_limit",
        "_mtime": mtime,
    }


def test_band_stats_counts_in_band_and_blocked_separately():
    cells = [
        _scored("a", 0.45),  # in band
        _scored("b", 0.55),  # in band
        _scored("c", 0.30),  # below
        _scored("d", 0.85),  # above
        _blocked("e"),       # blocked
        _blocked("f"),       # blocked
    ]
    s = compute_band_stats(cells, 0.40, 0.60, n_target=100)
    assert s.n_scored == 4
    assert s.n_blocked == 2
    assert s.n_in_band == 2
    assert s.n_above_band == 1
    assert s.n_below_band == 1
    assert s.yield_pct == 50.0


def test_band_stats_projection_extrapolates_at_overall_yield():
    """1 in-band of 4 scored == 25% yield → 25 of 100 candidates."""
    cells = [
        _scored("a", 0.50),
        _scored("b", 0.30),
        _scored("c", 0.85),
        _scored("d", 0.20),
    ]
    s = compute_band_stats(cells, 0.40, 0.60, n_target=100)
    assert s.yield_pct == 25.0
    assert s.projected_at_total == 25


def test_band_stats_median_p_hat_uses_scored_only():
    cells = [
        _scored("a", 0.10),
        _scored("b", 0.50),
        _scored("c", 0.90),
        _blocked("d"),
    ]
    s = compute_band_stats(cells, 0.40, 0.60, n_target=10)
    assert s.median_p_hat == 0.50


def test_band_stats_candidates_needed_for_min_uses_overall_yield():
    """At 10% yield, need 100 in-band, currently have 5 → 950 more cand
    at 10% gives 95 more in-band, total 100."""
    cells = []
    # 5 in-band out of 50 scored = 10% yield
    for i in range(5):
        cells.append(_scored(f"hit_{i}", 0.5))
    for i in range(45):
        cells.append(_scored(f"miss_{i}", 0.20))
    s = compute_band_stats(cells, 0.40, 0.60, n_target=2000, min_items=100)
    assert s.n_in_band == 5
    assert s.yield_pct == 10.0
    # need 95 more in-band → at 10% yield need ~950 more candidates
    assert s.candidates_needed_for_min is not None
    assert 940 <= s.candidates_needed_for_min <= 960


def test_band_stats_candidates_needed_for_min_none_when_floor_met():
    cells = [_scored(f"hit_{i}", 0.5) for i in range(30)]
    s = compute_band_stats(cells, 0.40, 0.60, n_target=200, min_items=25)
    assert s.candidates_needed_for_min is None  # already over the floor


def test_band_stats_recent_yield_reflects_late_window():
    """Front-loaded in-band items shouldn't make recent yield look healthy.
    First 10 scored = all in-band (100% historical), next 10 scored = none
    in-band → recent yield over last 10 should be 0%."""
    cells = []
    for i in range(10):
        cells.append(_scored(f"early_{i}", 0.50, mtime=float(i)))
    for i in range(10):
        cells.append(_scored(f"late_{i}", 0.20, mtime=float(100 + i)))
    s = compute_band_stats(cells, 0.40, 0.60, n_target=50, recent_window=10)
    assert s.yield_pct == 50.0          # 10/20 overall
    assert s.recent_yield_pct == 0.0     # 0/10 in last window


def test_band_stats_handles_empty_cells():
    s = compute_band_stats([], 0.40, 0.60, n_target=100)
    assert s.n_scored == 0
    assert s.n_blocked == 0
    assert s.n_in_band == 0
    assert s.yield_pct == 0.0
    assert s.median_p_hat is None
    assert s.projected_at_total == 0
    assert s.recent_yield_pct is None


def test_band_stats_above_below_use_strict_inequalities():
    """Boundary p̂ values exactly equal to target_lo / target_hi are in-band,
    not above/below. (target band is inclusive on both ends.)"""
    cells = [
        _scored("lo_edge", 0.40),
        _scored("hi_edge", 0.60),
        _scored("just_below", 0.39),
        _scored("just_above", 0.61),
    ]
    s = compute_band_stats(cells, 0.40, 0.60, n_target=10)
    assert s.n_in_band == 2
    assert s.n_below_band == 1
    assert s.n_above_band == 1
