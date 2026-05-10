"""Layout-agnostic carrier for one frame of dashboard state.

Sources (calibration tracker, pilot pass dirs, future experiments)
construct a RunSnapshot from their on-disk layout. Shared panels read
universal fields (title, progress, metadata, cells); source-specific
panels pull from `extras`. The snapshot is what decouples dashboard
rendering from the source layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunSnapshot:
    """One frame of dashboard state.

    Attributes:
        title: Header text rendered at the top of the dashboard.
        cells_done: Cells written / scored so far.
        cells_total: Expected total cells when the run completes.
        metadata: Run-level params and metrics (model, seed, usage_*,
            stages). Source-shaped, but shared panels look up
            well-known keys via `metadata.get(...)`.
        cells: Per-cell records when the source can enumerate them
            cheaply. Each entry should have a `_mtime` field for
            recent-window stats. Empty when the source streams or
            samples cells.
        extras: Source-specific fields (e.g., calibration's
            target_lo/target_hi/min_items). Source-contributed panels
            read from here.
        final: Final-state record when the run is complete (e.g., the
            calibration JSON or pilot's summary). None while in flight.
    """

    title: str
    cells_done: int
    cells_total: int
    metadata: dict
    cells: list[dict] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    final: dict | None = None

    @property
    def progress_pct(self) -> float:
        if self.cells_total <= 0:
            return 0.0
        return 100.0 * self.cells_done / self.cells_total

    @property
    def is_done(self) -> bool:
        if self.final is not None:
            return True
        return self.cells_total > 0 and self.cells_done >= self.cells_total
