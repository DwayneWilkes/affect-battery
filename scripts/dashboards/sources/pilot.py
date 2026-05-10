"""Dashboard source for N-pass pilot runs. Reads three artifact
classes under the wrapper's `--output-base`:

- `run_manifest.txt` — run-level config (n_passes, n_items, n_levels)
- `pass_NN/manifest.yaml` — per-pass model / provider / seed
- `pass_NN/data/level_M/neutral/<NNNN>.json` — per-cell results

Also exports `passes_panel`, the per-pass progress grid rendered in
the pilot dashboard's right column."""
from __future__ import annotations

import re
from pathlib import Path

import yaml
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.dashboards.snapshot import RunSnapshot


_MANIFEST_LINE = re.compile(r"^(\w[\w/-]*?):\s+(.+?)\s*$")
# Numeric-basename cell files written by `affect-battery run`
# (e.g., `0042.json`). Strays like `manifest.yaml` or scratch files
# fail this check and are skipped from cell counts.
_CELL_BASENAME = re.compile(r"^\d+\.json$")


def _parse_run_manifest(path: Path) -> dict:
    """Parse the wrapper's plain-text run manifest into a dict of
    `{key: value}`. Whitespace padding is stripped; digit-only
    values are coerced to `int`. Returns `{}` when the file is
    absent, which lets the dashboard render a 'waiting' panel
    during early startup."""
    if not path.is_file():
        return {}
    out: dict = {}
    for line in path.read_text().splitlines():
        m = _MANIFEST_LINE.match(line)
        if not m:
            continue
        key = m.group(1).strip().replace(" ", "_").replace("/", "_per_")
        val = m.group(2).strip()
        if val.isdigit():
            out[key] = int(val)
        else:
            out[key] = val
    return out


def _count_cells(pass_dir: Path) -> int:
    """Count numeric-basename `*.json` cells under
    `pass_NN/data/level_*/neutral/`."""
    data = pass_dir / "data"
    if not data.is_dir():
        return 0
    return sum(
        1 for p in data.glob("level_*/neutral/*.json")
        if _CELL_BASENAME.match(p.name)
    )


def _load_pass_manifest(pass_dir: Path) -> dict:
    md = pass_dir / "manifest.yaml"
    if not md.is_file():
        return {}
    try:
        return yaml.safe_load(md.read_text()) or {}
    except yaml.YAMLError:
        return {}


def _build_pass_breakdown(
    output_base: Path, n_passes: int, expected_per_pass: int,
) -> tuple[list[dict], int, dict]:
    """Walk `pass_*` subdirs and produce a per-pass status list. The
    output is padded with empty entries for the full `n_passes` range
    so the grid shows every requested slot.

    Returns `(passes, total_cells_done, first_pass_manifest)`."""
    pass_dirs = sorted(
        p for p in output_base.glob("pass_*")
        if p.is_dir() and p.name.startswith("pass_")
    )
    passes: list[dict] = []
    cells_done = 0
    first_pass_md: dict = {}
    for pdir in pass_dirs:
        try:
            pass_num = int(pdir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        n = _count_cells(pdir)
        cells_done += n
        passes.append({
            "pass_num": pass_num,
            "cells_done": n,
            "expected": expected_per_pass,
            "manifest": _load_pass_manifest(pdir),
        })
        if not first_pass_md:
            first_pass_md = passes[-1]["manifest"]
    seen = {p["pass_num"] for p in passes}
    for n in range(1, n_passes + 1):
        if n not in seen:
            passes.append({
                "pass_num": n,
                "cells_done": 0,
                "expected": expected_per_pass,
                "manifest": {},
            })
    passes.sort(key=lambda p: p["pass_num"])
    return passes, cells_done, first_pass_md


class PilotSource:
    """Constructs a `RunSnapshot` from the pilot wrapper's output dir.

    `title` is the dashboard header text. Callers set it per
    experiment (e.g., 'H3b Phase 1A Pilot', 'H4 Pilot')."""

    def __init__(self, title: str = "Pilot") -> None:
        self._title = title

    def load(self, output_base: Path) -> RunSnapshot:
        run_md = _parse_run_manifest(output_base / "run_manifest.txt")
        n_passes = int(run_md.get("n_passes", 0))
        n_items = int(run_md.get("n_bank_items", 0))
        n_levels = int(run_md.get("n_levels", 0))
        expected_per_pass = n_items * n_levels
        cells_total = n_passes * expected_per_pass

        passes, cells_done, first_pass_md = _build_pass_breakdown(
            output_base, n_passes, expected_per_pass,
        )

        params = {
            "model": first_pass_md.get("model"),
            "provider": first_pass_md.get("provider"),
            "seed": first_pass_md.get("seed"),
            "temperature": first_pass_md.get("temperature"),
            "transfer_bank": first_pass_md.get("transfer_bank"),
            "n_passes": n_passes,
            "n_items": n_items,
            "n_levels": n_levels,
            "started_utc": run_md.get("started_utc"),
        }

        return RunSnapshot(
            title=self._title,
            cells_done=cells_done,
            cells_total=cells_total,
            metadata={"params": params, "metrics": {}, "stages": {}},
            cells=[],
            extras={"passes": passes},
        )


def passes_panel(snap: RunSnapshot) -> Panel:
    """Per-pass cell-count grid. One row per pass with columns for
    cell count (`cells_done / expected`) and status (`✓ complete`,
    `running N%`, or `waiting`)."""
    passes = snap.extras.get("passes", [])
    if not passes:
        return Panel(Text.from_markup("[dim](no passes dispatched yet)[/dim]"),
                     title="passes", border_style="cyan")
    table = Table(show_header=True, header_style="bold dim",
                  pad_edge=False, expand=True)
    table.add_column("pass", width=6)
    table.add_column("cells", justify="right")
    table.add_column("status", width=14)
    for p in passes:
        n = p["cells_done"]
        expected = p["expected"]
        if n == 0:
            status = Text.from_markup("[dim]waiting[/dim]")
        elif n < expected:
            pct = 100.0 * n / max(expected, 1)
            status = Text.from_markup(f"[yellow]running {pct:.0f}%[/yellow]")
        else:
            status = Text.from_markup("[green]✓ complete[/green]")
        table.add_row(
            f"pass_{p['pass_num']:02d}",
            f"{n:,} / {expected:,}",
            status,
        )
    return Panel(table, title="passes", border_style="cyan")
