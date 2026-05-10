"""Pilot-specific dashboard panel: per-pass progress grid showing
which passes are complete, in-flight, or not yet dispatched."""
from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.dashboards.snapshot import RunSnapshot


def passes_panel(snap: RunSnapshot) -> Panel:
    """Per-pass cell-count grid. Rows: pass_NN. Columns: cells_done /
    expected, status (✓ complete / … running / waiting)."""
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
