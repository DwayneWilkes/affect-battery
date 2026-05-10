"""Shared rich panels rendered from a `RunSnapshot` regardless of
source. Each function returns a `rich.panel.Panel`. Source-specific
panels live alongside their source module and are passed in at render
time."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import humanize
from rich.align import Align
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from scripts.dashboards.snapshot import RunSnapshot


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration. Sub-second falls back to numeric since
    `humanize.precisedelta` rounds anything < 1s to '0 seconds'."""
    if seconds < 1.0:
        return f"{seconds:.2f}s"
    return humanize.precisedelta(timedelta(seconds=seconds), minimum_unit="seconds")


def header_panel(snap: RunSnapshot) -> Panel:
    return Panel(
        Align.center(Text.from_markup(
            f"[bold cyan]{snap.title} Dashboard[/bold cyan]  "
            f"[dim]· {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
        )),
        border_style="cyan",
    )


def config_panel(snap: RunSnapshot) -> Panel:
    """Run config: model / provider / seed / temperature / etc.
    Reads well-known keys from `snap.metadata['params']`."""
    params = snap.metadata.get("params", {})
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=12)
    table.add_column()
    interesting = (
        ("model", "model"),
        ("provider", "provider"),
        ("seed", "seed"),
        ("temperature", "temperature"),
        ("n_candidates", "n_candidates"),
        ("n_passes", "n_passes"),
        ("n_items", "n_items"),
        ("n_levels", "n_levels"),
        ("n_reps", "n_reps"),
    )
    rows = 0
    for label, key in interesting:
        val = params.get(key)
        if val is None:
            continue
        table.add_row(label, str(val))
        rows += 1
    if rows == 0:
        table.add_row("(no params yet)", "")
    return Panel(table, title="config", border_style="blue")


def progress_panel(snap: RunSnapshot) -> Panel:
    """Cells done / cells total + percentage bar."""
    pct = snap.progress_pct
    bar = ProgressBar(total=max(snap.cells_total, 1), completed=snap.cells_done,
                      width=40)
    body = Table.grid(padding=(0, 1))
    body.add_column()
    body.add_row(bar)
    body.add_row(Text.from_markup(
        f"[bold]{snap.cells_done:,}[/bold] / {snap.cells_total:,} cells "
        f"([cyan]{pct:.1f}%[/cyan])"
    ))
    return Panel(body, title="progress", border_style="green")


def usage_panel(snap: RunSnapshot) -> Panel:
    """API usage / cost from `snap.metadata['metrics']` `usage_*`
    fields. Renders 'no data yet' when the source doesn't carry usage
    metrics (e.g., pilot's pre-completion state)."""
    metrics = snap.metadata.get("metrics", {})
    n_calls = metrics.get("usage_n_calls")
    if n_calls is None:
        return Panel(Text.from_markup("[dim](no usage data yet)[/dim]"),
                     title="usage", border_style="magenta")
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=18)
    table.add_column()
    table.add_row("api calls", f"{n_calls:,}")
    for k, label in (
        ("usage_prompt_tokens", "prompt tokens"),
        ("usage_completion_tokens", "completion tokens"),
        ("usage_reasoning_tokens", "reasoning tokens"),
    ):
        v = metrics.get(k)
        if v is not None:
            table.add_row(label, f"{int(v):,}")
    cost = metrics.get("usage_estimated_usd")
    if cost is not None:
        table.add_row("estimated cost", f"[bold]${float(cost):.4f}[/bold]")
    return Panel(table, title="usage", border_style="magenta")


def stages_panel(snap: RunSnapshot) -> Panel:
    """Stage timings from `snap.metadata['stages']`. Each stage gets a
    row showing duration in seconds (or 'in flight' if not yet done)."""
    stages = snap.metadata.get("stages", {})
    if not stages:
        return Panel(Text.from_markup("[dim](no stages yet)[/dim]"),
                     title="stages", border_style="yellow")
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim")
    table.add_column(justify="right")
    for name, info in stages.items():
        dur = info.get("duration_seconds") if isinstance(info, dict) else None
        if dur is None:
            table.add_row(name, "[dim]in flight[/dim]")
        else:
            table.add_row(name, _fmt_dur(float(dur)))
    return Panel(table, title="stages", border_style="yellow")
