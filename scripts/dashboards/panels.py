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
    """Human-readable duration string at second resolution. Values
    that round to zero render as '<1 second' rather than '0 seconds'
    so the panel never claims a stage took no time at all."""
    s = int(round(seconds))
    if s <= 0:
        return "<1 second"
    return humanize.precisedelta(timedelta(seconds=s), minimum_unit="seconds")


def header_panel(snap: RunSnapshot) -> Panel:
    return Panel(
        Align.center(Text.from_markup(
            f"[bold cyan]{snap.title} Dashboard[/bold cyan]  "
            f"[dim]· {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
        )),
        border_style="cyan",
    )


_DEFAULT_CONFIG_FIELDS: tuple[tuple[str, str], ...] = (
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


def config_panel(snap: RunSnapshot) -> Panel:
    """Run config table. Renders the `(label, params_key)` pairs in
    `snap.config_fields` when set; otherwise renders the default
    field list applicable to both calibration and pilot."""
    params = snap.metadata.get("params", {})
    fields = snap.config_fields or _DEFAULT_CONFIG_FIELDS
    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", width=12)
    table.add_column()
    rows = 0
    for label, key in fields:
        val = params.get(key)
        if val is None:
            continue
        table.add_row(label, str(val))
        rows += 1
    if rows == 0:
        table.add_row("(no params yet)", "")
    return Panel(table, title="config", border_style="blue")


def progress_panel(snap: RunSnapshot) -> Panel:
    """Cells done / total + percentage bar, plus elapsed wall-clock
    and ETA when `started_utc` is in `metadata.params`. ETA
    extrapolates linearly from the cell-rate over the run so far."""
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
    started = snap.metadata.get("params", {}).get("started_utc")
    timing = _timing_line(started, snap.cells_done, snap.cells_total)
    if timing:
        body.add_row(Text.from_markup(timing))
    return Panel(body, title="progress", border_style="green")


def _timing_line(started_iso, done: int, total: int) -> str | None:
    """Format an `elapsed · ETA` line for the progress panel. Returns
    None when `started_iso` is missing or unparseable."""
    if not started_iso:
        return None
    try:
        started_dt = datetime.fromisoformat(str(started_iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if started_dt.tzinfo is None:
        started_dt = started_dt.replace(tzinfo=timezone.utc)
    elapsed_s = max((datetime.now(timezone.utc) - started_dt).total_seconds(), 0.0)
    line = f"elapsed [bold]{_fmt_dur(elapsed_s)}[/bold]"
    if done > 0 and total > done and elapsed_s > 0:
        rate = done / elapsed_s
        if rate > 0:
            remaining_s = (total - done) / rate
            line += f" · ETA [bold]{_fmt_dur(remaining_s)}[/bold]"
    return line


def usage_panel(snap: RunSnapshot) -> Panel:
    """API usage / cost from the `usage_*` keys in
    `snap.metadata['metrics']`. Sources that cannot surface usage set
    `extras["usage_unavailable"]` so the panel renders an explicit
    'not tracked' message; absent metrics without the flag render
    'no data yet' for early-startup states that will populate."""
    metrics = snap.metadata.get("metrics", {})
    n_calls = metrics.get("usage_n_calls")
    if n_calls is None:
        if snap.extras.get("usage_unavailable"):
            msg = (
                "[dim]usage not tracked for this run mode\n"
                "(see platform.openai.com/usage for live spend)[/dim]"
            )
        else:
            msg = "[dim](no usage data yet)[/dim]"
        return Panel(Text.from_markup(msg),
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
    """Stage timings from `snap.metadata['stages']`. Each row shows a
    stage name and either its duration or 'in flight' for stages
    still running."""
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
