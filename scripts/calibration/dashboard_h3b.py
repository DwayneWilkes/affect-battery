"""Live terminal dashboard for an H3b calibration run.

Reads the ExperimentTracker output dir (per-bank cache + run_metadata.json)
and live-refreshes a snapshot showing progress, ETA, p̂ histogram, in-band
yield with overall + recent-window comparison, blocked count, items-needed
math against a `--min-items` floor, recent in-band hits, and stage timings.
Built on rich for layout / live-refresh / colors and humanize for duration
prose.

Usage:
    python scripts/calibration/dashboard_h3b.py \\
        configs/h3b_calibration_2026-05-08.json \\
        [--refresh 5] [--expected-total 1319] [--min-items 32]

Run from a second terminal while the calibration is in flight. Exits
when the final JSON appears (calibration complete) or on Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import humanize
import plotille
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


@dataclass
class BandStats:
    """Pure-data summary of cache cells for the dashboard panels.

    Separated from rich rendering so it can be unit-tested without
    driving the live layout.
    """
    n_scored: int = 0
    n_blocked: int = 0
    n_in_band: int = 0
    n_above_band: int = 0
    n_below_band: int = 0
    yield_pct: float = 0.0
    median_p_hat: float | None = None
    projected_at_total: int = 0
    recent_yield_pct: float | None = None
    candidates_needed_for_min: int | None = None
    p_hats: list[float] = field(default_factory=list)


def compute_band_stats(
    cells: list[dict],
    target_lo: float,
    target_hi: float,
    n_target: int,
    min_items: int | None = None,
    recent_window: int = 50,
) -> BandStats:
    """Aggregate cache cells into a snapshot of progress + yield.

    `recent_yield_pct` is over the last `recent_window` scored cells by
    cache mtime; useful to spot late-run yield drift hidden by overall
    averages. `candidates_needed_for_min` extrapolates from overall yield
    and returns None when the floor is already met or yield is zero.
    """
    scored = [c for c in cells if c.get("kind") == "scored"]
    blocked = [c for c in cells if c.get("kind") == "blocked"]
    p_hats = [float(c.get("p_hat", 0.0)) for c in scored]

    in_band = [p for p in p_hats if target_lo <= p <= target_hi]
    above = [p for p in p_hats if p > target_hi]
    below = [p for p in p_hats if p < target_lo]

    yield_pct = (100.0 * len(in_band) / len(scored)) if scored else 0.0
    median = statistics.median(p_hats) if p_hats else None
    projected = round(yield_pct * n_target / 100.0) if n_target else 0

    recent_pct: float | None = None
    if scored:
        scored_sorted = sorted(scored, key=lambda c: c.get("_mtime", 0.0))
        tail = scored_sorted[-recent_window:]
        if tail:
            tail_in = sum(
                1 for c in tail
                if target_lo <= float(c.get("p_hat", 0.0)) <= target_hi
            )
            recent_pct = 100.0 * tail_in / len(tail)

    needed: int | None = None
    if min_items is not None and len(in_band) < min_items and yield_pct > 0:
        gap = min_items - len(in_band)
        needed = int(round(gap / (yield_pct / 100.0)))

    return BandStats(
        n_scored=len(scored),
        n_blocked=len(blocked),
        n_in_band=len(in_band),
        n_above_band=len(above),
        n_below_band=len(below),
        yield_pct=yield_pct,
        median_p_hat=median,
        projected_at_total=projected,
        recent_yield_pct=recent_pct,
        candidates_needed_for_min=needed,
        p_hats=p_hats,
    )


def find_bank_subdir(tracker_root: Path) -> Path | None:
    if not tracker_root.is_dir():
        return None
    candidates = [
        p for p in tracker_root.iterdir()
        if p.is_dir() and p.name.startswith("bank_")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_metadata(bank_dir: Path) -> dict:
    path = bank_dir / "run_metadata.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_cache_items(bank_dir: Path) -> list[dict]:
    cache_dir = bank_dir / "cache"
    if not cache_dir.is_dir():
        return []
    items = []
    for p in cache_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            data["_mtime"] = p.stat().st_mtime
            items.append(data)
        except Exception:
            continue
    items.sort(key=lambda d: d["_mtime"])
    return items


def fmt_dur(seconds: float) -> str:
    if seconds is None or seconds <= 0:
        return "—"
    return humanize.precisedelta(timedelta(seconds=seconds), minimum_unit="seconds", format="%0.0f")


def make_config_panel(params: dict, bank_subdir_name: str) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("bank", str(params.get("bank", "?")))
    t.add_row("bank sha[:12]", bank_subdir_name.replace("bank_", "") if bank_subdir_name else "?")
    t.add_row("model", f"{params.get('provider', '?')} / {params.get('model', '?')}")
    t.add_row("target band", f"[{params.get('target_lo', '?')}, {params.get('target_hi', '?')}]")
    t.add_row("n_candidates", str(params.get("n_candidates", "?")))
    t.add_row("n_reps", str(params.get("n_reps", "?")))
    t.add_row("max_concurrent", str(params.get("max_concurrent", "?")))
    t.add_row("candidates_per_batch", str(params.get("candidates_per_batch", "?")))
    return Panel(t, title="[bold]run config[/bold]", border_style="cyan")


def make_progress_panel(n_cached: int, target: int, cells: list[dict],
                        stats: BandStats) -> Panel:
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", justify="right")
    t.add_column()

    bar = ProgressBar(total=target if target else 1, completed=n_cached, width=40)
    pct = (100.0 * n_cached / target) if target else 0.0
    t.add_row("cached", f"[bold]{n_cached}[/bold] / {target}  ({pct:.1f}%)")
    t.add_row("", bar)
    # Surface scored vs blocked split so a creeping rate-limit problem
    # (lots of cells, little progress) is visible without checking logs.
    blocked_color = "red" if stats.n_blocked else "dim"
    t.add_row(
        "split",
        f"[green]{stats.n_scored}[/green] scored · "
        f"[{blocked_color}]{stats.n_blocked}[/{blocked_color}] blocked",
    )

    if n_cached >= 2 and cells:
        first_t = cells[0]["_mtime"]
        elapsed = max(time.time() - first_t, 1e-9)
        rate_per_min = n_cached * 60.0 / elapsed
        remaining = max(0, target - n_cached)
        eta_sec = remaining / max(rate_per_min / 60.0, 1e-9)
        eta_at = datetime.now(timezone.utc) + timedelta(seconds=eta_sec)
        t.add_row("elapsed", fmt_dur(elapsed))
        t.add_row("rate", f"[green]{rate_per_min:.2f}[/green] cand/min")
        t.add_row("ETA", f"[yellow]{fmt_dur(eta_sec)}[/yellow]  ({eta_at.strftime('%H:%M:%S UTC')})")
    else:
        t.add_row("status", "[dim]waiting for items...[/dim]")
    return Panel(t, title="[bold]progress[/bold]", border_style="green")


def make_histogram_panel(cells: list[dict], target_lo: float, target_hi: float,
                         target: int, stats: BandStats,
                         min_items: int | None = None) -> Panel:
    if not stats.p_hats:
        return Panel("[dim]no scored items yet[/dim]",
                     title="[bold]p̂ distribution[/bold]", border_style="magenta")

    p_hats = stats.p_hats

    # Headline: in-band count + projection. The two-line layout keeps
    # the most operationally important numbers (in-band + yield) visible
    # even on narrow terminals.
    line1 = Text.from_markup(
        f"  [bold green]IN BAND: {stats.n_in_band} / {stats.n_scored}[/bold green]  "
        f"([bold]{stats.yield_pct:.1f}%[/bold])  "
        f"→  projected [bold yellow]~{stats.projected_at_total}[/bold yellow] at {target} candidates"
    )

    # Distribution shape: median + the share above / below the band.
    # Tells the operator at a glance whether the model is too easy
    # (mass piles up above) or too hard (mass piles up below) for the band.
    median_str = f"{stats.median_p_hat:.3f}" if stats.median_p_hat is not None else "—"
    pct_above = 100.0 * stats.n_above_band / max(stats.n_scored, 1)
    pct_below = 100.0 * stats.n_below_band / max(stats.n_scored, 1)
    line2 = Text.from_markup(
        f"  [dim]median p̂[/dim] [bold]{median_str}[/bold]  ·  "
        f"[red]{pct_below:.0f}% < band[/red]  ·  "
        f"[blue]{pct_above:.0f}% > band[/blue]"
    )

    # Recent-window yield: surfaces yield drift that overall rate hides.
    if stats.recent_yield_pct is not None:
        delta = stats.recent_yield_pct - stats.yield_pct
        arrow = "↑" if delta > 1.0 else ("↓" if delta < -1.0 else "·")
        line3_color = "green" if delta >= 0 else "yellow"
        line3 = Text.from_markup(
            f"  [dim]recent (last 50)[/dim]  "
            f"[{line3_color}]{stats.recent_yield_pct:.1f}%[/{line3_color}] {arrow} "
            f"[dim]vs overall {stats.yield_pct:.1f}%[/dim]"
        )
    else:
        line3 = Text("")

    # Items-needed-math: only if we have a min_items floor and haven't met it.
    if min_items is not None:
        if stats.n_in_band >= min_items:
            line4 = Text.from_markup(
                f"  [bold green]✓ floor met[/bold green]  "
                f"({stats.n_in_band} ≥ min_items={min_items})"
            )
        elif stats.candidates_needed_for_min is not None:
            line4 = Text.from_markup(
                f"  [bold]need[/bold] [cyan]{min_items - stats.n_in_band}[/cyan] "
                f"more in-band  →  ~[cyan]{stats.candidates_needed_for_min}[/cyan] "
                f"more candidates at current yield  "
                f"[dim](floor={min_items})[/dim]"
            )
        else:
            line4 = Text.from_markup(
                f"  [yellow]no in-band hits yet[/yellow]  [dim](floor={min_items})[/dim]"
            )
    else:
        line4 = Text("")

    # Custom 10-bin histogram with in-band rows highlighted. Each bar is
    # scaled to a fixed width; in-band rows get a magenta tint and a "◀──"
    # marker so the band is unambiguous at a glance.
    bins = [0] * 10
    for p in p_hats:
        idx = min(int(p * 10), 9)
        bins[idx] += 1
    max_count = max(bins) if bins else 1
    bar_width = 28

    rows = Table.grid(padding=(0, 1))
    rows.add_column(no_wrap=True)                     # bracket left
    rows.add_column(no_wrap=True, style="dim")        # bin label
    rows.add_column(no_wrap=True)                     # bar
    rows.add_column(no_wrap=True, justify="right")    # count
    rows.add_column(no_wrap=True)                     # band marker

    # Identify the in-band bin indices so we can frame them with
    # corner / pipe characters on the left edge — visually unambiguous
    # boundary even when the bars are skinny.
    in_band_idx = [i for i in range(10)
                   if target_lo <= (i + 0.5) / 10.0 <= target_hi]
    first_in = min(in_band_idx) if in_band_idx else -1
    last_in = max(in_band_idx) if in_band_idx else -1

    for i, n in enumerate(bins):
        lo, hi = i / 10.0, (i + 1) / 10.0
        in_target = first_in <= i <= last_in if first_in >= 0 else False
        bar_len = round(bar_width * n / max_count) if max_count else 0
        bar = "█" * bar_len + " " * (bar_width - bar_len)
        close = ")" if hi < 1.0 else "]"
        label = f"[{lo:.1f}, {hi:.1f}{close}"
        if in_target:
            # Left bracket: ┏ for first in-band row, ┗ for last,
            # ┃ for middle. Solid pink marker on the right.
            if i == first_in and i == last_in:
                bracket = "┣━"  # both edges in one row
            elif i == first_in:
                bracket = "┏━"
            elif i == last_in:
                bracket = "┗━"
            else:
                bracket = "┃ "
            rows.add_row(
                Text.from_markup(f"[bold magenta]{bracket}[/bold magenta]"),
                Text.from_markup(f"[bold magenta]{label}[/bold magenta]"),
                Text.from_markup(f"[bold magenta]{bar}[/bold magenta]"),
                Text.from_markup(f"[bold magenta]{n}[/bold magenta]"),
                Text.from_markup("[bold magenta]◀ target[/bold magenta]"),
            )
        else:
            rows.add_row(
                Text("  "),
                Text.from_markup(f"[dim]{label}[/dim]"),
                Text.from_markup(f"[cyan]{bar}[/cyan]"),
                Text(str(n)),
                Text(""),
            )

    body = Group(line1, line2, line3, line4, Text(""), rows)
    return Panel(
        body,
        title=f"[bold]p̂ distribution[/bold]  [dim](target [{target_lo:.2f}, {target_hi:.2f}])[/dim]",
        border_style="magenta",
    )


def make_recent_panel(cells: list[dict]) -> Panel:
    if not cells:
        return Panel("[dim]none[/dim]", title="[bold]recent[/bold]", border_style="blue")
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="left")
    t.add_column(justify="right")
    t.add_column(style="dim")
    for c in cells[-8:][::-1]:
        kind = c.get("kind", "?")
        iid = c.get("item_id", "?")
        if kind == "scored":
            p = c.get("p_hat", 0)
            color = "magenta" if 0.40 <= p <= 0.60 else "white"
            t.add_row(
                f"[{color}]{iid}[/{color}]",
                f"[{color}]p̂={p:.2f}[/{color}]",
                f"({c.get('n_correct', 0)}/{c.get('n_reps', 0)})",
            )
        else:
            t.add_row(f"[yellow]{iid}[/yellow]", "[yellow]BLOCKED[/yellow]", "")
    return Panel(t, title="[bold]recent (last 8)[/bold]", border_style="blue")


def make_in_band_panel(cells: list[dict], target_lo: float, target_hi: float) -> Panel:
    """Most-recent in-band hits — what the operator most wants to see.

    Distinct from `make_recent_panel`, which mixes scored + blocked +
    out-of-band. This panel filters to scored cells whose p̂ is in the
    target band, sorted by mtime descending. Empty state explicitly
    invites patience rather than implying a failure.
    """
    in_band = [
        c for c in cells
        if c.get("kind") == "scored"
        and target_lo <= float(c.get("p_hat", 0.0)) <= target_hi
    ]
    in_band.sort(key=lambda c: c.get("_mtime", 0.0), reverse=True)
    if not in_band:
        return Panel(
            "[dim]waiting for the first in-band item...[/dim]",
            title="[bold]recent in-band[/bold]",
            border_style="magenta",
        )
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="left")
    t.add_column(justify="right")
    t.add_column(style="dim")
    for c in in_band[:10]:
        p = float(c.get("p_hat", 0.0))
        t.add_row(
            f"[bold magenta]{c.get('item_id', '?')}[/bold magenta]",
            f"[bold magenta]p̂={p:.2f}[/bold magenta]",
            f"({c.get('n_correct', 0)}/{c.get('n_reps', 0)})",
        )
    return Panel(t, title=f"[bold]recent in-band[/bold] [dim](top 10 by recency)[/dim]",
                 border_style="magenta")


def make_usage_panel(md: dict) -> Panel:
    """Surface running token totals + (when pricing supplied) cost.

    Reads `metrics.usage_*` from run_metadata.json — populated by the
    calibration script after every batch via `tracker.log_metrics`.
    Empty state is informative rather than blank: tells the operator
    how to enable cost estimation when they haven't passed pricing.
    """
    metrics = md.get("metrics", {})
    n_calls = metrics.get("usage_n_calls")
    if n_calls is None:
        return Panel(
            "[dim]usage not yet logged (next batch will populate)[/dim]",
            title="[bold]API usage[/bold]", border_style="cyan",
        )
    prompt = metrics.get("usage_prompt_tokens", 0)
    completion = metrics.get("usage_completion_tokens", 0)
    reasoning = metrics.get("usage_reasoning_tokens", 0)
    cost = metrics.get("usage_estimated_usd")

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", justify="right")
    t.add_column()
    t.add_row("calls", f"[bold]{int(n_calls):,}[/bold]")
    t.add_row("prompt", f"{int(prompt):,} tokens")
    t.add_row("completion", f"{int(completion):,} tokens")
    if reasoning:
        t.add_row("reasoning", f"[yellow]{int(reasoning):,}[/yellow] tokens")
    if cost is not None:
        t.add_row("cost", f"[bold green]${float(cost):.4f}[/bold green]")
        # Per-call average lets the operator extrapolate to remaining work.
        per_call = float(cost) / max(int(n_calls), 1)
        t.add_row("per call", f"[dim]${per_call:.6f}[/dim]")
    else:
        t.add_row("cost", "[dim]pricing not supplied[/dim]")
    return Panel(t, title="[bold]API usage[/bold]", border_style="cyan")


def make_stages_panel(md: dict) -> Panel:
    stages = md.get("stages", {})
    if not stages:
        return Panel("[dim]no completed stages yet[/dim]", title="[bold]stages[/bold]", border_style="yellow")
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", justify="right")
    t.add_column()
    for name, info in stages.items():
        dur = info.get("duration_seconds")
        t.add_row(name, fmt_dur(dur) if dur else "[dim]in progress[/dim]")
    return Panel(t, title="[bold]stages[/bold]", border_style="yellow")


def render(args, console: Console) -> tuple[bool, Layout]:
    """Build a Layout for one frame. Returns (done, layout)."""
    output_path = args.output_path
    final_json = output_path.is_file()
    tracker_root = output_path.with_suffix(output_path.suffix + ".tracker")
    bank_dir = find_bank_subdir(tracker_root)

    layout = Layout()

    if final_json:
        try:
            final = json.loads(output_path.read_text())
            done_panel = Panel(
                Text.from_markup(
                    f"\n[bold green]✓ COMPLETE[/bold green]\n\n"
                    f"  final JSON  : {output_path}\n"
                    f"  n_calibrated: [bold]{final.get('n_calibrated', '?')}[/bold] of {final.get('n_candidates', '?')}\n"
                    f"  n_blocked   : {final.get('n_blocked', '?')}\n"
                    f"  band        : [{final.get('target_lo', '?')}, {final.get('target_hi', '?')}]\n"
                    f"  generated   : {final.get('generated_utc', '?')}\n"
                ),
                title=f"[bold]H3b Calibration[/bold] — {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}",
                border_style="green",
            )
            layout.update(done_panel)
            return True, layout
        except Exception as e:
            layout.update(Panel(f"[red]final JSON exists but failed to parse: {e}[/red]"))
            return True, layout

    if bank_dir is None:
        layout.update(Panel(
            f"[dim]waiting for tracker dir at {tracker_root}...[/dim]",
            title="[bold]H3b Calibration Dashboard[/bold]",
            border_style="cyan",
        ))
        return False, layout

    md = load_metadata(bank_dir)
    params = md.get("params", {})
    cells = load_cache_items(bank_dir)

    target = args.expected_total or params.get("n_candidates", 0)

    target_lo = params.get("target_lo", 0.40)
    target_hi = params.get("target_hi", 0.60)

    stats = compute_band_stats(
        cells, target_lo, target_hi, n_target=target,
        min_items=args.min_items,
    )

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(make_config_panel(params, bank_dir.name), name="config"),
        Layout(make_progress_panel(len(cells), target, cells, stats), name="progress"),
        Layout(make_usage_panel(md), name="usage", size=10),
        Layout(make_stages_panel(md), name="stages", size=6),
    )
    layout["right"].split_column(
        Layout(make_histogram_panel(cells, target_lo, target_hi, target,
                                    stats, args.min_items),
               name="hist", size=20),
        Layout(make_in_band_panel(cells, target_lo, target_hi),
               name="in_band", size=14),
        Layout(make_recent_panel(cells), name="recent"),
    )
    layout["header"].update(
        Panel(
            Align.center(Text.from_markup(
                f"[bold cyan]H3b Calibration Dashboard[/bold cyan]  "
                f"[dim]· {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}[/dim]"
            )),
            border_style="cyan",
        )
    )
    return False, layout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("output_path", type=Path,
                    help="Calibration JSON output path (final + tracker root)")
    ap.add_argument("--refresh", type=float, default=5.0,
                    help="Refresh interval in seconds (default: 5)")
    ap.add_argument("--expected-total", type=int, default=None,
                    help="Override total candidate count for progress %% "
                         "(default: read from run_metadata.json's n_candidates)")
    ap.add_argument("--min-items", type=int, default=32,
                    help="Minimum in-band items needed to satisfy the "
                         "follow-on bank's --min-items floor. Default 32 "
                         "matches the simulation-derived strong-claim n at "
                         "100%% reliability (CI half-width < 0.05 in every "
                         "trial, per "
                         "results/probes/h3b_precision_report_2026-05-08.json). "
                         "The histogram headline shows progress toward this "
                         "and projects how many more candidates are "
                         "needed at current yield. Set to 0 to disable.")
    args = ap.parse_args()
    if args.min_items <= 0:
        args.min_items = None

    console = Console()
    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                done, layout = render(args, console)
                live.update(layout)
                if done:
                    time.sleep(2)
                    break
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        console.print("[dim]dashboard exited[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
