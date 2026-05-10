"""Live terminal dashboard for an in-flight experiment run.

Works for any experiment whose source produces a `RunSnapshot`. Two
built-in sources:

- `calibration` — ExperimentTracker layout
  (`<output>.tracker/bank_<sha>/cache/`). Path argument is the
  calibration JSON output path.
- `pilot` — N-pass layout
  (`<output-base>/pass_NN/data/level_M/neutral/`). Path argument is
  the `--output-base` directory.

Mode is detected from the path's suffix: `.json` is `calibration`,
anything else is `pilot`. `--mode` overrides.

Usage:
    python -m scripts.dashboards.dashboard <path> \\
        [--mode {calibration,pilot,auto}] [--refresh 5]

Run from a second terminal while the run is in flight. Exits when the
source reports `is_done` or on Ctrl-C.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from . import panels
from .snapshot import RunSnapshot
from .sources.calibration import CalibrationSource
from .sources.pilot import PilotSource, passes_panel

MODE_CALIBRATION = "calibration"
MODE_PILOT = "pilot"
MODE_AUTO = "auto"
_MODES = (MODE_CALIBRATION, MODE_PILOT)


def detect_mode(path: Path) -> str:
    """`.json` suffix → calibration; anything else → pilot.

    The calibrator's `--output` is always a `.json` file and the
    pilot wrapper's `--output-base` is always a directory, so the
    suffix carries the full discriminator."""
    if path.suffix == ".json":
        return MODE_CALIBRATION
    return MODE_PILOT


def _frame_with_left_extras(layout: Layout, snap: RunSnapshot,
                            left_extras: list) -> None:
    """Build the standard 2-column frame: header row, then body with
    `config + progress + usage + *left_extras` in the left column. The
    caller fills `layout["right"]`."""
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
    )
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    layout["left"].split_column(
        Layout(panels.config_panel(snap), name="config"),
        Layout(panels.progress_panel(snap), name="progress", size=6),
        Layout(panels.usage_panel(snap), name="usage", size=10),
        *left_extras,
    )
    layout["header"].update(panels.header_panel(snap))


def render_calibration(snap: RunSnapshot) -> Layout:
    layout = Layout()
    if snap.final is not None:
        f = snap.final
        body = Text.from_markup(
            f"\n[bold green]✓ COMPLETE[/bold green]\n\n"
            f"  n_calibrated : [bold]{f.get('n_calibrated', '?')}[/bold] "
            f"of {f.get('n_candidates', '?')}\n"
            f"  n_blocked    : {f.get('n_blocked', '?')}\n"
            f"  band         : [{f.get('target_lo', '?')}, "
            f"{f.get('target_hi', '?')}]\n"
        )
        layout.update(Panel(body, title=snap.title, border_style="green"))
        return layout

    _frame_with_left_extras(layout, snap, [
        Layout(panels.stages_panel(snap), name="stages", size=8),
    ])
    params = snap.metadata.get("params", {})
    target_lo = params.get("target_lo")
    target_hi = params.get("target_hi")
    if target_lo is not None and target_hi is not None and snap.cells:
        in_band = sum(
            1 for c in snap.cells
            if c.get("kind") == "scored"
            and target_lo <= float(c.get("p_hat", 0.0)) <= target_hi
        )
        scored = sum(1 for c in snap.cells if c.get("kind") == "scored")
        right = Text.from_markup(
            f"\n  scored cells : [bold]{scored}[/bold]\n"
            f"  in band      : [bold green]{in_band}[/bold green]"
            f" ([{target_lo:.2f}, {target_hi:.2f}])\n"
            f"  out of band  : {scored - in_band}\n"
        )
    else:
        right = Text.from_markup("[dim](waiting for cells)[/dim]")
    layout["right"].update(Panel(right, title="cells", border_style="cyan"))
    return layout


def render_pilot(snap: RunSnapshot) -> Layout:
    layout = Layout()
    _frame_with_left_extras(layout, snap, [])
    layout["right"].update(passes_panel(snap))
    return layout


# Per-experiment dashboards register here as `mode -> (source, renderer)`.
SOURCES = {
    MODE_CALIBRATION: (CalibrationSource(title="H3b Calibration"), render_calibration),
    MODE_PILOT: (PilotSource(title="H3b Phase 1A Pilot"), render_pilot),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", type=Path,
                    help="Calibration JSON output path or pilot --output-base "
                         "directory")
    ap.add_argument("--mode", choices=[*_MODES, MODE_AUTO],
                    default=MODE_AUTO,
                    help="Override the source picked from the path shape")
    ap.add_argument("--refresh", type=float, default=5.0,
                    help="Refresh interval in seconds (default: 5)")
    args = ap.parse_args()

    mode = args.mode if args.mode != MODE_AUTO else detect_mode(args.path)
    if mode not in SOURCES:
        print(f"unsupported mode {mode!r}", file=sys.stderr)
        return 1
    source, renderer = SOURCES[mode]

    console = Console()
    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                snap = source.load(args.path)
                live.update(renderer(snap))
                if snap.is_done:
                    time.sleep(2)
                    break
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        console.print("[dim]dashboard exited[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
