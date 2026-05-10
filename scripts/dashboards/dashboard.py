"""Live terminal dashboard for an in-flight experiment run.

Works for any experiment whose source produces a `RunSnapshot`. The
two built-in sources:

- `calibration` — `scripts/calibration/h3b_calibration.py` runs
  (`<output>.tracker/bank_<sha>/cache/` layout via ExperimentTracker).
  Path argument is the calibration JSON output path.

- `pilot` — `scripts/pilots/run_h3b_phase1a.py`-style N-pass runs
  (`<output-base>/pass_NN/data/level_M/neutral/` layout). Path
  argument is the `--output-base` directory.

Mode is auto-detected: a path that is or will be a JSON file dispatches
to `calibration`; a path that is or will be a directory containing
`run_manifest.txt` dispatches to `pilot`. `--mode {calibration,pilot}`
overrides.

Usage:
    python scripts/dashboards/dashboard.py <path> \\
        [--mode {calibration,pilot,auto}] [--refresh 5]

Run from a second terminal while the run is in flight. Exits when the
source reports `is_done` or on Ctrl-C.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from scripts.dashboards import panels  # noqa: E402
from scripts.dashboards.snapshot import RunSnapshot  # noqa: E402
from scripts.dashboards.sources.calibration import CalibrationSource  # noqa: E402
from scripts.dashboards.sources.pilot import PilotSource  # noqa: E402
from scripts.dashboards.sources import pilot_extras  # noqa: E402

MODE_CALIBRATION = "calibration"
MODE_PILOT = "pilot"
MODE_AUTO = "auto"
_MODES = (MODE_CALIBRATION, MODE_PILOT)


def detect_mode(path: Path) -> str:
    """A `.json` path (existent or not) is calibration; any other path
    is pilot. The wrapper's `--output-base` is always a directory; the
    calibrator's `--output` is always a `.json` file. There's no
    overlap, so suffix is a sufficient discriminator."""
    if path.suffix == ".json":
        return MODE_CALIBRATION
    return MODE_PILOT


def _frame_with_left_extras(layout: Layout, snap: RunSnapshot,
                            left_extras: list) -> None:
    """Build the standard 2-column frame: header on top, then body
    with config + progress + usage in the left column followed by any
    `left_extras`. The `right` column is left for the caller to fill."""
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
    layout["right"].update(pilot_extras.passes_panel(snap))
    return layout


SOURCES = {
    MODE_CALIBRATION: (CalibrationSource(), render_calibration),
    MODE_PILOT: (PilotSource(), render_pilot),
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
