"""Dashboard source for `scripts/pilots/run_h3b_phase1a.py`-style
N-pass pilot runs. Reads the wrapper's `run_manifest.txt`, per-pass
`pass_NN/manifest.yaml`, and per-cell JSONs under
`pass_NN/data/level_M/neutral/`."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from scripts.dashboards.snapshot import RunSnapshot


_MANIFEST_LINE = re.compile(r"^([\w/\s-]+?):\s+(.+?)\s*$")


def _parse_run_manifest(path: Path) -> dict:
    """Parse the wrapper's plain-text run manifest into a dict.

    The wrapper writes `<key>: <value>` lines (with whitespace
    padding); we strip the padding and coerce numeric fields. Returns
    `{}` when the file is missing so callers can render a 'waiting'
    panel during early startup."""
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
    """Count `*.json` cells under `pass_NN/data/level_*/neutral/`."""
    data = pass_dir / "data"
    if not data.is_dir():
        return 0
    return sum(1 for _ in data.glob("level_*/neutral/*.json"))


def _load_pass_manifest(pass_dir: Path) -> dict:
    md = pass_dir / "manifest.yaml"
    if not md.is_file():
        return {}
    try:
        return yaml.safe_load(md.read_text()) or {}
    except yaml.YAMLError:
        return {}


class PilotSource:
    """Constructs a `RunSnapshot` from the pilot wrapper's output dir."""

    def load(self, output_base: Path) -> RunSnapshot:
        run_md = _parse_run_manifest(output_base / "run_manifest.txt")
        n_passes = int(run_md.get("n_passes", 0))
        n_items = int(run_md.get("n_bank_items", 0))
        n_levels = int(run_md.get("n_levels", 0))
        expected_per_pass = n_items * n_levels
        cells_total = n_passes * expected_per_pass

        # Discover existing pass dirs and compute per-pass status.
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

        # Surface every requested pass slot in extras (including
        # not-yet-dispatched ones), so the per-pass grid renders the
        # full picture without holes.
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
            title="H3b Phase 1A Pilot",
            cells_done=cells_done,
            cells_total=cells_total,
            metadata={"params": params, "metrics": {}, "stages": {}},
            cells=[],
            extras={"passes": passes, "expected_per_pass": expected_per_pass},
        )
