"""One-shot snapshot of an in-flight H3b calibration.

Reads the tracker dir directly (no live API calls) and prints:
  - cache progress (cached / target, blocked count)
  - in-band count + floor status
  - cost burn (if pricing was passed to the run) + per-call avg
  - elapsed wall-clock + projected remaining

Designed to be cheap and re-runnable: each invocation reads disk state
once and exits. Pair with the live dashboard, or run from a third
terminal during long calibrations.

Usage:
    uv run --active python scripts/dev/check_calibration_status.py \\
        configs/h3b_calibration_<YYYY-MM-DD>.json [--min-items 32]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.lib.tracker_io import (  # noqa: E402
    find_bank_subdir, load_cache_items, load_run_metadata, tracker_root_for,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_path", type=Path,
                    help="Calibration JSON output path (same value passed "
                         "to h3b_calibration.py --output)")
    ap.add_argument("--min-items", type=int, default=32,
                    help="In-band floor; status reports met / not-met")
    args = ap.parse_args()

    if args.output_path.is_file():
        final = json.loads(args.output_path.read_text())
        print(f"✓ COMPLETE: {args.output_path}")
        print(f"  n_calibrated : {final.get('n_calibrated')} / "
              f"{final.get('n_candidates')}")
        print(f"  n_blocked    : {final.get('n_blocked')}")
        if final.get("usage"):
            u = final["usage"]
            print(f"  api calls    : {u.get('n_calls', 0):,}")
            if "estimated_usd" in u:
                print(f"  cost         : ${u['estimated_usd']:.4f}")
        return 0

    tracker_root = tracker_root_for(args.output_path)
    bank_dir = find_bank_subdir(tracker_root)
    if bank_dir is None:
        print(f"no tracker dir at {tracker_root} — calibration hasn't started yet")
        return 0

    md = load_run_metadata(bank_dir)
    params = md.get("params", {})
    metrics = md.get("metrics", {})
    target = int(params.get("n_candidates", 0))

    cells = load_cache_items(bank_dir)
    if not cells:
        print(f"tracker exists but cache empty (yet) — {bank_dir}")
        return 0

    scored = blocked = in_band = 0
    p_hats = []
    for d in cells:
        kind = d.get("kind")
        if kind == "scored":
            scored += 1
            p = float(d.get("p_hat", 0.0))
            p_hats.append(p)
            if 0.40 <= p <= 0.60:
                in_band += 1
        elif kind == "blocked":
            blocked += 1

    n_cached = len(cells)
    pct = 100.0 * n_cached / target if target else 0.0
    last_t = max(d["_mtime"] for d in cells)
    # Anchor elapsed on the tracker's `created_at` so cache carry-over
    # (mass-copy of cells from a prior run) doesn't bias the rate
    # downward. Fall back to oldest cell mtime if the field is missing.
    created_iso = md.get("created_at")
    if created_iso:
        try:
            created_dt = datetime.fromisoformat(created_iso.replace("Z", "+00:00"))
            first_t = created_dt.timestamp()
        except Exception:
            first_t = min(d["_mtime"] for d in cells)
    else:
        first_t = min(d["_mtime"] for d in cells)
    elapsed = max(time.time() - first_t, 1)
    # Count only cells modified after `created_at` for the rate calc
    # (skip carryover cells that pre-date the run start).
    fresh_cells = [d for d in cells if d["_mtime"] >= first_t]
    rate_per_min = len(fresh_cells) * 60.0 / elapsed
    remaining = max(0, target - n_cached)
    eta_sec = remaining / max(rate_per_min / 60.0, 1e-9)

    print(f"bank tracker  : {bank_dir.name}")
    print(f"cached        : {n_cached:,} / {target:,}  ({pct:.1f}%)")
    print(f"  scored      : {scored:,}")
    print(f"  blocked     : {blocked:,}  "
          f"{'⚠' if blocked else ''}")
    print(f"in-band       : {in_band}  (floor={args.min_items}, "
          f"{'✓ met' if in_band >= args.min_items else 'not yet'})")
    if scored:
        yield_pct = 100.0 * in_band / scored
        proj = round(yield_pct * target / 100.0)
        print(f"yield         : {yield_pct:.1f}%  →  projected ~{proj} at {target}")
    n_carryover = n_cached - len(fresh_cells)
    print(f"elapsed       : {elapsed/60:.1f} min "
          f"(run start {time.strftime('%H:%M:%S', time.localtime(first_t))} local, "
          f"latest cell {time.strftime('%H:%M:%S', time.localtime(last_t))})")
    if n_carryover:
        print(f"  (rate excludes {n_carryover} carry-over cells)")
    print(f"rate          : {rate_per_min:.2f} cand/min "
          f"({len(fresh_cells)} fresh cells)")
    if remaining:
        print(f"ETA           : ~{eta_sec/60:.0f} min "
              f"({eta_sec/3600:.1f} hr)")

    if "usage_n_calls" in metrics:
        n_calls = int(metrics["usage_n_calls"])
        cost = metrics.get("usage_estimated_usd")
        prompt = int(metrics.get("usage_prompt_tokens", 0))
        compl = int(metrics.get("usage_completion_tokens", 0))
        reasoning = int(metrics.get("usage_reasoning_tokens", 0))
        print(f"api calls     : {n_calls:,}")
        print(f"  prompt tk   : {prompt:,}")
        # `completion_tokens` already includes reasoning tokens for
        # reasoning models; reasoning is reported as a breakdown.
        print(f"  output tk   : {compl:,}  ({reasoning:,} reasoning)")
        if cost is not None:
            per_call = float(cost) / max(n_calls, 1)
            # Project remaining cost at observed avg.
            remaining_calls = remaining * params.get("n_reps", 100)
            proj_remaining = remaining_calls * per_call
            print(f"  cost so far : ${float(cost):.4f}  "
                  f"(${per_call:.6f}/call)")
            print(f"  proj total  : ${float(cost) + proj_remaining:.4f}  "
                  f"(${proj_remaining:.4f} remaining)")
        else:
            print("  cost        : pricing flags not passed to calibration")
    return 0


if __name__ == "__main__":
    sys.exit(main())
