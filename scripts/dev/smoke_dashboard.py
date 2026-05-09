"""Render one frame of the H3b dashboard against a synthetic tracker dir.

Useful when iterating on the dashboard layout without firing a real
calibration run. Writes a fake bank cache + run_metadata.json to a temp
dir, then calls `render()` once and prints the resulting layout to the
console.

Usage:
    uv run --active python scripts/dev/smoke_dashboard.py [--n-scored 30]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console

from scripts.calibration.dashboard_h3b import render
from src.lib.tracker_io import tracker_root_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scored", type=int, default=30,
                    help="Synthetic scored items in the cache")
    ap.add_argument("--n-blocked", type=int, default=2,
                    help="Synthetic blocked items in the cache")
    ap.add_argument("--n-target", type=int, default=1319,
                    help="Total candidate target (for projection math)")
    ap.add_argument("--min-items", type=int, default=25)
    args = ap.parse_args()

    rng = random.Random(42)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "calib.json"
        tracker_root = tracker_root_for(out_path)
        bank_dir = tracker_root / "bank_abc123def456"
        cache_dir = bank_dir / "cache"
        cache_dir.mkdir(parents=True)

        # Synthetic usage telemetry, sized as if we're partway through:
        # ~150 input + ~280 output tokens per call (nano + reasoning),
        # at illustrative pricing.
        # Token counts per call calibrated against the May 2026 export
        # (~65 input, ~176 output incl. reasoning). Pricing is gpt-5.4-nano
        # standard tier ($0.20/M in, $1.25/M out, verified to $0.0000 drift
        # against the actual invoice).
        n_calls_so_far = args.n_scored * 100  # n_reps assumed 100 per item
        synth_usage = {
            "usage_n_calls": n_calls_so_far,
            "usage_prompt_tokens": n_calls_so_far * 65,
            "usage_completion_tokens": n_calls_so_far * 30,
            "usage_reasoning_tokens": n_calls_so_far * 146,
            "usage_estimated_usd": (
                n_calls_so_far * 65 * 0.20 / 1_000_000
                + n_calls_so_far * (30 + 146) * 1.25 / 1_000_000
            ),
        }
        run_md = {
            "params": {
                "bank": "configs/banks/gsm_hard_full_2026-05-08.yaml",
                "provider": "openai",
                "model": "gpt-5.4-nano",
                "target_lo": 0.40,
                "target_hi": 0.60,
                "n_candidates": args.n_target,
                "n_reps": 100,
                "max_concurrent": 50,
                "candidates_per_batch": 20,
            },
            "metrics": synth_usage,
            "stages": {
                "load_bank": {"duration_seconds": 0.42},
                "filter_difficulty": {"duration_seconds": 0.05},
                "score_candidates": {"duration_seconds": None},
            },
        }
        (bank_dir / "run_metadata.json").write_text(json.dumps(run_md, indent=2))

        now = time.time()
        for i in range(args.n_scored):
            # Sample p̂ from a beta-ish bias toward the band edges so the
            # dashboard exercises both above/below shading.
            r = rng.random()
            if r < 0.25:
                p = rng.uniform(0.40, 0.60)
            elif r < 0.65:
                p = rng.uniform(0.0, 0.40)
            else:
                p = rng.uniform(0.60, 1.0)
            cell = {
                "kind": "scored",
                "item_id": f"gsm_hard_{i:04d}",
                "question": "synthetic",
                "expected": "1234",
                "n_reps": 100,
                "n_correct": int(p * 100),
                "p_hat": round(p, 2),
                "n_blocked_reps": 0,
                "n_error_reps": 0,
            }
            path = cache_dir / f"item_{i:04d}.json"
            path.write_text(json.dumps(cell))
            # Stagger mtime so recent-window stats are non-trivial.
            t = now - (args.n_scored - i) * 30.0
            import os
            os.utime(path, (t, t))

        for i in range(args.n_blocked):
            cell = {
                "kind": "blocked",
                "item_id": f"gsm_hard_block_{i:03d}",
                "reason": "rate_limit_after_retries",
            }
            path = cache_dir / f"blocked_{i:03d}.json"
            path.write_text(json.dumps(cell))

        ns = SimpleNamespace(
            output_path=out_path,
            refresh=5.0,
            expected_total=None,
            min_items=args.min_items,
        )
        console = Console()
        done, layout = render(ns, console)
        console.print(layout)
        print(f"\n[done={done}]  rendered against {bank_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
