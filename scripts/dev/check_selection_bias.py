"""Re-screen a random sample of calibrated items to quantify selection bias.

Items in the calibrated subset were retained because their measured p̂
landed in [target_lo, target_hi] on a single calibration draw. With
n_reps=100 the per-item p̂ has SE ≈ 0.05, so items near the band edges
may regress *out* of band on a fresh re-screen — a regression-to-the-mean
effect that can bias the H3b contrast.

This script samples N items uniformly from the calibrated bank, scores
each at the same n_reps as the original calibration, and reports:
  - per-item drift (new p̂ − calibrated p̂)
  - count and IDs of items that fell out of band
  - mean drift, p̂ correlation

Usage:
    uv run --active python scripts/dev/check_selection_bias.py \\
        --bank configs/banks/h3b_calibrated_v2.yaml \\
        --provider openai --model gpt-5.4-nano \\
        --n-sample 20 --n-reps 100 \\
        --input-usd-per-million 0.20 --output-usd-per-million 1.25 \\
        --output results/probes/h3b_selection_bias_<YYYY-MM-DD>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.calibration.h3b_calibration import (
    make_client, run_one_candidate,
)


async def _run_sample(items, client, n_reps, max_concurrent):
    """Re-screen items concurrently with a tqdm progress bar. Bar updates
    as candidates complete (asyncio.as_completed yields in finish order),
    so the operator sees progress and an ETA rather than silence until
    the whole gather resolves."""
    from tqdm.asyncio import tqdm as atqdm

    sem = asyncio.Semaphore(max_concurrent)
    tasks = [
        asyncio.create_task(run_one_candidate(client, it, n_reps, sem))
        for it in items
    ]
    results: list = []
    pbar = atqdm(asyncio.as_completed(tasks), total=len(tasks),
                 desc="re-screen", unit="item")
    async for fut in pbar:
        result = await fut
        # Inline status update so the user sees per-item p_hat and
        # in-band status without waiting for the final summary.
        if result["kind"] == "scored":
            p = result["p_hat"]
            in_band = 0.40 <= p <= 0.60
            pbar.set_postfix_str(
                f"last={result['item_id']} p̂={p:.2f} "
                f"{'in' if in_band else 'OUT'}",
                refresh=False,
            )
        else:
            pbar.set_postfix_str(
                f"last={result['item_id']} BLOCKED", refresh=False
            )
        results.append(result)
    # Reorder back to input order so per-item drift comparison aligns
    # with the sample list (otherwise rows are in finish-order).
    by_id = {r.get("item_id"): r for r in results}
    return [by_id[it["id"]] for it in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-sample", type=int, default=20)
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--target-lo", type=float, default=0.40)
    ap.add_argument("--target-hi", type=float, default=0.60)
    ap.add_argument("--max-concurrent", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-usd-per-million", type=float, default=None)
    ap.add_argument("--output-usd-per-million", type=float, default=None)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    bank = yaml.safe_load(args.bank.read_text())
    items = bank["items"]
    rng = random.Random(args.seed)
    sample = rng.sample(items, args.n_sample)
    print(f"sampled {len(sample)} of {len(items)} items from {args.bank.name}",
          file=sys.stderr)

    client = make_client(args.provider, args.model, dry_run=False)
    print(f"re-screening at n_reps={args.n_reps}, max_concurrent={args.max_concurrent}...",
          file=sys.stderr)
    results = asyncio.run(_run_sample(sample, client, args.n_reps, args.max_concurrent))

    per_item = []
    fell_out = 0
    drift_sum = 0.0
    for src, res in zip(sample, results):
        old_p = float(src.get("p_hat_calib", src.get("p_hat", 0.0)))
        if res["kind"] != "scored":
            per_item.append({
                "item_id": src["id"],
                "old_p_hat": old_p,
                "new_p_hat": None,
                "kind": res["kind"],
                "reason": res.get("reason", ""),
            })
            continue
        new_p = float(res["p_hat"])
        in_band_old = args.target_lo <= old_p <= args.target_hi
        in_band_new = args.target_lo <= new_p <= args.target_hi
        if in_band_old and not in_band_new:
            fell_out += 1
        per_item.append({
            "item_id": src["id"],
            "old_p_hat": old_p,
            "new_p_hat": new_p,
            "drift": new_p - old_p,
            "in_band_old": in_band_old,
            "in_band_new": in_band_new,
        })
        drift_sum += new_p - old_p

    scored = [r for r in per_item if r.get("new_p_hat") is not None]
    if scored:
        mean_drift = drift_sum / len(scored)
        max_drift = max(abs(r["drift"]) for r in scored)
    else:
        mean_drift = max_drift = 0.0

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bank": str(args.bank),
        "n_sample": args.n_sample,
        "n_scored": len(scored),
        "n_blocked": len(per_item) - len(scored),
        "n_fell_out_of_band": fell_out,
        "pct_fell_out_of_band": 100.0 * fell_out / max(len(scored), 1),
        "mean_drift": mean_drift,
        "max_abs_drift": max_drift,
        "target_band": [args.target_lo, args.target_hi],
        "n_reps": args.n_reps,
        "per_item": per_item,
    }
    if hasattr(client, "usage_summary"):
        kwargs = {}
        if (args.input_usd_per_million is not None
                and args.output_usd_per_million is not None):
            kwargs["input_usd_per_million"] = args.input_usd_per_million
            kwargs["output_usd_per_million"] = args.output_usd_per_million
        summary["usage"] = client.usage_summary(**kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    print("", file=sys.stderr)
    print(f"  scored:        {len(scored)} / {args.n_sample}", file=sys.stderr)
    print(f"  fell out:      {fell_out}  ({summary['pct_fell_out_of_band']:.1f}%)",
          file=sys.stderr)
    print(f"  mean drift:    {mean_drift:+.4f}", file=sys.stderr)
    print(f"  max abs drift: {max_drift:.4f}", file=sys.stderr)
    if "usage" in summary and summary["usage"].get("estimated_usd") is not None:
        print(f"  cost:          ${summary['usage']['estimated_usd']:.4f}",
              file=sys.stderr)
    print(f"  report:        {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
