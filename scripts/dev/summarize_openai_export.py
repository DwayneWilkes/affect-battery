"""Summarize an OpenAI usage + cost export.

Aggregates the per-minute completions usage export (`completions_usage_*.json`)
by model + computes the implied per-call cost from the matched daily-cost
export (`cost_*.json`). Use after dropping fresh exports from
platform.openai.com/usage to recalibrate the project's pricing
assumptions.

Usage:
    uv run --active python scripts/dev/summarize_openai_export.py \\
        --usage <usage_export>.json \\
        --cost <cost_export>.json \\
        [--model-prefix gpt-5.4-nano]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", type=Path, required=True,
                    help="completions_usage_*.json from OpenAI dashboard export")
    ap.add_argument("--cost", type=Path, required=True,
                    help="cost_*.json from OpenAI dashboard export")
    ap.add_argument("--model-prefix", default=None,
                    help="Restrict to model IDs starting with this prefix")
    args = ap.parse_args()

    usage = json.loads(args.usage.read_text())
    cost = json.loads(args.cost.read_text())
    total_usd = sum(
        float(r["amount"]["value"])
        for bucket in cost.get("data", [])
        for r in bucket.get("results", [])
    )

    by_model: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "in": 0, "out": 0, "in_cached": 0}
    )
    for bucket in usage.get("data", []):
        for r in bucket.get("results", []):
            m = r["model"]
            if args.model_prefix and not m.startswith(args.model_prefix):
                continue
            by_model[m]["calls"] += r.get("num_model_requests", 0)
            by_model[m]["in"] += r.get("input_tokens", 0)
            by_model[m]["out"] += r.get("output_tokens", 0)
            by_model[m]["in_cached"] += r.get("input_cached_tokens", 0)

    print(f"total cost across export window: ${total_usd:.4f}")
    print()
    headers = ("model", "calls", "input", "output", "in/call", "out/call", "$/call")
    print(f"{headers[0]:<35} {headers[1]:>8} {headers[2]:>12} {headers[3]:>12} "
          f"{headers[4]:>9} {headers[5]:>9} {headers[6]:>12}")

    grand_calls = sum(v["calls"] for v in by_model.values())
    for m, v in sorted(by_model.items(), key=lambda kv: -kv[1]["calls"]):
        if v["calls"] == 0:
            continue
        # Cost is reported at the org level, not per-model. When only one
        # model dominates the window, $/call is informative; with mixed
        # models the column is pro-rated by call share.
        share = v["calls"] / grand_calls if grand_calls else 0
        per_call = (total_usd * share) / v["calls"] if v["calls"] else 0
        print(
            f"{m:<35} {v['calls']:>8,} {v['in']:>12,} {v['out']:>12,} "
            f"{v['in']/v['calls']:>9.1f} {v['out']/v['calls']:>9.1f} "
            f"${per_call:>11.6f}"
        )

    if grand_calls:
        avg_in = sum(v["in"] for v in by_model.values()) / grand_calls
        avg_out = sum(v["out"] for v in by_model.values()) / grand_calls
        avg_cost = total_usd / grand_calls
        print()
        print(f"aggregate avg cost / call : ${avg_cost:.6f}")
        print(f"aggregate avg in / call   : {avg_in:.1f} tokens")
        print(f"aggregate avg out / call  : {avg_out:.1f} tokens")
        total_in = sum(v["in"] for v in by_model.values())
        total_out = sum(v["out"] for v in by_model.values())

        # Verify the published gpt-5.4-nano rates ($0.20 in / $1.25 out per
        # 1M, standard tier) against the actual cost — drift here is a
        # signal that rates changed or the model in the export is on a
        # different tier (batch / flex / priority).
        published_in = 0.20
        published_out = 1.25
        predicted = (total_in * published_in + total_out * published_out) / 1_000_000
        drift = predicted - total_usd
        drift_pct = 100.0 * drift / total_usd if total_usd else 0
        print()
        print("verification against published gpt-5.4-nano standard rates "
              f"(${published_in:.2f}/M in, ${published_out:.2f}/M out):")
        print(f"  predicted    : ${predicted:.4f}")
        print(f"  observed     : ${total_usd:.4f}")
        print(f"  drift        : ${drift:+.4f}  ({drift_pct:+.1f}%)")
        if abs(drift_pct) > 5:
            print("  ⚠ drift > 5%: rates may have changed or the export "
                  "spans a non-standard tier")

        print()
        print("implied output rate ($/1M) at other input-rate hypotheses:")
        for in_rate in (0.05, 0.10, 0.15, 0.20, 0.25):
            out_cost = total_usd - total_in / 1_000_000 * in_rate
            out_rate = out_cost / (total_out / 1_000_000) if total_out else 0
            print(f"  if input ${in_rate:.2f}/M → output ${out_rate:.3f}/M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
