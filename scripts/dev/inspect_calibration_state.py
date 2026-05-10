"""Inspect a bank YAML and (optionally) a calibration JSON side by side.

Useful when diagnosing grading mismatches: e.g., does the bank's
`expected` field format (`'4933828'` vs `'4933828.0'`) match the format
that produced in-band p̂ values during a prior calibration run?

Usage:
    uv run --active python scripts/dev/inspect_calibration_state.py \\
        --bank configs/banks/gsm8k_v1.yaml \\
        [--calibration configs/h3b_calibration_2026-05-08.json] \\
        [--source gsm-hard] \\
        [--n 5]

Read-only — does not modify any files.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True,
                    help="Path to a bank YAML")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="Optional calibration JSON to summarize")
    ap.add_argument("--source", type=str, default=None,
                    help="Filter bank items by source_dataset (e.g. gsm-hard)")
    ap.add_argument("--n", type=int, default=5,
                    help="How many sample items to print")
    args = ap.parse_args()

    bank = yaml.safe_load(args.bank.read_text())
    items = bank["items"]
    if args.source:
        items = [it for it in items if it.get("source_dataset") == args.source]
    label = f"{args.source or 'all'} items in {args.bank.name}"
    print(f"{label}: {len(items)} total; first {args.n}:")
    for it in items[: args.n]:
        print(f"  {it['id']}  expected={it['expected']!r}  diff={it.get('difficulty')}")

    if args.calibration and args.calibration.exists():
        cal = json.loads(args.calibration.read_text())
        n_cal = cal.get("n_calibrated", "?")
        print(f"\nIn-band items from {args.calibration.name} (n={n_cal}); first {args.n}:")
        for it in cal.get("calibrated_subset", [])[: args.n]:
            p_hat = it.get("p_hat")
            p_str = f"{p_hat:.3f}" if isinstance(p_hat, (int, float)) else "?"
            print(f"  {it.get('item_id')}  expected={it.get('expected')!r}  p_hat={p_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
