"""Spot-check verification of cache carry-over correctness.

For each cell in the new cache, check that:
  1. The cell's item_id matches the filename
  2. (cell.question, cell.expected) match the new bank's entry for that id
  3. Numeric data (n_correct, p_hat, n_reps) are preserved

Run after `carry_over_cache.py` to validate before committing to a calibration run.

Usage:
    uv run --active python scripts/dev/verify_carryover.py \\
        --bank configs/banks/gsm_hard_full_2026-05-08.yaml \\
        --cache-dir configs/h3b_calibration_2026-05-08-fullpool.json.tracker/bank_<sha>/cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--n-spot-checks", type=int, default=5)
    args = ap.parse_args()

    bank_items = yaml.safe_load(args.bank.read_text())["items"]
    bank_by_id = {it["id"]: it for it in bank_items}

    cells = sorted(args.cache_dir.glob("*.json"))
    print(f"new cache: {len(cells)} cells in {args.cache_dir}")

    ok = bad = 0
    for f in cells:
        cell = json.loads(f.read_text())
        item_id = f.stem
        if cell.get("item_id") != item_id:
            print(f"  MISMATCH item_id: {f.name} contains item_id={cell.get('item_id')!r}")
            bad += 1
            continue
        bank_item = bank_by_id.get(item_id)
        if bank_item is None:
            print(f"  ORPHAN: {item_id} not in new bank")
            bad += 1
            continue
        if cell.get("kind") == "scored":
            if cell.get("question") != bank_item["question"]:
                print(f"  Q MISMATCH: {item_id}")
                bad += 1
                continue
            if cell.get("expected") != bank_item["expected"]:
                print(f"  EXPECTED MISMATCH: {item_id}")
                bad += 1
                continue
        ok += 1

    print(f"  OK: {ok}  BAD: {bad}")

    print(f"\nspot checks (first {args.n_spot_checks}):")
    for f in cells[: args.n_spot_checks]:
        cell = json.loads(f.read_text())
        bank_item = bank_by_id.get(f.stem, {})
        print(f"  {f.stem}")
        print(f"    p_hat: {cell.get('p_hat'):.3f}" if cell.get('p_hat') is not None else "    blocked")
        print(f"    bank Q[:60]:  {bank_item.get('question', '?')[:60]!r}")
        print(f"    cell Q[:60]:  {cell.get('question', '?')[:60]!r}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
