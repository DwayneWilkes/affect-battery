"""Carry over per-item cache cells from an old bank's tracker dir to a new bank's.

When the bank YAML is replaced with a content-superset (e.g. expanding
a sampled bank to the full source pool), the calibration script's
per-bank cache fingerprint changes and the new run would re-screen
every item unless an explicit bridge is performed.

This script matches each old cache cell to a new-bank item by content
(question + expected, looked up via the OLD BANK YAML — bank-driven, not
cell-driven, so blocked cells without question/expected fields still
carry over). When a match exists, the cell is copied into the new
cache dir under its NEW item_id, with the cell's `item_id` field
rewritten in place so downstream consumers stay consistent. Existing
cells in the new cache are never overwritten; the cache-isolation
invariant (do not overwrite existing cells when bridging) is preserved.

Usage:
    uv run --active python scripts/dev/carry_over_cache.py \\
        --from-bank configs/banks/gsm8k_v1.yaml \\
        --from-cache-dir configs/h3b_calibration_2026-05-08.json.tracker/bank_<sha>/cache \\
        --to-bank configs/banks/gsm_hard_full_2026-05-08.yaml \\
        --to-cache-dir configs/h3b_calibration_<NEW>.json.tracker/bank_<NEW_sha>/cache \\
        [--dry-run]

Use `inspect_bank_overlap.py` first to confirm the new bank is a
content-superset of the old one before running this.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass
class CarryOverResult:
    """Summary of a carry-over run.

    `matched`: old cells whose item_id resolved to a new-bank item.
    `copied`: cells actually written to the new cache.
    `would_copy`: cells that would have been copied (only meaningful in dry-run).
    `already_cached`: matched cells where the new cache already had an entry; left alone.
    `unmatched`: cells whose old-bank entry has no matching content in new bank.
    `missing_in_old_bank`: cache files whose item_id isn't in the old bank YAML.
    """
    matched: int = 0
    copied: int = 0
    would_copy: int = 0
    already_cached: int = 0
    unmatched: int = 0
    missing_in_old_bank: int = 0


def _build_new_bank_index(new_bank: Iterable[dict]) -> dict[tuple[str, str], str]:
    """Map (question, expected) → new_item_id. Last-write-wins on collisions
    (callers should validate uniqueness with `inspect_bank_overlap.py`)."""
    out: dict[tuple[str, str], str] = {}
    for it in new_bank:
        out[(it["question"], it["expected"])] = it["id"]
    return out


def carry_over_cache(
    old_bank: list[dict],
    old_cache_dir: Path,
    new_bank: list[dict],
    new_cache_dir: Path,
    *,
    dry_run: bool = False,
) -> CarryOverResult:
    """Match old cache cells to new-bank items and copy with rewritten IDs.

    Bank-driven matching: each old cache filename gives an item_id; we
    look up that id in the OLD BANK to get its (question, expected),
    then look up the new bank for a matching content tuple. The cache
    cell's `item_id` field is rewritten to the new bank's id when copied.
    """
    new_idx = _build_new_bank_index(new_bank)
    old_by_id = {it["id"]: it for it in old_bank}

    result = CarryOverResult()
    if not old_cache_dir.is_dir():
        return result
    if not dry_run:
        new_cache_dir.mkdir(parents=True, exist_ok=True)

    for cache_file in sorted(old_cache_dir.glob("*.json")):
        old_id = cache_file.stem
        old_item = old_by_id.get(old_id)
        if old_item is None:
            result.missing_in_old_bank += 1
            continue
        key = (old_item["question"], old_item["expected"])
        new_id = new_idx.get(key)
        if new_id is None:
            result.unmatched += 1
            continue
        result.matched += 1

        target = new_cache_dir / f"{new_id}.json"
        if target.exists():
            result.already_cached += 1
            continue

        if dry_run:
            result.would_copy += 1
            continue

        cell = json.loads(cache_file.read_text())
        cell["item_id"] = new_id
        target.write_text(json.dumps(cell, indent=2, default=str))
        result.copied += 1

    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-bank", required=True, type=Path,
                    help="Old bank YAML (source of question/expected for old cells)")
    ap.add_argument("--from-cache-dir", required=True, type=Path,
                    help="Old per-item cache dir (.../bank_<sha>/cache)")
    ap.add_argument("--to-bank", required=True, type=Path,
                    help="New bank YAML (the destination's content source)")
    ap.add_argument("--to-cache-dir", required=True, type=Path,
                    help="New per-item cache dir to populate")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report matches without writing")
    args = ap.parse_args()

    old_bank = yaml.safe_load(args.from_bank.read_text())["items"]
    new_bank = yaml.safe_load(args.to_bank.read_text())["items"]
    print(
        f"old bank: {len(old_bank)} items in {args.from_bank.name}",
        file=sys.stderr,
    )
    print(
        f"new bank: {len(new_bank)} items in {args.to_bank.name}",
        file=sys.stderr,
    )

    result = carry_over_cache(
        old_bank, args.from_cache_dir, new_bank, args.to_cache_dir,
        dry_run=args.dry_run,
    )
    label = "(dry-run)" if args.dry_run else ""
    print(f"\nCarry-over summary {label}", file=sys.stderr)
    print(f"  matched              : {result.matched}", file=sys.stderr)
    print(f"  copied               : {result.copied}", file=sys.stderr)
    print(f"  would copy           : {result.would_copy}", file=sys.stderr)
    print(f"  already cached       : {result.already_cached}", file=sys.stderr)
    print(f"  unmatched            : {result.unmatched}", file=sys.stderr)
    print(f"  missing in old bank  : {result.missing_in_old_bank}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
