"""Report exact (question, expected) overlap between two bank YAMLs.

Useful before running cache carry-over: confirms the new bank's items
are a superset of the old bank's by content, so cached scores from the
old run can be safely reused under the new bank's item IDs.

Usage:
    uv run --active python scripts/dev/inspect_bank_overlap.py \\
        --old configs/banks/gsm8k_v1.yaml \\
        --new configs/banks/gsm_hard_full_2026-05-08.yaml \\
        [--source-filter gsm-hard]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True, help="Old bank YAML")
    ap.add_argument("--new", type=Path, required=True, help="New bank YAML")
    ap.add_argument("--source-filter", default=None,
                    help="Restrict comparison to items with source_dataset==FILTER")
    args = ap.parse_args()

    old = yaml.safe_load(args.old.read_text())["items"]
    new = yaml.safe_load(args.new.read_text())["items"]
    if args.source_filter:
        old = [it for it in old if it.get("source_dataset") == args.source_filter]
        new = [it for it in new if it.get("source_dataset") == args.source_filter]
    print(f"old: {len(old)} items in {args.old.name}")
    print(f"new: {len(new)} items in {args.new.name}")

    new_idx: dict[tuple[str, str], list[str]] = {}
    for it in new:
        key = (it["question"], it["expected"])
        new_idx.setdefault(key, []).append(it["id"])

    matched = 0
    unmatched: list[str] = []
    for it in old:
        key = (it["question"], it["expected"])
        if key in new_idx:
            matched += 1
        else:
            unmatched.append(it["id"])

    print(f"matched (exact q+expected):  {matched}")
    print(f"unmatched (old not in new):  {len(unmatched)}")
    if unmatched[:3]:
        print("  first few unmatched ids:", unmatched[:3])

    dupes = [(k, ids) for k, ids in new_idx.items() if len(ids) > 1]
    print(f"duplicate (q,e) pairs in new bank: {len(dupes)}")
    return 0 if matched == len(old) and not dupes else 1


if __name__ == "__main__":
    sys.exit(main())
