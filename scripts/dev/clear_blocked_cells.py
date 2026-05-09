"""Remove blocked cells from a calibration tracker cache.

When the calibration script catches a NonRetryableAPIError (account
quota exhausted, sustained rate-limit thrash, etc.), it writes the cell
as `kind: blocked` and continues. On restart the blocked cell is
treated as cached and skipped, so the candidate is never retried.

This script removes blocked cells from the cache so the next run
re-screens them, optionally filtered by reason substring (default: any
quota-related reason). Scored cells are never touched.

Usage:
    uv run --active python scripts/dev/clear_blocked_cells.py \\
        --cache-dir configs/h3b_calibration_<date>.json.tracker/bank_<sha>/cache \\
        [--reason-contains quota|rate-limit|all] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--reason-contains", default="quota",
                    help="Substring match on cell.reason (case-insensitive). "
                         "Pass 'all' to clear every blocked cell regardless "
                         "of reason. Default: 'quota'.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts without deleting")
    args = ap.parse_args()

    if not args.cache_dir.is_dir():
        print(f"cache dir not found: {args.cache_dir}", file=sys.stderr)
        return 1

    needle = args.reason_contains.lower()
    matched: list[Path] = []
    scored = blocked_other = 0
    for f in sorted(args.cache_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        kind = d.get("kind")
        if kind == "scored":
            scored += 1
            continue
        if kind != "blocked":
            continue
        reason = (d.get("reason") or "").lower()
        if needle == "all" or needle in reason:
            matched.append(f)
        else:
            blocked_other += 1

    print(f"scored cells (kept):           {scored}")
    print(f"blocked cells matching filter: {len(matched)}")
    print(f"blocked cells not matching:    {blocked_other}")

    if not matched:
        return 0

    if args.dry_run:
        print(f"\n(dry-run) would delete {len(matched)} cells")
        return 0

    for f in matched:
        f.unlink()
    print(f"\ndeleted {len(matched)} cells from {args.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
