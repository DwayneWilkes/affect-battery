"""Reclassify tdc-era Qwen runs as pilot data.

Per design.md migration plan + GAPS.md G12: tdc-era results get
`is_pilot: true` flag added to config so downstream analysis excludes
them from primary corpora.

Safety properties:
- Originals backed up to `<name>.orig` before mutation.
- Original checksum preserved in `orig_checksum` field of migrated payload.
- Idempotent: re-running on an already-migrated file is a no-op.
- Selective: only Qwen models get migrated; Llama/Mistral/Gemma runs
  are untouched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable


def is_tdc_era_result(path: Path) -> bool:
    """Return True iff the result file at `path` is a tdc-era Qwen run
    that has not yet been flagged as pilot."""
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    config = data.get("config") or {}
    if config.get("is_pilot") is True:
        return False
    model = (config.get("model_name") or config.get("model") or "").lower()
    return "qwen" in model


def migrate_result_file(path: Path) -> bool:
    """Add `is_pilot: True` to the result config, preserve the original
    checksum as `orig_checksum`, back up original to `.orig`.

    Returns True if migrated, False if skipped (idempotent no-op).
    """
    if not is_tdc_era_result(path):
        return False

    backup = path.with_suffix(".orig")
    shutil.copy2(path, backup)

    data = json.loads(path.read_text())
    data.setdefault("orig_checksum", data.get("checksum", ""))
    data.setdefault("config", {})["is_pilot"] = True
    path.write_text(json.dumps(data, indent=2))
    return True


def walk_and_migrate(results_dir: Path) -> list[Path]:
    """Walk results_dir recursively, migrate every tdc-era file.

    Returns the list of paths that were migrated (excluding skipped /
    already-flagged files).
    """
    migrated: list[Path] = []
    if not results_dir.exists():
        return migrated
    for path in sorted(results_dir.rglob("*.json")):
        # skip .orig backups themselves
        if path.suffix == ".orig" or ".orig" in path.suffixes:
            continue
        if migrate_result_file(path):
            migrated.append(path)
    return migrated


def _cli() -> None:
    """Optional CLI entry point: `python -m src.migrations.reclassify_tdc_runs results/`."""
    import argparse
    p = argparse.ArgumentParser(prog="migrate-tdc-runs")
    p.add_argument("results_dir", type=Path)
    p.add_argument("--dry-run", action="store_true",
                   help="List files that would be migrated without modifying.")
    args = p.parse_args()

    if args.dry_run:
        for path in args.results_dir.rglob("*.json"):
            if path.suffix == ".orig":
                continue
            if is_tdc_era_result(path):
                print(f"would migrate: {path}")
        return

    migrated = walk_and_migrate(args.results_dir)
    print(f"Migrated {len(migrated)} files:")
    for p in migrated:
        print(f"  {p}")


if __name__ == "__main__":
    _cli()
