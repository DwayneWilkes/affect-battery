"""Compact a verbose block-style YAML to mixed flow/block style.

Re-serializes each `items:` entry as a single-line flow mapping while
keeping the file's top-level metadata block-styled for readability.
Input YAML must round-trip via PyYAML — this script does NOT preserve
comments. Run only on machine-generated bank YAMLs (arithmetic_*.yaml,
folio_active.yaml, etc.).

Usage:
    python -m scripts.compact_yaml configs/banks/arithmetic_easy_v1.yaml
    python -m scripts.compact_yaml --output <out.yaml> <in.yaml>

Idempotent: running twice on the same file produces identical output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def compact(path: Path, output: Path | None = None) -> Path:
    bank = yaml.safe_load(path.read_text())
    if "items" not in bank or not isinstance(bank["items"], list):
        raise SystemExit(
            f"error: {path} has no `items:` list to compact"
        )

    items = bank.pop("items")

    # Top-level metadata: block style (default), preserves readability.
    head = yaml.safe_dump(bank, sort_keys=False, allow_unicode=True)

    # Items: each rendered as a single-line flow mapping. We want one
    # item per line so grep/diff still works.
    lines: list[str] = ["items:"]
    for item in items:
        flow = yaml.safe_dump(
            item,
            default_flow_style=True,
            allow_unicode=True,
            width=10_000,  # never wrap
        ).rstrip()
        # safe_dump wraps the dict in {} and adds a trailing newline; we
        # want a leading hyphen + the bracketed payload.
        lines.append(f"- {flow}")

    out_path = output or path
    out_path.write_text(head + "\n".join(lines) + "\n")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="compact_yaml")
    p.add_argument("path", type=Path, help="Input YAML")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path (defaults to overwriting input)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = compact(args.path, args.output)
    new_lines = sum(1 for _ in out.read_text().splitlines())
    print(f"compacted {args.path} -> {out} ({new_lines} lines)")
