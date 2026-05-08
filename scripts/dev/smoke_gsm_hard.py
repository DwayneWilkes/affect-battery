"""Smoke-load the GSM-Hard pool from HuggingFace.

Quick sanity check that `load_gsm_hard()` in scripts/banks/source_gsm8k.py
can pull the full canonical mirror (reasoning-machines/gsm-hard, train
split, ~1,319 items) and return rows with the expected schema.

Usage:
    HF_HOME=/tmp/claude-1000/hf-cache uv run --active python \\
        scripts/dev/smoke_gsm_hard.py

Notes:
- HF_HOME defaults to ~/.cache/huggingface, which is read-only inside
  the WSL2 sandbox; redirect to a writable location before running.
- This script is read-only — does not write any bank YAML.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.banks.source_gsm8k import load_gsm_hard


def main() -> int:
    items = load_gsm_hard()
    print(f"GSM-Hard total items: {len(items)}")
    print(f"First item keys: {list(items[0].keys())}")
    print(f"First item id: {items[0].get('id')}")
    print(f"First Q (truncated): {items[0].get('question', '')[:140]}")
    print(f"First expected: {items[0].get('expected')!r}")

    expected_types: dict[str, int] = {}
    for it in items:
        t = type(it.get("expected")).__name__
        expected_types[t] = expected_types.get(t, 0) + 1
    print(f"Expected type distribution: {expected_types}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
