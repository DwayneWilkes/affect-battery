"""Re-score an exp3a corpus under the updated extract_numeric_answer.

The original extractor picked the FIRST '=' match in chain-of-thought
responses, capturing an intermediate calculation rather than the final
answer. The fixed extractor picks the LAST match in each priority tier.

This script reads each exp3a result JSON, re-runs the scorer on the
preserved model_response, and writes a parallel corpus with updated
binary_correct values. The original corpus is preserved.

Usage:
    uv run python scripts/analysis/rescore_exp3a.py \\
        --input results/h3a_2026-04-27_n122 \\
        --output results/h3a_2026-04-27_n122_rescored
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scoring.accuracy import extract_numeric_answer  # noqa: E402


def _score(response: str, expected: str) -> int:
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return 0
    try:
        target = float(expected)
    except (TypeError, ValueError):
        return 0
    return int(abs(extracted - target) < 0.01)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"error: input dir not found: {args.input}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    if (args.input / "manifest.yaml").exists():
        shutil.copy(args.input / "manifest.yaml", args.output / "manifest.yaml")

    n_total = 0
    n_changed = 0
    flips_to_correct = 0
    flips_to_incorrect = 0

    for src in args.input.rglob("*.json"):
        rel = src.relative_to(args.input)
        if rel.parts[0] not in ("data",):
            continue
        try:
            data = json.loads(src.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment_type") != "exp3a":
            continue

        body = data.get("body") or {}
        response = body.get("model_response", "")
        expected = body.get("expected_answer", "")
        old = int(body.get("binary_correct", 0))
        new = _score(response, expected)

        n_total += 1
        if new != old:
            n_changed += 1
            if new == 1 and old == 0:
                flips_to_correct += 1
            else:
                flips_to_incorrect += 1
            body["binary_correct"] = new
            data["body"] = body

        dst = args.output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, indent=2))

    print(f"Re-scored {n_total} cells", file=sys.stderr)
    print(f"  changed: {n_changed}", file=sys.stderr)
    print(f"  flipped 0 -> 1: {flips_to_correct}", file=sys.stderr)
    print(f"  flipped 1 -> 0: {flips_to_incorrect}", file=sys.stderr)
    print(f"Output written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
