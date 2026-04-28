"""H3a sensitivity analyses (amendment 002).

Runs three analyses against an exp3a results corpus and writes a single
JSON report:

  1. Parent pre-reg quadratic fit on signed levels (existing pipeline).
  2. Arousal-as-magnitude quadratic fit on |level - 4|.
  3. Within-subjects per-item-fixed-effects quadratic fit (only when
     the corpus has sampling_mode == "within_subjects").

Usage:
    uv run python scripts/analysis/h3a_sensitivity.py \\
        --results-dir results/h3a_2026-04-27_n122 \\
        --output results/h3a_2026-04-27_n122/sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis.exp3a import (  # noqa: E402
    analyze_arousal_magnitude,
    analyze_exp3a,
    analyze_within_subjects,
)


def _load_corpus(results_dir: Path) -> list[dict]:
    """Load every JSON under results_dir/data/level_*/<condition>/*.json."""
    corpus = []
    for json_path in sorted(results_dir.rglob("*.json")):
        if json_path.name in {"manifest.yaml", "AGGREGATE_REPORT.md"}:
            continue
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("experiment_type") == "exp3a":
            corpus.append(data)
    return corpus


def _accuracy_by_level(corpus: list[dict]) -> dict[int, list[float]]:
    by_level: dict[int, list[float]] = {}
    for record in corpus:
        body = record.get("body") or {}
        level = body.get("intensity_level")
        binary = body.get("binary_correct")
        if level is None or binary is None:
            continue
        by_level.setdefault(int(level), []).append(float(binary))
    return by_level


def _detect_sampling_mode(corpus: list[dict]) -> str:
    modes = {
        (record.get("body") or {}).get("sampling_mode", "cross_level_disjoint")
        for record in corpus
    }
    if len(modes) == 1:
        return modes.pop()
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="Pilot root produced by `affect-battery run --experiment exp3a`.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write the sensitivity JSON.")
    args = ap.parse_args()

    corpus = _load_corpus(args.results_dir)
    if not corpus:
        print(f"error: no exp3a results found under {args.results_dir}", file=sys.stderr)
        return 2

    sampling_mode = _detect_sampling_mode(corpus)
    accuracy_by_level = _accuracy_by_level(corpus)

    primary = analyze_exp3a(accuracy_by_level)
    arousal = analyze_arousal_magnitude(accuracy_by_level)
    within = None
    if sampling_mode == "within_subjects":
        try:
            within = analyze_within_subjects(corpus)
        except ValueError as e:
            within = {"error": str(e)}

    report = {
        "results_dir": str(args.results_dir),
        "n_records": len(corpus),
        "sampling_mode_detected": sampling_mode,
        "primary_signed_axis": primary,
        "arousal_magnitude": arousal,
        "within_subjects": within,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))

    print(f"Sensitivity report written to {args.output}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"sampling_mode: {sampling_mode}", file=sys.stderr)
    print(f"primary β₂ = {primary['beta_2']:+.4f}  (SE {primary['beta_2_se']:.4f}, p={primary['beta_2_p_one_sided']:.3f})", file=sys.stderr)
    print(f"arousal β₂ = {arousal['beta_2']:+.4f}  (SE {arousal['beta_2_se']:.4f}, p={arousal['beta_2_p_one_sided']:.3f})", file=sys.stderr)
    if within and "beta_2" in within:
        print(f"within   β₂ = {within['beta_2']:+.4f}  (SE {within['beta_2_se']:.4f}, p={within['beta_2_p_one_sided']:.3f}, n_items={within['n_items_used']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
