"""Build an H3b calibrated bank YAML from a calibration JSON.

Reads the JSON output of h3b_calibration_robust.py and writes a task-bank
YAML containing every item in the calibration's `calibrated_subset` (no
truncation, no ranking by closeness to 0.5). Simulation under H0 shows
"all qualifiers" gives strictly better contrast precision than truncating
to a fixed n with no bias cost; ranking by `|p̂ - 0.5|` was an arbitrary
limit that cost ~14-50% in CI width with no scientific gain.

Usage:
    direnv exec . uv run --active python scripts/h3b_build_bank_from_calibration.py \\
        --calibration configs/h3b_calibration_2026-05-08.json \\
        --output configs/banks/h3b_calibrated_v2.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--calibration", required=True, type=Path,
                    help="Calibration JSON from h3b_calibration_robust.py")
    ap.add_argument("--min-items", type=int, default=32,
                    help="Minimum acceptable yield from calibration. Default 32 "
                         "is the n at which the simulated bootstrap CI half-width "
                         "on c stays below 0.05 in 100%% of trials (vs 81%% at "
                         "n=17 and 84%% at n=18), per "
                         "results/probes/h3b_precision_report_2026-05-08.json. "
                         "Below this, the prereg's strong 'bounded-small' claim "
                         "(c_ci95_hi < 0.05) is at material risk of failing to "
                         "lock in.")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output bank YAML path")
    ap.add_argument("--bank-id", default="h3b_calibrated_v2",
                    help="bank_id field for the output")
    ap.add_argument("--bank-version", type=int, default=2)
    ap.add_argument("--parent-bank", default="gsm8k_v1")
    args = ap.parse_args()

    calibration = json.loads(args.calibration.read_text())
    calibrated = calibration.get("calibrated_subset", [])
    if len(calibrated) < args.min_items:
        print(
            f"ERROR: calibrated_subset has {len(calibrated)} items, "
            f"below --min-items={args.min_items} floor. Run a larger "
            f"calibration (more --n-candidates) and try again.",
            file=sys.stderr,
        )
        return 1

    # Take ALL qualifiers — no truncation, no ranking-by-closeness-to-0.5.
    # See module docstring for the simulation result that motivates this.
    selected = sorted(calibrated, key=lambda p: abs(p["p_hat"] - 0.5))

    # Read target band from the calibration JSON itself rather than CLI
    # defaults — the rationale string would otherwise lie if a future
    # caller passed inconsistent flags here.
    target_lo = calibration.get("target_lo", 0.40)
    target_hi = calibration.get("target_hi", 0.60)

    # Compute calibration JSON SHA so the bank can reference its
    # provenance pin in the alignment_review.rationale.
    calib_sha = hashlib.sha256(args.calibration.read_bytes()).hexdigest()

    bank = {
        "bank_id": args.bank_id,
        "bank_version": args.bank_version,
        "bank_type": "task",
        "status": "active",
        "task_type_subtype": "math-word-problem",
        "parent_bank": args.parent_bank,
        "alignment_review": {
            "reviewer": (
                f"h3b calibration probe "
                f"(n_candidates={calibration.get('n_candidates', '?')}, "
                f"n_reps={calibration.get('n_reps', '?')})"
            ),
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "verdict": "pass",
            "rationale": (
                f"{len(selected)} GSM-Hard items pre-screened on "
                f"{calibration.get('model', '?')} with no stimulus at "
                f"temperature {calibration.get('temperature', 0.7)}, retained "
                f"for p_hat in [{target_lo:.2f}, {target_hi:.2f}] (all "
                f"qualifiers, no truncation). "
                f"Calibration JSON: {args.calibration}, sha256 {calib_sha}."
            ),
        },
        "items": [
            {
                "id": item["item_id"],
                "question": item["question"],
                "expected": item["expected"],
                "answer_aliases": [],
                "difficulty": "hard",
                "p_hat_calib": round(item["p_hat"], 4),
            }
            for item in selected
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.dump(bank, sort_keys=False, allow_unicode=True))
    print(f"wrote {args.output}", file=sys.stderr)
    print(f"  {len(selected)} items, p̂ range "
          f"[{min(s['p_hat'] for s in selected):.2f}, "
          f"{max(s['p_hat'] for s in selected):.2f}]",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
