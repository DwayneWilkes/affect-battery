"""Run the H3b precision simulation and write a report.

Drives `src.power.h3b_simulation.find_min_n_for_precision` against the
calibrated p̂ pool from the most recent calibration JSON, using the
prereg's n_reps_per_cell=20 budget. Reports the smallest n_items that
meets each of the prereg's interpretive thresholds:

- CI half-width < 0.05  (strong claim: "bounded small")
- CI half-width < 0.10  (informative-vs-uninformative threshold)

Usage:
    uv run --active python scripts/probes/h3b_precision_report.py \\
        --calibration configs/h3b_calibration_2026-05-08.json \\
        --output results/probes/h3b_precision_report_<YYYY-MM-DD>.json \\
        [--n-simulations 200] [--n-bootstrap 2000] [--n-max 80]

The report serializes recommended n at both thresholds plus the search
trace (per-n half-width medians + reliability percentages) so a reviewer
can audit the curve. Random seed is fixed for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.power.h3b_simulation import find_min_n_for_precision


def _result_to_dict(r) -> dict:
    return {
        "n_items": r.n_items,
        "n_reps_per_cell": r.n_reps_per_cell,
        "c_assumed": r.c_assumed,
        "median_ci_half_width": r.median_ci_half_width,
        "mean_ci_half_width": r.mean_ci_half_width,
        "pct_below_0_05": r.pct_below_0_05,
        "pct_below_0_10": r.pct_below_0_10,
        "median_c_estimate": r.median_c_estimate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True, type=Path,
                    help="Calibration JSON with calibrated_subset[].p_hat")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write the JSON report")
    ap.add_argument("--n-simulations", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=80)
    ap.add_argument("--n-reps-per-cell", type=int, default=20,
                    help="Per-cell rep budget. Match the prereg.")
    ap.add_argument("--target-reliability-strong", type=float, default=80.0)
    ap.add_argument("--target-reliability-informative", type=float, default=95.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text())
    p_hats = [it["p_hat"] for it in cal.get("calibrated_subset", [])]
    if not p_hats:
        raise SystemExit(
            f"No calibrated_subset[].p_hat in {args.calibration}. "
            "Run a calibration first."
        )
    print(
        f"Loaded {len(p_hats)} calibrated p̂s from {args.calibration} "
        f"(mean={sum(p_hats)/len(p_hats):.3f})",
        file=sys.stderr,
    )

    print(
        f"Searching n in [{args.n_min}, {args.n_max}] for CI HW < 0.05 "
        f"at ≥{args.target_reliability_strong}% reliability...",
        file=sys.stderr,
    )
    n_strong, trace_strong = find_min_n_for_precision(
        p_hat_per_item=p_hats,
        target_ci_half_width=0.05,
        target_reliability=args.target_reliability_strong,
        n_min=args.n_min, n_max=args.n_max,
        n_reps_per_cell=args.n_reps_per_cell,
        n_simulations=args.n_simulations,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(f"  → {n_strong}", file=sys.stderr)

    print(
        f"Searching n in [{args.n_min}, {args.n_max}] for CI HW < 0.10 "
        f"at ≥{args.target_reliability_informative}% reliability...",
        file=sys.stderr,
    )
    n_informative, trace_informative = find_min_n_for_precision(
        p_hat_per_item=p_hats,
        target_ci_half_width=0.10,
        target_reliability=args.target_reliability_informative,
        n_min=args.n_min, n_max=args.n_max,
        n_reps_per_cell=args.n_reps_per_cell,
        n_simulations=args.n_simulations,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(f"  → {n_informative}", file=sys.stderr)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_source": str(args.calibration),
        "n_calibrated_in_pool": len(p_hats),
        "p_hat_pool_mean": sum(p_hats) / len(p_hats),
        "p_hat_pool_min": min(p_hats),
        "p_hat_pool_max": max(p_hats),
        "n_reps_per_cell": args.n_reps_per_cell,
        "n_simulations": args.n_simulations,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "thresholds": {
            "strong_claim": {
                "ci_half_width_max": 0.05,
                "target_reliability_pct": args.target_reliability_strong,
                "recommended_n_items": n_strong,
                "interpretation": (
                    "Smallest n_items at which the bootstrap CI half-width on c "
                    "stays below 0.05 in at least the target % of simulations. "
                    "This is the n needed to reliably reach the prereg's "
                    "'bounded-small' inferential claim (c_ci95_hi < 0.05)."
                ),
                "trace": [_result_to_dict(r) for r in trace_strong],
            },
            "informative_claim": {
                "ci_half_width_max": 0.10,
                "target_reliability_pct": args.target_reliability_informative,
                "recommended_n_items": n_informative,
                "interpretation": (
                    "Smallest n_items at which CI half-width stays below 0.10 in "
                    "≥ target %. Below this, the prereg's 'uninformative' "
                    "inferential claim (c_ci95_hi >= 0.10) becomes likely."
                ),
                "trace": [_result_to_dict(r) for r in trace_informative],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nReport written to {args.output}", file=sys.stderr)
    print(f"  strong-claim min n   : {n_strong}", file=sys.stderr)
    print(f"  informative min n    : {n_informative}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
