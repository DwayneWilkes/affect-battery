"""Generate an H3a power report from a variance-estimate JSON.

Reads a JSON file containing per-level variance estimates (typically
written by a variance probe), runs the simulation-based power analysis
across a grid of assumed effect sizes, and writes a power report JSON +
human-readable summary.

The report format matches the harness's `--power-report-path` contract:
the recommended n and the simulation parameters become the input to the
data-collection gate.

Input variance JSON shape:
    {
      "model": "<model_name>",
      "task": "GSM8K + GSM-Hard",
      "n_levels": 7,
      "sigma_per_level": [s1, s2, ..., s7],
      "icc": 0.20,
      "notes": "..."
    }

Usage:
    uv run python scripts/probes/h3a_power_report.py \\
        --variance-json results/probes/h3a_variance_<date>.json \\
        --output results/probes/h3a_power_report_<date>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.power.h3a_simulation import find_min_n, power_at_n  # noqa: E402


# Scenarios to run by default. Each is (label, assumed_beta2). Effect sizes
# correspond to inverted-U curves with peak at level 4, drop from peak to
# edges of: 0.025, 0.05, 0.10, 0.15. beta2 = -drop / (3^2) for symmetric
# 7-level design (max distance from peak = 3).
DEFAULT_SCENARIOS = [
    ("very_small_drop_0.025", -0.0028),
    ("small_drop_0.05",       -0.0056),
    ("moderate_drop_0.10",    -0.0111),
    ("large_drop_0.15",       -0.0167),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variance-json", required=True, type=Path,
                    help="Variance probe output JSON")
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write the power report JSON")
    ap.add_argument("--target-power", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-min", type=int, default=5)
    ap.add_argument("--n-max", type=int, default=300)
    ap.add_argument("--n-simulations", type=int, default=500,
                    help="Monte Carlo iterations per power calc")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.variance_json.exists():
        print(f"variance JSON not found: {args.variance_json}", file=sys.stderr)
        return 2
    var_data = json.loads(args.variance_json.read_text())
    sigma_per_level = var_data.get("sigma_per_level")
    if not isinstance(sigma_per_level, list) or len(sigma_per_level) != 7:
        print("variance JSON must include 'sigma_per_level' as a 7-element list",
              file=sys.stderr)
        return 1
    icc = float(var_data.get("icc", 0.20))

    print(f"Variance estimates from {args.variance_json}:")
    print(f"  model: {var_data.get('model', '<unknown>')}")
    print(f"  task:  {var_data.get('task', '<unknown>')}")
    print(f"  sigma_per_level: {[f'{s:.4f}' for s in sigma_per_level]}")
    print(f"  ICC: {icc}")
    print()

    scenarios_results = []
    print(f"Power scenarios (target_power={args.target_power}, alpha={args.alpha}):")
    print(f"{'scenario':<28s} {'beta2':>10s} {'n_recommended':>15s} {'power@n_max':>12s}")
    for label, beta2 in DEFAULT_SCENARIOS:
        n_rec, trace = find_min_n(
            beta2_assumed=beta2,
            sigma_per_level=sigma_per_level,
            icc=icc,
            target_power=args.target_power,
            n_min=args.n_min,
            n_max=args.n_max,
            n_simulations=args.n_simulations,
            seed=args.seed,
            alpha=args.alpha,
        )
        # The first entry in `trace` is the n_max probe, regardless of
        # whether the search succeeded.
        power_at_max = trace[0].power
        print(f"  {label:<26s} {beta2:>+10.4f} {str(n_rec) if n_rec else 'NA':>15s} {power_at_max:>12.3f}")
        scenarios_results.append({
            "label": label,
            "beta2_assumed": beta2,
            "n_recommended": n_rec,
            "power_at_n_max": power_at_max,
            "search_trace": [
                {"n_per_level": t.n_per_level, "power": t.power, "n_significant": t.n_significant}
                for t in trace
            ],
        })
    print()

    # Cross-recommendation: take max n across scenarios that achieved target,
    # OR report MDE-not-achievable for the smallest effects.
    achievable = [s for s in scenarios_results if s["n_recommended"] is not None]
    if achievable:
        max_n = max(s["n_recommended"] for s in achievable)
        print(f"Recommended n_per_level (max across achievable scenarios): {max_n}")
    else:
        print("No scenario achieved the target power within n_max. Consider:")
        print("  - Relaxing target_power")
        print("  - Increasing n_max")
        print("  - Re-examining the assumed effect sizes against pilot data")

    report = {
        "report_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target_power": args.target_power,
        "alpha": args.alpha,
        "n_simulations_per_calc": args.n_simulations,
        "seed": args.seed,
        "variance_input": {
            "source": str(args.variance_json),
            "model": var_data.get("model"),
            "task": var_data.get("task"),
            "sigma_per_level": sigma_per_level,
            "icc": icc,
        },
        "scenarios": scenarios_results,
        "recommended_n_per_level": (
            max(s["n_recommended"] for s in achievable) if achievable else None
        ),
        "mde_assumption_note": (
            "Recommended n is the maximum across power scenarios that achieved "
            "target_power within n_max. Power for a smaller-than-tested effect "
            "would require an even larger n; the report does not extrapolate "
            "beyond the tested scenarios."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"\nPower report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
