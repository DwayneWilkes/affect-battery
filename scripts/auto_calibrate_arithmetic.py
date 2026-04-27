#!/usr/bin/env python3
"""Auto-calibrate `arithmetic_hard_v1` against a running vLLM endpoint.

Binary-searches per-operator `digit_level` until observed model accuracy
lands in [target_min, target_max], then generates a calibrated bank YAML
at the chosen per-operator difficulty settings.

Typical invocation after pod is up with vLLM serving Qwen2.5-7B:

    python scripts/auto_calibrate_arithmetic.py \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen2.5-7B \\
        --output configs/banks/arithmetic_hard_v1.yaml \\
        --target-min 0.60 --target-max 0.85 \\
        --n-per-probe 20

Tests (tests/test_auto_calibrate_script.py) exercise the pipeline
end-to-end with a mock client, no pod required.
"""

import argparse
from pathlib import Path

from src.calibration.auto_calibrator import CalibratorConfig, SweetSpotResult
from src.calibration.pipeline import run_calibration
from src.models import VLLMCompletionClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="vLLM base URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--model", required=True, help="Model name served by vLLM")
    p.add_argument("--output", required=True, type=Path, help="Output bank YAML path")
    p.add_argument("--target-min", type=float, default=0.60)
    p.add_argument("--target-max", type=float, default=0.85)
    p.add_argument("--digit-lo", type=int, default=2)
    p.add_argument("--digit-hi", type=int, default=8)
    p.add_argument("--max-iter", type=int, default=6)
    p.add_argument("--n-per-probe", type=int, default=20)
    p.add_argument("--total-items", type=int, default=300)
    p.add_argument("--bank-id", default="arithmetic_hard_v1")
    p.add_argument("--bank-version", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    client = VLLMCompletionClient(base_url=args.base_url, model=args.model)
    config = CalibratorConfig(
        target_min=args.target_min,
        target_max=args.target_max,
        digit_range=(args.digit_lo, args.digit_hi),
        max_iter=args.max_iter,
    )
    print(
        f"Calibrating {args.model} @ {args.base_url}\n"
        f"  target accuracy window: [{args.target_min}, {args.target_max}]\n"
        f"  digit_level search range: [{args.digit_lo}, {args.digit_hi}]\n"
        f"  n per probe: {args.n_per_probe}"
    )
    results = run_calibration(
        client=client,
        calibrator_config=config,
        output_path=args.output,
        n_items_per_probe=args.n_per_probe,
        total_bank_items=args.total_items,
        bank_id=args.bank_id,
        bank_version=args.bank_version,
    )
    print("\nCalibration results per operator:")
    for op, result in results.items():
        if isinstance(result, SweetSpotResult):
            print(
                f"  {op:>3}: digit_level={result.digit_level}  "
                f"accuracy={result.accuracy:.3f}  "
                f"probes={len(result.probe_history)}"
            )
        else:
            easier = result.easier_side
            harder = result.harder_side
            print(
                f"  {op:>3}: BRACKET (no integer sweet spot in window)  "
                f"easier_side={_fmt(easier)}  harder_side={_fmt(harder)}"
            )
    print(f"\nWrote calibrated bank to {args.output}")


def _fmt(side) -> str:
    if side is None:
        return "none"
    return f"d={side.digit_level}@{side.accuracy:.3f}"


if __name__ == "__main__":
    main()
