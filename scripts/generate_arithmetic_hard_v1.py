#!/usr/bin/env python3
"""Entry point that generates `configs/banks/arithmetic_hard_v1.yaml`
using the hand-tuned module defaults in `src/calibration/generator.py`.

For auto-calibrated bank generation, see
`scripts/auto_calibrate_arithmetic.py`.
"""

from pathlib import Path

from src.calibration.generator import (
    BANK_ID,
    build_bank_yaml,
    generate_items,
    summarize,
    write_bank,
)


def main() -> None:
    items = generate_items()
    summarize(items)
    bank = build_bank_yaml(items)
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "configs" / "banks" / f"{BANK_ID}.yaml"
    write_bank(bank, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
