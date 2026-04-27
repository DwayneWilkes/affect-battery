#!/usr/bin/env python3
"""One-shot migration: export the legacy generator output as YAML bank.

The legacy generator at `src/conditioning/tasks.py::get_arithmetic_problems`
produces 2-digit (operand range [10, 99]) items with operators + / - / *.
This script runs the generator with a frozen `(seed, n)` and serializes the
output to `configs/banks/arithmetic_easy_v1.yaml` per the design.md D1 schema.

Freezing `seed=0, n=300` gives a deterministic, inspectable item set. Any
cached legacy run that was itself parameterized by `(seed=0, n=300)` will
produce identical operands and answers, so cache identity is preserved when
callers migrate to the YAML bank.

Re-running this script yields byte-identical output.
"""

from __future__ import annotations

import random
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

# Allow running as `python scripts/migrate_arithmetic_easy_v1.py`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.generator import (  # noqa: E402
    count_add_carries,
    count_mul_long_carries,
    count_sub_borrows,
    digit_count,
)


BANK_ID = "arithmetic_easy_v1"
BANK_VERSION = 1
SEED = 0
N_ITEMS = 300
OP_RANGE = (10, 99)

# Map legacy `+ - *` to the canonical operator names.
_OP_NAME = {"+": "add", "-": "sub", "*": "mul"}


def _n_carries(op: str, a: int, b: int) -> int:
    if op == "add":
        return count_add_carries(a, b)
    if op == "sub":
        # Legacy generator does not enforce a >= b; normalize to the larger-minus-
        # smaller semantic that the carry counter expects. Negative differences
        # don't change the number of borrow steps.
        hi, lo = (a, b) if a >= b else (b, a)
        return count_sub_borrows(hi, lo)
    if op == "mul":
        return count_mul_long_carries(a, b)
    raise ValueError(f"Unsupported operator for easy_v1 migration: {op}")


def _generate_items() -> list[OrderedDict]:
    """Reproduce the legacy generator's draw sequence exactly.

    Mirrors `src.conditioning.tasks.get_arithmetic_problems(n=N_ITEMS, seed=SEED)`
    with the same RNG call order so operands and answers match legacy output.
    """
    rng = random.Random(SEED)
    items: list[OrderedDict] = []
    for idx in range(N_ITEMS):
        a = rng.randint(*OP_RANGE)
        b = rng.randint(*OP_RANGE)
        op_sym = rng.choice(["+", "-", "*"])
        op = _OP_NAME[op_sym]
        if op == "add":
            answer = a + b
        elif op == "sub":
            answer = a - b
        else:
            answer = a * b

        item = OrderedDict()
        item["id"] = f"arith_easy_v1_{idx + 1:04d}"
        item["operands"] = [a, b]
        item["operator"] = op
        item["answer"] = answer
        item["digit_count"] = digit_count(a, b)
        item["n_carries"] = _n_carries(op, a, b)
        items.append(item)
    return items


def _build_bank() -> OrderedDict:
    bank = OrderedDict()
    bank["bank_id"] = BANK_ID
    bank["bank_version"] = BANK_VERSION
    profile = OrderedDict()
    profile["expected_accuracy_class"] = "ceiling"
    profile["per_operand_digit_range"] = [2, 2]
    profile["operand_range"] = list(OP_RANGE)
    profile["operator_mix"] = OrderedDict([
        ("add", 1 / 3),
        ("sub", 1 / 3),
        ("mul", 1 / 3),
    ])
    profile["calibration_source"] = "legacy generator export (seed=0, n=300)"
    profile["notes"] = (
        "Migrated from src.conditioning.tasks.get_arithmetic_problems "
        "with seed=0, n=300. Operand range is [10, 99]; operators are a "
        "uniform choice over + / - / *. digit_count is per-operand; "
        "n_carries is the standard pen-and-paper carry/borrow count."
    )
    bank["difficulty_profile"] = profile
    bank["items"] = _generate_items()
    return bank


def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping(
        "tag:yaml.org,2002:map", list(data.items()), flow_style=False
    )


def main() -> None:
    yaml.add_representer(OrderedDict, _ordered_dict_representer, Dumper=yaml.SafeDumper)
    bank = _build_bank()
    out = ROOT / "configs" / "banks" / f"{BANK_ID}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(bank, sort_keys=False, default_flow_style=False))
    print(f"wrote {out} with {len(bank['items'])} items")


if __name__ == "__main__":
    main()
