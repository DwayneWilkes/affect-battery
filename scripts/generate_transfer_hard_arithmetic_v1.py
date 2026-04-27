#!/usr/bin/env python3
"""Generate `configs/banks/transfer_hard_arithmetic_v1.yaml`: 30 harder
arithmetic items for the same-task-type / hard cell of the 2x2 transfer
design.

These items share the operator taxonomy of the conditioning bank but are
disjoint from it (different seed, different item IDs) so a model's
conditioning-phase memorization can't bleed into the transfer phase.

Disjointness note: both banks draw operands uniformly at random from
large integer ranges; exact operand-tuple collisions are possible but
negligibly probable at ~300 conditioning items + 30 transfer items. The
item-id prefix differs (`arith_hard_v1_*` vs `transfer_hard_arith_v1_*`)
so analysis code can always distinguish bank provenance.

Spec: affect-battery-task-difficulty-calibration::task-difficulty-calibration::
"Calibration pilot protocol" (2x2 transfer design, same-task-type cell).
Task 2.2.
"""

from collections import OrderedDict
from pathlib import Path
import random

import yaml

from src.calibration.generator import (
    DEFAULT_OPERATOR_SPECS,
    OPERATOR_MIX,
    generate_items_for_operator,
)


BANK_ID = "transfer_hard_arithmetic_v1"
BANK_VERSION = 1
TOTAL_ITEMS = 30
RNG_SEED = 20260422  # disjoint from conditioning-bank seed (20260421)

# Operator mix proportional to the conditioning bank but scaled down.
# Preserves relative difficulty profile so transfer-same-type results are
# comparable to conditioning-phase results.


def _op_symbol(op: str) -> str:
    return {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]


def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _ordered_dict_representer)


def _item_to_transfer_schema(item: dict, idx: int) -> OrderedDict:
    """Reshape an arithmetic generator item into the TransferBank schema:
    question + expected_answer (both strings), task_type, difficulty_class."""
    a, b = item["operands"]
    symbol = _op_symbol(item["operator"])
    return OrderedDict(
        [
            ("id", f"{BANK_ID}_{idx:04d}"),
            ("question", f"What is {a} {symbol} {b}?"),
            ("expected_answer", str(item["answer"])),
            ("task_type", "arithmetic"),
            ("difficulty_class", "hard"),
            ("operator", item["operator"]),
            ("n_carries", item["n_carries"]),
        ]
    )


def build_items() -> list[OrderedDict]:
    rng = random.Random(RNG_SEED)
    target_counts = {op: round(TOTAL_ITEMS * frac) for op, frac in OPERATOR_MIX.items()}
    delta = TOTAL_ITEMS - sum(target_counts.values())
    target_counts["add"] += delta

    raw_items: list[dict] = []
    for op in OPERATOR_MIX:
        bucket = generate_items_for_operator(
            operator=op,
            spec=DEFAULT_OPERATOR_SPECS[op],
            count=target_counts[op],
            rng=rng,
            id_prefix="raw",
        )
        raw_items.extend(bucket)

    return [_item_to_transfer_schema(item, idx=i + 1) for i, item in enumerate(raw_items)]


def build_bank(items: list[OrderedDict]) -> OrderedDict:
    return OrderedDict(
        [
            ("bank_id", BANK_ID),
            ("bank_version", BANK_VERSION),
            (
                "difficulty_profile",
                OrderedDict(
                    [
                        ("expected_accuracy_class", "mid"),
                        ("task_type", "arithmetic"),
                        ("difficulty_class", "hard"),
                        ("operator_mix", dict(OPERATOR_MIX)),
                        (
                            "notes",
                            "Same-task-type / hard cell of the 2x2 transfer "
                            "design (see design.md D5). Items share the "
                            "conditioning bank's operator taxonomy but are "
                            "disjoint by seed (conditioning=20260421, "
                            "transfer=20260422) so memorization can't bleed "
                            "from conditioning into transfer. Question "
                            "rendering matches the default_prompt_fn in "
                            "src/calibration/pipeline.py."
                        ),
                    ]
                ),
            ),
            ("items", items),
        ]
    )


def main() -> None:
    items = build_items()
    bank = build_bank(items)
    repo_root = Path(__file__).resolve().parents[1]
    out = repo_root / "configs" / "banks" / f"{BANK_ID}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        yaml.dump(bank, f, sort_keys=False, default_flow_style=False, width=120)
    print(f"Wrote {out} ({len(items)} items)")


if __name__ == "__main__":
    main()
