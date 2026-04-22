#!/usr/bin/env python3
"""One-shot migration: export the existing 6-item factual-QA pool from
`src/conditioning/tasks.py` to `configs/banks/transfer_easy_v1.yaml`.

Preserves existing items so cached legacy results remain valid. Items are
tagged per the Stimulus-bank-schema Requirement (task_type: recall,
difficulty_class: easy). `stimulus_bank_hash` is computed at load time
by the TransferBank loader — the YAML itself only carries the enumerated
items.
"""

from collections import OrderedDict
from pathlib import Path

import yaml

from src.conditioning.tasks import _FACTUAL_QA


BANK_ID = "transfer_easy_v1"
BANK_VERSION = 1


def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, _ordered_dict_representer)


def build_items() -> list[OrderedDict]:
    """Turn each TransferQuestion into the bank's per-item schema."""
    items: list[OrderedDict] = []
    for i, q in enumerate(_FACTUAL_QA, start=1):
        items.append(
            OrderedDict(
                [
                    ("id", f"{BANK_ID}_{i:04d}"),
                    ("question", q.question),
                    ("expected_answer", q.expected_answer),
                    ("task_type", "recall"),
                    ("difficulty_class", "easy"),
                    # Original difficulty tag from the source; preserved so
                    # the review for the hard banks can compare against it.
                    ("source_difficulty", q.difficulty),
                ]
            )
        )
    return items


def build_bank(items: list[OrderedDict]) -> OrderedDict:
    return OrderedDict(
        [
            ("bank_id", BANK_ID),
            ("bank_version", BANK_VERSION),
            (
                "difficulty_profile",
                OrderedDict(
                    [
                        ("expected_accuracy_class", "ceiling"),
                        ("task_type", "recall"),
                        ("difficulty_class", "easy"),
                        (
                            "notes",
                            "Single-fact factual recall items formalized from "
                            "src/conditioning/tasks.py _FACTUAL_QA. This bank "
                            "is the same-task-type / easy cell of the 2x2 "
                            "transfer design (see design.md D5). On Qwen2.5-7B "
                            "family these items land at ceiling; the cell is "
                            "retained for backward compatibility with legacy "
                            "cached results, not because it yields measurable "
                            "transfer signal.",
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
