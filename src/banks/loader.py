"""Bank YAML loader for intensity-axis experiments.

Returns each item as a dict with id, question, expected, and an optional
difficulty tag. Both the variance probe and the production run_exp3a
runner consume this shape; one loader keeps them aligned on field names
and validation.
"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_bank_items(bank_path: str | Path) -> list[dict]:
    """Load items from a bank YAML. Each entry has id, question, expected.

    Raises ValueError if the bank has no `items` array.
    """
    bank_path = Path(bank_path)
    data = yaml.safe_load(bank_path.read_text())
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{bank_path}: bank has no `items` array")
    out = []
    for item in items:
        out.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "difficulty": item.get("difficulty"),
        })
    return out
