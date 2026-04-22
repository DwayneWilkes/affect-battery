"""Stimulus-bank YAML loaders (arithmetic + transfer).

Banks are versioned artifacts at `configs/banks/<bank_id>.yaml`. Each bank
enumerates its items with structured property tags per design.md D1.

The loader computes `stimulus_bank_hash` as SHA-256 over the canonicalized
items list (sorted by item id, JSON-serialized with stable key order and
no whitespace). Cache identity keys on this hash, not `bank_id` alone, so
mid-curation edits invalidate caches.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BANKS_DIR = Path(__file__).resolve().parents[2] / "configs" / "banks"


class BankNotFoundError(FileNotFoundError):
    """Raised when a bank YAML cannot be located for the given bank_id."""


def _canonical_items_hash(items: list[dict[str, Any]]) -> str:
    """SHA-256 over items sorted by id and JSON-serialized with stable keys."""
    sorted_items = sorted(items, key=lambda it: it["id"])
    blob = json.dumps(sorted_items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _resolve_bank_path(bank_id: str, banks_dir: Path | None) -> Path:
    base = Path(banks_dir) if banks_dir is not None else DEFAULT_BANKS_DIR
    path = base / f"{bank_id}.yaml"
    if not path.is_file():
        raise BankNotFoundError(
            f"Bank {bank_id!r} not found at {path}. "
            f"Check configs/banks/ for available bank ids."
        )
    return path


# --------------------------------------------------------------------------- #
# Arithmetic                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArithmeticItem:
    id: str
    operands: list[int]
    operator: str  # "add" | "sub" | "mul" | "div"
    answer: int | float
    digit_count: list[int]
    n_carries: int


@dataclass
class ArithmeticBank:
    bank_id: str
    bank_version: int
    difficulty_profile: dict[str, Any]
    items: list[ArithmeticItem]
    stimulus_bank_hash: str

    @classmethod
    def load(
        cls, bank_id: str, *, banks_dir: Path | None = None
    ) -> "ArithmeticBank":
        path = _resolve_bank_path(bank_id, banks_dir)
        raw = yaml.safe_load(path.read_text())

        items_raw: list[dict[str, Any]] = raw["items"]
        items = [
            ArithmeticItem(
                id=str(it["id"]),
                operands=list(it["operands"]),
                operator=str(it["operator"]),
                answer=it["answer"],
                digit_count=list(it["digit_count"]),
                n_carries=int(it["n_carries"]),
            )
            for it in items_raw
        ]

        return cls(
            bank_id=str(raw["bank_id"]),
            bank_version=int(raw["bank_version"]),
            difficulty_profile=dict(raw.get("difficulty_profile") or {}),
            items=items,
            stimulus_bank_hash=_canonical_items_hash(items_raw),
        )


# --------------------------------------------------------------------------- #
# Transfer                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TransferItem:
    id: str
    prompt: str
    expected_answer: str
    task_type: str  # "arithmetic" | "reasoning" | "recall"
    difficulty_class: str  # "easy" | "hard"


@dataclass
class TransferBank:
    bank_id: str
    bank_version: int
    difficulty_profile: dict[str, Any]
    items: list[TransferItem]
    stimulus_bank_hash: str

    @classmethod
    def load(
        cls, bank_id: str, *, banks_dir: Path | None = None
    ) -> "TransferBank":
        path = _resolve_bank_path(bank_id, banks_dir)
        raw = yaml.safe_load(path.read_text())

        items_raw: list[dict[str, Any]] = raw["items"]
        items = [
            TransferItem(
                id=str(it["id"]),
                prompt=str(it["prompt"]),
                expected_answer=str(it["expected_answer"]),
                task_type=str(it["task_type"]),
                difficulty_class=str(it["difficulty_class"]),
            )
            for it in items_raw
        ]

        return cls(
            bank_id=str(raw["bank_id"]),
            bank_version=int(raw["bank_version"]),
            difficulty_profile=dict(raw.get("difficulty_profile") or {}),
            items=items,
            stimulus_bank_hash=_canonical_items_hash(items_raw),
        )
