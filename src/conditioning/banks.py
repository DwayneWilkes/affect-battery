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


# Paper §3.2.1 cross-domain transfer task types (the only valid task_type
# values for a TransferBank used in Exp 1 cross-domain transfer per
# `cross-domain-transfer-tasks/spec.md`). Validated at use-time, not at
# load-time, so legacy task-difficulty-calibration TransferBanks with
# arithmetic / reasoning task_type still load (they're use-rejected when
# registered for Exp 1).
PAPER_3_2_1_TASK_TYPES = frozenset({
    "logic-puzzle",
    "factual-qa",
    "reading-comprehension",
})


@dataclass
class ArithmeticBank:
    bank_id: str
    bank_version: int
    difficulty_profile: dict[str, Any]
    items: list[ArithmeticItem]
    stimulus_bank_hash: str
    # bank_type discriminator + status (design.md D9). Both have safe
    # defaults so existing arithmetic_easy_v1 / arithmetic_hard_v1 banks
    # without these fields continue to load.
    bank_type: str = "arithmetic"
    status: str = "active"

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
            bank_type=str(raw.get("bank_type", "arithmetic")),
            status=str(raw.get("status", "active")),
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


@dataclass(frozen=True)
class AlignmentReview:
    """Reviewer-recorded verdict on whether a candidate transfer bank
    conforms to paper §3.2.1's cross-domain definition. Required before
    promoting a candidate bank to active status (cross-domain-transfer-tasks
    spec "Alignment review for re-scoped transfer banks")."""
    reviewer: str
    date: str
    verdict: str  # "pass" | "fail"


class BankAlignmentError(ValueError):
    """Raised when a candidate bank is upgraded without a passed
    alignment review."""


@dataclass
class TransferBank:
    bank_id: str
    bank_version: int
    difficulty_profile: dict[str, Any]
    items: list[TransferItem]
    stimulus_bank_hash: str
    # bank_type discriminator + status (design.md D9 + cross-domain-transfer-tasks
    # spec). status="candidate" indicates an unreviewed bank that must not
    # enter primary analysis until alignment-review passes.
    bank_type: str = "transfer"
    status: str = "active"
    alignment_review: AlignmentReview | None = None

    def upgrade_to_active(self) -> None:
        """Promote a candidate bank to active status. Raises BankAlignmentError
        if alignment_review is missing or its verdict is not 'pass'."""
        if (
            self.alignment_review is None
            or self.alignment_review.verdict != "pass"
        ):
            raise BankAlignmentError(
                f"TransferBank {self.bank_id!r} cannot be upgraded to active: "
                f"alignment review missing or not passed. "
                f"(Current alignment_review: {self.alignment_review!r}.) "
                f"See cross-domain-transfer-tasks spec 'Alignment review for "
                f"re-scoped transfer banks' for the required reviewer pass."
            )
        self.status = "active"

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

        ar_raw = raw.get("alignment_review")
        alignment_review = (
            AlignmentReview(
                reviewer=str(ar_raw["reviewer"]),
                date=str(ar_raw["date"]),
                verdict=str(ar_raw["verdict"]),
            )
            if ar_raw is not None
            else None
        )

        return cls(
            bank_id=str(raw["bank_id"]),
            bank_version=int(raw["bank_version"]),
            difficulty_profile=dict(raw.get("difficulty_profile") or {}),
            items=items,
            stimulus_bank_hash=_canonical_items_hash(items_raw),
            bank_type=str(raw.get("bank_type", "transfer")),
            status=str(raw.get("status", "active")),
            alignment_review=alignment_review,
        )

    def validate_for_exp1_transfer(self) -> None:
        """Reject task_types that don't conform to paper §3.2.1's cross-domain
        transfer definition. Per cross-domain-transfer-tasks/spec.md scenario
        "Arithmetic-to-arithmetic is rejected": valid task types are
        logic-puzzle, factual-qa, reading-comprehension only.

        Raises:
            ValueError: if any item's task_type is not in PAPER_3_2_1_TASK_TYPES.
        """
        bad_types = sorted({
            item.task_type
            for item in self.items
            if item.task_type not in PAPER_3_2_1_TASK_TYPES
        })
        if bad_types:
            raise ValueError(
                f"TransferBank {self.bank_id!r} contains task_type(s) "
                f"{bad_types} that violate paper §3.2.1 cross-domain "
                f"transfer definition. Valid types: "
                f"{sorted(PAPER_3_2_1_TASK_TYPES)}."
            )


def is_primary_analysis_eligible(bank: "ArithmeticBank | TransferBank") -> bool:
    """Return True if the bank may participate in primary-experiment
    analysis (per cross-domain-transfer-tasks spec "Alignment review for
    re-scoped transfer banks"):

    - Active banks: eligible.
    - Candidate banks: eligible only if alignment_review.verdict == "pass".
    - Archived banks: ineligible (use --allow-archived-bank for replication).

    The Phase 3+ analysis aggregators MUST filter result corpora by this
    predicate before computing primary effect sizes. Runs from ineligible
    banks are tagged candidate_bank=True / archived_bank=True in their
    result config and excluded from primary aggregation.
    """
    if bank.status == "active":
        return True
    if bank.status == "archived":
        return False
    if bank.status == "candidate":
        ar = getattr(bank, "alignment_review", None)
        return ar is not None and ar.verdict == "pass"
    return False
