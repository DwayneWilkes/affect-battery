"""Tests for the stimulus-bank YAML loader (ArithmeticBank, TransferBank)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.conditioning.banks import (
    ArithmeticBank,
    ArithmeticItem,
    BankNotFoundError,
    TransferBank,
    TransferItem,
)


BANKS_DIR = Path(__file__).parent.parent / "configs" / "banks"


# --------------------------------------------------------------------------- #
# ArithmeticBank.load                                                         #
# --------------------------------------------------------------------------- #


class TestArithmeticBankLoad:
    def test_loads_valid_yaml(self) -> None:
        bank = ArithmeticBank.load("arithmetic_easy_v1")
        assert bank.bank_id == "arithmetic_easy_v1"
        assert isinstance(bank.bank_version, int)
        assert bank.bank_version >= 1
        assert isinstance(bank.difficulty_profile, dict)
        assert isinstance(bank.items, list)
        assert len(bank.items) > 0

    def test_loads_hard_bank(self) -> None:
        bank = ArithmeticBank.load("arithmetic_hard_v1")
        assert bank.bank_id == "arithmetic_hard_v1"
        assert len(bank.items) > 0

    def test_items_expose_property_tags(self) -> None:
        bank = ArithmeticBank.load("arithmetic_hard_v1")
        for item in bank.items:
            assert isinstance(item, ArithmeticItem)
            assert isinstance(item.id, str) and item.id
            assert isinstance(item.operands, list) and len(item.operands) >= 2
            assert item.operator in {"add", "sub", "mul", "div"}
            assert isinstance(item.answer, (int, float))
            assert isinstance(item.digit_count, list)
            assert all(isinstance(d, int) for d in item.digit_count)
            assert isinstance(item.n_carries, int)

    def test_raises_on_unknown_bank_id(self) -> None:
        with pytest.raises(BankNotFoundError) as exc:
            ArithmeticBank.load("not_a_real_bank_xyz")
        assert "not_a_real_bank_xyz" in str(exc.value)

    def test_stimulus_bank_hash_is_sha256_hex(self) -> None:
        bank = ArithmeticBank.load("arithmetic_easy_v1")
        assert isinstance(bank.stimulus_bank_hash, str)
        assert len(bank.stimulus_bank_hash) == 64
        int(bank.stimulus_bank_hash, 16)  # hex-parseable

    def test_stimulus_bank_hash_is_stable_across_loads(self) -> None:
        b1 = ArithmeticBank.load("arithmetic_easy_v1")
        b2 = ArithmeticBank.load("arithmetic_easy_v1")
        assert b1.stimulus_bank_hash == b2.stimulus_bank_hash

    def test_stimulus_bank_hash_matches_canonical_spec(self) -> None:
        """Hash MUST be SHA-256 over canonicalized items list: sorted by
        item id, JSON-serialized with stable key order."""
        bank = ArithmeticBank.load("arithmetic_easy_v1")
        # Recompute directly from the YAML contents to verify canonical form.
        raw = yaml.safe_load((BANKS_DIR / "arithmetic_easy_v1.yaml").read_text())
        items = sorted(raw["items"], key=lambda it: it["id"])
        blob = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
        expected = hashlib.sha256(blob).hexdigest()
        assert bank.stimulus_bank_hash == expected

    def test_mid_curation_edit_invalidates_cache(self, tmp_path: Path) -> None:
        """Same bank_id, different item content => different stimulus_bank_hash."""
        base_path = BANKS_DIR / "arithmetic_easy_v1.yaml"
        raw = yaml.safe_load(base_path.read_text())

        edited = dict(raw)
        edited["items"] = list(raw["items"]) + [{
            "id": "arith_easy_v1_inserted_zzz",
            "operands": [11, 22],
            "operator": "add",
            "answer": 33,
            "digit_count": [2, 2],
            "n_carries": 0,
        }]

        custom_dir = tmp_path / "banks"
        custom_dir.mkdir()
        (custom_dir / "arithmetic_easy_v1.yaml").write_text(yaml.safe_dump(edited))

        original = ArithmeticBank.load("arithmetic_easy_v1")
        modified = ArithmeticBank.load("arithmetic_easy_v1", banks_dir=custom_dir)

        assert original.bank_id == modified.bank_id
        assert original.stimulus_bank_hash != modified.stimulus_bank_hash

    def test_custom_banks_dir(self, tmp_path: Path) -> None:
        raw = {
            "bank_id": "custom_x1",
            "bank_version": 1,
            "difficulty_profile": {"expected_accuracy_class": "mid"},
            "items": [
                {
                    "id": "custom_x1_0001",
                    "operands": [1, 2],
                    "operator": "add",
                    "answer": 3,
                    "digit_count": [1, 1],
                    "n_carries": 0,
                },
            ],
        }
        (tmp_path / "custom_x1.yaml").write_text(yaml.safe_dump(raw))
        bank = ArithmeticBank.load("custom_x1", banks_dir=tmp_path)
        assert bank.bank_id == "custom_x1"
        assert bank.items[0].id == "custom_x1_0001"


# --------------------------------------------------------------------------- #
# TransferBank.load                                                           #
# --------------------------------------------------------------------------- #


class TestTransferBankLoad:
    def test_loads_yaml_with_task_type_and_difficulty_class(
        self, tmp_path: Path
    ) -> None:
        raw = {
            "bank_id": "transfer_easy_v1",
            "bank_version": 1,
            "difficulty_profile": {"difficulty_class": "easy"},
            "items": [
                {
                    "id": "transfer_easy_v1_0001",
                    "prompt": "What is the capital of France?",
                    "expected_answer": "Paris",
                    "task_type": "recall",
                    "difficulty_class": "easy",
                },
                {
                    "id": "transfer_easy_v1_0002",
                    "prompt": "What is 1 + 1?",
                    "expected_answer": "2",
                    "task_type": "arithmetic",
                    "difficulty_class": "easy",
                },
            ],
        }
        (tmp_path / "transfer_easy_v1.yaml").write_text(yaml.safe_dump(raw))

        bank = TransferBank.load("transfer_easy_v1", banks_dir=tmp_path)

        assert bank.bank_id == "transfer_easy_v1"
        assert bank.bank_version == 1
        assert len(bank.items) == 2
        for item in bank.items:
            assert isinstance(item, TransferItem)
            assert item.task_type in {"arithmetic", "reasoning", "recall"}
            assert item.difficulty_class in {"easy", "hard"}
            assert isinstance(item.prompt, str) and item.prompt
            assert isinstance(item.expected_answer, str)

    def test_raises_on_unknown_bank_id(self) -> None:
        with pytest.raises(BankNotFoundError):
            TransferBank.load("not_a_real_transfer_bank_xyz")

    def test_stimulus_bank_hash_is_sha256_hex(self, tmp_path: Path) -> None:
        raw = {
            "bank_id": "transfer_tmp_v1",
            "bank_version": 1,
            "difficulty_profile": {},
            "items": [
                {
                    "id": "transfer_tmp_v1_0001",
                    "prompt": "Q?",
                    "expected_answer": "A",
                    "task_type": "recall",
                    "difficulty_class": "easy",
                },
            ],
        }
        (tmp_path / "transfer_tmp_v1.yaml").write_text(yaml.safe_dump(raw))
        bank = TransferBank.load("transfer_tmp_v1", banks_dir=tmp_path)
        assert isinstance(bank.stimulus_bank_hash, str)
        assert len(bank.stimulus_bank_hash) == 64
        int(bank.stimulus_bank_hash, 16)

    def test_mid_curation_edit_invalidates_cache(self, tmp_path: Path) -> None:
        base = {
            "bank_id": "transfer_edit_v1",
            "bank_version": 1,
            "difficulty_profile": {},
            "items": [
                {
                    "id": "transfer_edit_v1_0001",
                    "prompt": "Q1?",
                    "expected_answer": "A1",
                    "task_type": "recall",
                    "difficulty_class": "easy",
                },
            ],
        }
        (tmp_path / "transfer_edit_v1.yaml").write_text(yaml.safe_dump(base))
        original = TransferBank.load("transfer_edit_v1", banks_dir=tmp_path)

        edited = dict(base)
        edited["items"] = list(base["items"]) + [{
            "id": "transfer_edit_v1_0002",
            "prompt": "Q2?",
            "expected_answer": "A2",
            "task_type": "recall",
            "difficulty_class": "easy",
        }]
        (tmp_path / "transfer_edit_v1.yaml").write_text(yaml.safe_dump(edited))
        modified = TransferBank.load("transfer_edit_v1", banks_dir=tmp_path)

        assert original.bank_id == modified.bank_id
        assert original.stimulus_bank_hash != modified.stimulus_bank_hash
