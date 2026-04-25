"""Task 3.1 Red — LogiQA TransferBank as first paper-§3.2.1 transfer bank.

Per cross-domain-transfer-tasks spec: every transfer bank used in Exp 1
must conform to paper §3.2.1 cross-domain task types (logic-puzzle,
factual-qa, reading-comprehension). LogiQA satisfies logic-puzzle.
"""

from pathlib import Path

import pytest

from src.conditioning.banks import TransferBank, PAPER_3_2_1_TASK_TYPES


REPO = Path(__file__).resolve().parents[1]
BANK_PATH = REPO / "configs" / "banks" / "logiqa_v1.yaml"


class TestLogiqaBankExists:
    def test_yaml_file_exists(self):
        assert BANK_PATH.exists(), f"Missing bank at {BANK_PATH}"


class TestLogiqaBankSchema:
    def test_loads_via_TransferBank(self):
        bank = TransferBank.load("logiqa_v1")
        assert bank.bank_id == "logiqa_v1"
        assert bank.bank_type == "transfer"

    def test_all_items_use_logic_puzzle_task_type(self):
        bank = TransferBank.load("logiqa_v1")
        for item in bank.items:
            assert item.task_type == "logic-puzzle"

    def test_passes_exp1_transfer_validation(self):
        """Per cross-domain-transfer-tasks spec: logic-puzzle is in
        PAPER_3_2_1_TASK_TYPES; bank passes use-time validation."""
        bank = TransferBank.load("logiqa_v1")
        bank.validate_for_exp1_transfer()  # no exception

    def test_has_minimum_item_count(self):
        """At least 10 items so per-condition n=5 with stratification works."""
        bank = TransferBank.load("logiqa_v1")
        assert len(bank.items) >= 10

    def test_difficulty_classes_present(self):
        bank = TransferBank.load("logiqa_v1")
        difficulties = {item.difficulty_class for item in bank.items}
        # Per cross-domain-transfer-tasks spec, difficulty_class is
        # easy/medium/hard. We accept any non-empty subset of those.
        assert difficulties.issubset({"easy", "medium", "hard"})
        assert len(difficulties) >= 1
