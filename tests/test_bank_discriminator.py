"""Task 0.3 Red — unified bank loader with `bank_type` discriminator + `status`.

Per design.md D9 + cross-domain-transfer-tasks spec:
- ArithmeticBank carries `bank_type: "arithmetic"`, default
- TransferBank carries `bank_type: "transfer"`, default
- Both carry `status: "active" | "archived" | "candidate"`
- TransferBank.validate_for_exp1_transfer() rejects arithmetic task_type
  with paper §3.2.1 error citation (use-time check, not load-time, so
  existing arithmetic-task-type tests still pass)
"""

from pathlib import Path
import textwrap

import pytest
import yaml

from src.conditioning.banks import (
    ArithmeticBank,
    TransferBank,
)


def _write_arithmetic_bank(tmp_path: Path, bank_id: str = "arith_v1") -> Path:
    bank_path = tmp_path / f"{bank_id}.yaml"
    bank_path.write_text(textwrap.dedent(f"""
        bank_id: {bank_id}
        bank_version: 1
        difficulty_profile: {{}}
        items:
          - id: a1
            problem: "1 + 1"
            answer: "2"
            operator: add
            operand_a: 1
            operand_b: 1
            digit_count_a: 1
            digit_count_b: 1
            n_carries: 0
            requires_borrow: false
    """))
    return bank_path


def _write_transfer_bank(
    tmp_path: Path,
    bank_id: str = "transfer_v1",
    task_type: str = "logic-puzzle",
) -> Path:
    bank_path = tmp_path / f"{bank_id}.yaml"
    bank_path.write_text(textwrap.dedent(f"""
        bank_id: {bank_id}
        bank_version: 1
        difficulty_profile: {{}}
        items:
          - id: t1
            prompt: "All A are B. Some B are C. Conclude?"
            expected_answer: "Some A may be C; cannot determine"
            task_type: {task_type}
            difficulty_class: easy
    """))
    return bank_path


class TestArithmeticBankDiscriminator:
    def test_arithmetic_bank_has_bank_type_attribute(self, tmp_path):
        _write_arithmetic_bank(tmp_path)
        bank = ArithmeticBank.load("arith_v1", banks_dir=tmp_path)
        assert bank.bank_type == "arithmetic"

    def test_arithmetic_bank_default_status_is_active(self, tmp_path):
        _write_arithmetic_bank(tmp_path)
        bank = ArithmeticBank.load("arith_v1", banks_dir=tmp_path)
        assert bank.status == "active"

    def test_arithmetic_bank_loads_archived_status(self, tmp_path):
        bank_path = tmp_path / "arith_archived.yaml"
        bank_path.write_text(textwrap.dedent("""
            bank_id: arith_archived
            bank_version: 1
            status: archived
            difficulty_profile: {}
            items:
              - id: a1
                problem: "1 + 1"
                answer: "2"
                operator: add
                operand_a: 1
                operand_b: 1
                digit_count_a: 1
                digit_count_b: 1
                n_carries: 0
                requires_borrow: false
        """))
        bank = ArithmeticBank.load("arith_archived", banks_dir=tmp_path)
        assert bank.status == "archived"


class TestTransferBankDiscriminator:
    def test_transfer_bank_has_bank_type_attribute(self, tmp_path):
        _write_transfer_bank(tmp_path)
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert bank.bank_type == "transfer"

    def test_transfer_bank_default_status_is_active(self, tmp_path):
        _write_transfer_bank(tmp_path)
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert bank.status == "active"


class TestExp1TransferValidation:
    """validate_for_exp1_transfer rejects arithmetic / reasoning task types
    that don't conform to paper §3.2.1's cross-domain definition."""

    def test_logic_puzzle_passes_validation(self, tmp_path):
        _write_transfer_bank(tmp_path, task_type="logic-puzzle")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        bank.validate_for_exp1_transfer()  # no exception

    def test_factual_qa_passes_validation(self, tmp_path):
        _write_transfer_bank(tmp_path, task_type="factual-qa")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        bank.validate_for_exp1_transfer()

    def test_reading_comprehension_passes_validation(self, tmp_path):
        _write_transfer_bank(tmp_path, task_type="reading-comprehension")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        bank.validate_for_exp1_transfer()

    def test_arithmetic_rejected_for_exp1_transfer(self, tmp_path):
        _write_transfer_bank(tmp_path, task_type="arithmetic")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        with pytest.raises(ValueError, match="paper §3.2.1"):
            bank.validate_for_exp1_transfer()

    def test_reasoning_rejected_for_exp1_transfer(self, tmp_path):
        _write_transfer_bank(tmp_path, task_type="reasoning")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        with pytest.raises(ValueError, match="paper §3.2.1"):
            bank.validate_for_exp1_transfer()

    def test_recall_rejected_for_exp1_transfer(self, tmp_path):
        _write_transfer_bank(tmp_path, task_type="recall")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        with pytest.raises(ValueError, match="paper §3.2.1"):
            bank.validate_for_exp1_transfer()
