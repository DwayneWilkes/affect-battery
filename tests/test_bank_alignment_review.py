"""Candidate-bank alignment-review gate .

Per cross-domain-transfer-tasks/spec.md "Alignment review for re-scoped
transfer banks":
- Bank with status='candidate' and no alignment_review.verdict='pass'
  is filtered out of primary-analysis aggregation.
- Upgrading status from candidate → active without alignment_review.verdict='pass'
  raises ValidationError.
- Runs using a candidate bank carry candidate_bank=True flag in result config.
"""

from pathlib import Path
import textwrap

import pytest

from src.conditioning.banks import (
    TransferBank,
    AlignmentReview,
    BankAlignmentError,
    is_primary_analysis_eligible,
)


def _write_transfer_bank(
    tmp_path: Path,
    bank_id: str = "transfer_v1",
    task_type: str = "logic-puzzle",
    status: str = "active",
    alignment_review: dict | None = None,
) -> Path:
    bank_path = tmp_path / f"{bank_id}.yaml"
    extra = ""
    if alignment_review is not None:
        review_yaml = "\n".join(f"          {k}: {v}" for k, v in alignment_review.items())
        extra = f"\n        alignment_review:\n{review_yaml}"
    bank_path.write_text(textwrap.dedent(f"""
        bank_id: {bank_id}
        bank_version: 1
        status: {status}{extra}
        difficulty_profile: {{}}
        items:
          - id: t1
            prompt: "All A are B. Some B are C."
            expected_answer: "Cannot determine"
            task_type: {task_type}
            difficulty_class: easy
    """))
    return bank_path


class TestAlignmentReviewSchema:
    def test_alignment_review_schema_loaded(self, tmp_path):
        _write_transfer_bank(
            tmp_path,
            status="active",
            alignment_review={
                "reviewer": "Dwayne",
                "date": "2026-04-24",
                "verdict": "pass",
            },
        )
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert bank.alignment_review is not None
        assert bank.alignment_review.reviewer == "Dwayne"
        assert bank.alignment_review.verdict == "pass"

    def test_alignment_review_absent_yields_none(self, tmp_path):
        _write_transfer_bank(tmp_path, status="active")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert bank.alignment_review is None


class TestPrimaryAnalysisEligibility:
    def test_active_bank_is_eligible(self, tmp_path):
        _write_transfer_bank(tmp_path, status="active")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert is_primary_analysis_eligible(bank) is True

    def test_candidate_without_review_is_not_eligible(self, tmp_path):
        _write_transfer_bank(tmp_path, status="candidate")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert is_primary_analysis_eligible(bank) is False

    def test_candidate_with_passed_review_is_eligible(self, tmp_path):
        _write_transfer_bank(
            tmp_path,
            status="candidate",
            alignment_review={
                "reviewer": "Dwayne",
                "date": "2026-04-24",
                "verdict": "pass",
            },
        )
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert is_primary_analysis_eligible(bank) is True

    def test_candidate_with_failed_review_is_not_eligible(self, tmp_path):
        _write_transfer_bank(
            tmp_path,
            status="candidate",
            alignment_review={
                "reviewer": "Dwayne",
                "date": "2026-04-24",
                "verdict": "fail",
            },
        )
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert is_primary_analysis_eligible(bank) is False

    def test_archived_bank_is_not_eligible(self, tmp_path):
        _write_transfer_bank(tmp_path, status="archived")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        assert is_primary_analysis_eligible(bank) is False


class TestUpgradeFromCandidate:
    def test_upgrade_without_passed_review_raises(self, tmp_path):
        _write_transfer_bank(tmp_path, status="candidate")
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        with pytest.raises(BankAlignmentError, match="alignment review"):
            bank.upgrade_to_active()

    def test_upgrade_with_passed_review_succeeds(self, tmp_path):
        _write_transfer_bank(
            tmp_path,
            status="candidate",
            alignment_review={
                "reviewer": "Dwayne",
                "date": "2026-04-24",
                "verdict": "pass",
            },
        )
        bank = TransferBank.load("transfer_v1", banks_dir=tmp_path)
        bank.upgrade_to_active()
        assert bank.status == "active"
