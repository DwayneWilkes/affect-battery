"""Tests for --bank CLI flag propagation + cache identity by bank hash.

Spec: affect-battery-task-difficulty-calibration::conditioning-protocol::
"Arithmetic bank selection via --bank CLI flag" +
task-difficulty-calibration::"Stimulus bank schema" (cache-identity clause).

Tasks 5.1-5.7 from
`specs/changes/affect-battery-task-difficulty-calibration/tasks.md`.
"""

import asyncio
import json
from pathlib import Path

import pytest

from src.conditioning.prompts import Condition
from src.runner import ExperimentConfig, ExperimentType, RunResult


class TestExperimentConfigStimulusBankField:
    """ExperimentConfig MUST carry stimulus_bank + stimulus_bank_hash so
    the CLI can thread the selected bank into per-run config dicts."""

    def test_stimulus_bank_default_is_arithmetic_easy_v1(self):
        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
        )
        assert cfg.stimulus_bank == "arithmetic_easy_v1"

    def test_stimulus_bank_hash_default_is_empty(self):
        """The hash is assigned by the CLI from the loaded bank; the default
        is '' (not set) so the runner itself doesn't need to resolve it."""
        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
        )
        assert cfg.stimulus_bank_hash == ""

    def test_stimulus_bank_settable(self):
        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            stimulus_bank="arithmetic_hard_v1",
            stimulus_bank_hash="abcdef1234567890" * 4,
        )
        assert cfg.stimulus_bank == "arithmetic_hard_v1"
        assert cfg.stimulus_bank_hash.startswith("abcdef")


class TestStimulusBankInSavedResults:
    def test_saved_result_config_carries_stimulus_bank(self, tmp_path):
        from src.runner import save_result

        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            stimulus_bank="arithmetic_hard_v1",
            stimulus_bank_hash="deadbeefcafebabe" * 4,
        )
        result = RunResult(
            config={
                "model_name": "m",
                "condition": "neutral",
                "experiment_type": "transfer_within",
                "stimulus_bank": cfg.stimulus_bank,
                "stimulus_bank_hash": cfg.stimulus_bank_hash,
            },
            run_number=0,
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=[],
            transfer_questions=[],
            transfer_expected=[],
            start_time=0.0,
            end_time=0.0,
        )
        result.compute_checksum()
        path = save_result(result, tmp_path)
        data = json.loads(path.read_text())
        assert data["config"]["stimulus_bank"] == "arithmetic_hard_v1"
        assert data["config"]["stimulus_bank_hash"].startswith("deadbeef")


class TestIsValidCachedResultRejectsBankMismatch:
    """is_valid_cached_result MUST reject a cached file whose
    stimulus_bank_hash doesn't match the currently-loaded bank.
    This is the guard against silently mixing banks in aggregated analysis."""

    def test_mismatched_bank_hash_rejected(self, tmp_path):
        from src.runner import is_valid_cached_result

        cached = {
            "config": {
                "model_name": "m",
                "condition": "neutral",
                "experiment_type": "transfer_within",
                "stimulus_bank": "arithmetic_hard_v1",
                "stimulus_bank_hash": "OLD_HASH_FROM_PRIOR_BANK_EDITS",
            },
            "run_number": 0,
            "conditioning_responses": [],
            "conditioning_correct": [],
            "transfer_responses": [],
            "transfer_questions": [],
            "transfer_expected": [],
            "start_time": 0.0,
            "end_time": 0.0,
            "checksum": "x" * 16,
        }
        path = tmp_path / "cached.json"
        path.write_text(json.dumps(cached))
        # Current bank hash differs => cache miss.
        assert is_valid_cached_result(
            path, expected_stimulus_bank_hash="NEW_HASH_AFTER_EDIT"
        ) is False

    def test_matching_bank_hash_accepted(self, tmp_path):
        """If expected_stimulus_bank_hash matches the cached file's hash
        AND the baseline checksum validation would pass, the cache is valid.
        Exercised with matching hashes (baseline checksum validation handled
        by existing test_output_schema.py)."""
        from src.runner import is_valid_cached_result

        # Build a well-formed cached result (reuse save_result to get a
        # checksum that matches).
        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            stimulus_bank="arithmetic_hard_v1",
            stimulus_bank_hash="SAME_HASH",
        )
        result = RunResult(
            config={
                "model_name": "m",
                "condition": "neutral",
                "experiment_type": "transfer_within",
                "stimulus_bank": cfg.stimulus_bank,
                "stimulus_bank_hash": cfg.stimulus_bank_hash,
            },
            run_number=0,
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=[],
            transfer_questions=[],
            transfer_expected=[],
            start_time=0.0,
            end_time=0.0,
        )
        result.compute_checksum()
        from src.runner import save_result
        path = save_result(result, tmp_path)
        assert is_valid_cached_result(
            path, expected_stimulus_bank_hash="SAME_HASH"
        ) is True

    def test_no_expected_hash_falls_back_to_existing_behavior(self, tmp_path):
        """Backward-compat: calling is_valid_cached_result without
        expected_stimulus_bank_hash skips the bank-hash check (legacy cached
        results without stimulus_bank_hash field remain valid)."""
        from src.runner import is_valid_cached_result

        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
        )
        result = RunResult(
            config={
                "model_name": "m",
                "condition": "neutral",
                "experiment_type": "transfer_within",
                "stimulus_bank": cfg.stimulus_bank,
                "stimulus_bank_hash": cfg.stimulus_bank_hash,
            },
            run_number=0,
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=[],
            transfer_questions=[],
            transfer_expected=[],
            start_time=0.0,
            end_time=0.0,
        )
        result.compute_checksum()
        from src.runner import save_result
        path = save_result(result, tmp_path)
        # No expected hash argument => skip bank-hash check, rely on checksum.
        assert is_valid_cached_result(path) is True


class TestCliBankFlag:
    """--bank CLI flag selects the named bank and rejects unknown names."""

    def test_unknown_bank_exits_nonzero_with_listing(self, monkeypatch, capsys):
        from src import cli
        import sys

        class Args:
            dry_run = True
            base_model = False
            model = "test-model"
            base_url = "http://unused"
            temperature = 0.7
            output_dir = "/tmp/unknown-bank-test"
            max_concurrent = 1
            budget_max_calls = None
            cost_per_call = None
            rate_limit_rps = None
            circuit_breaker_threshold = 5
            bank = "arithmetic_nonexistent_v99"  # not in configs/banks/

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_pilot(Args())
        assert exit_info.value.code != 0
        captured = capsys.readouterr()
        # Error message lists known bank ids so the user can correct.
        assert "arithmetic_easy_v1" in captured.err or "arithmetic_easy_v1" in captured.out
