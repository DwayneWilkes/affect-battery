"""Tests for the experiment runner."""

import asyncio
import json
import hashlib
from dataclasses import asdict
from pathlib import Path

from src.runner import ExperimentConfig, ExperimentType, RunResult, run_single, save_result
from src.conditioning.prompts import Condition
from src.models import DryRunClient


class TestRunResult:
    def test_checksum_excludes_self(self):
        """Checksum must be verifiable: recomputing should give the same result."""
        result = RunResult(
            config={"model": "test", "condition": "neutral"},
            run_number=0,
            conditioning_responses=["42"],
            conditioning_correct=[True],
            transfer_responses=["Paris"],
            transfer_questions=["What is the capital of France?"],
            transfer_expected=["Paris"],
            start_time=1000.0,
            end_time=1001.0,
        )
        first = result.compute_checksum()
        second = result.compute_checksum()
        assert first == second, "Checksum is not stable across recomputations"

    def test_checksum_detects_tampering(self):
        result = RunResult(
            config={"model": "test"},
            run_number=0,
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=["Paris"],
            transfer_questions=["Q"],
            transfer_expected=["Paris"],
            start_time=1000.0,
            end_time=1001.0,
        )
        result.compute_checksum()
        original = result.checksum
        result.transfer_responses = ["London"]  # tamper
        result.compute_checksum()
        assert result.checksum != original


class TestRunSingle:
    def test_dry_run_completes(self):
        config = ExperimentConfig(
            model_name="test",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            num_conditioning_turns=2,
            num_transfer_questions=2,
            seed=42,
        )
        client = DryRunClient(responses=["42", "42", "Paris", "1989"])
        result = asyncio.run(run_single(config, client, 0))
        assert len(result.conditioning_responses) == 2
        assert len(result.transfer_responses) == 2
        assert result.checksum != ""

    def test_no_conditioning_skips(self):
        config = ExperimentConfig(
            model_name="test",
            condition=Condition.NO_CONDITIONING,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            num_conditioning_turns=5,
            num_transfer_questions=2,
            seed=42,
        )
        client = DryRunClient(responses=["Paris", "1989"])
        result = asyncio.run(run_single(config, client, 0))
        assert len(result.conditioning_responses) == 0
        assert len(result.transfer_responses) == 2

    def test_cross_session_clears_history(self):
        config = ExperimentConfig(
            model_name="test",
            condition=Condition.STRONG_NEGATIVE,
            experiment_type=ExperimentType.TRANSFER_CROSS,
            num_runs=1,
            num_conditioning_turns=2,
            num_transfer_questions=1,
            seed=42,
        )
        client = DryRunClient(responses=["42", "42", "Paris"])
        result = asyncio.run(run_single(config, client, 0))
        # Should still produce results
        assert len(result.transfer_responses) == 1


class TestSaveResult:
    def test_save_creates_file(self, tmp_path):
        result = RunResult(
            config={"model_name": "test-model", "condition": "neutral", "experiment_type": "transfer_within"},
            run_number=0,
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=["Paris"],
            transfer_questions=["Q"],
            transfer_expected=["Paris"],
            start_time=1000.0,
            end_time=1001.0,
        )
        result.compute_checksum()
        path = save_result(result, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["checksum"] != ""

    def test_deterministic_filename(self, tmp_path):
        result = RunResult(
            config={"model_name": "meta-llama/Llama-3", "condition": "strong_negative", "experiment_type": "transfer_within"},
            run_number=17,
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=[],
            transfer_questions=[],
            transfer_expected=[],
            start_time=0,
            end_time=0,
        )
        result.compute_checksum()
        path = save_result(result, tmp_path)
        # New layout: <output_dir>/<condition>/<NNNN>.json. The model lives
        # at the pilot-dir layer (set by cmd_pilot), and the experiment_type
        # lives at the parent of `output_dir`. Test the deterministic
        # (condition, run_number) encoding the leaf path is responsible for.
        assert path.parent.name == "strong_negative"
        assert path.name == "0017.json"
