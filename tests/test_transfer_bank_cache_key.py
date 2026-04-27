"""Cache identity must include the transfer bank: changing --transfer-bank
must invalidate any cached results from a previous run with a different
(or no) transfer bank.

The bug this guards against: pre-fix, the cache key was (output_dir,
condition, run_number). Two runs of `claude-haiku-4-5 / strong_negative
/ run 0` with different transfer banks (legacy hardcoded pool vs.
TriviaQA hard) would write to the same file path, and the second run
would see the first run's cached file as 'valid' and skip the API call.

Spec: affect-battery-proposal-realignment :: scoring-pipeline,
results-layout.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.conditioning.prompts import Condition
from src.models import ModelClient
from src.runner import (
    ExperimentConfig,
    ExperimentType,
    is_valid_cached_result,
    run_batch,
    run_single,
    save_result,
)


class _ScriptedClient(ModelClient):
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

    async def complete_text(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        raise NotImplementedError


def _bank_yaml(tmp_path: Path, name: str, items: list[dict]) -> Path:
    import yaml
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump({"bank_id": name, "items": items}))
    return p


def _config(condition: Condition, transfer_bank: str = "",
            transfer_bank_hash: str = "") -> ExperimentConfig:
    return ExperimentConfig(
        model_name="scripted-model",
        condition=condition,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=1,
        num_conditioning_turns=1,
        num_transfer_questions=1,
        seed=42,
        transfer_bank=transfer_bank,
        transfer_bank_hash=transfer_bank_hash,
    )


# --- 1. ExperimentConfig carries transfer_bank_hash -------------------------

class TestExperimentConfigTransferBankHash:
    def test_transfer_bank_hash_field_default_empty(self):
        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
        )
        assert cfg.transfer_bank_hash == ""

    def test_transfer_bank_hash_settable(self):
        cfg = _config(Condition.NEUTRAL, transfer_bank_hash="abc123")
        assert cfg.transfer_bank_hash == "abc123"


# --- 2. is_valid_cached_result rejects mismatched transfer_bank_hash --------

class TestIsValidCachedResultTransferBank:
    def test_matching_transfer_bank_hash_validates(self, tmp_path):
        cfg = _config(Condition.NEUTRAL, transfer_bank_hash="hash_a")
        client = _ScriptedClient(responses=["7", "Some answer."])
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        assert is_valid_cached_result(
            path, expected_transfer_bank_hash="hash_a",
        ) is True

    def test_mismatched_transfer_bank_hash_invalidates(self, tmp_path):
        """Cached file written under hash_a must NOT validate when the
        live config has hash_b. Otherwise re-piloting with a new
        transfer bank silently returns stale results."""
        cfg = _config(Condition.NEUTRAL, transfer_bank_hash="hash_a")
        client = _ScriptedClient(responses=["7", "Some answer."])
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        assert is_valid_cached_result(
            path, expected_transfer_bank_hash="hash_b",
        ) is False

    def test_no_expected_hash_passes_through(self, tmp_path):
        """When the caller doesn't supply expected_transfer_bank_hash,
        validation falls back to the prior behavior (schema+checksum
        only). Backward compat for callers that don't care about the
        transfer bank."""
        cfg = _config(Condition.NEUTRAL, transfer_bank_hash="hash_a")
        client = _ScriptedClient(responses=["7", "Some answer."])
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        # Calling without expected_transfer_bank_hash works.
        assert is_valid_cached_result(path) is True


# --- 3. run_batch invalidates cache when transfer_bank_hash changes ---------

class TestRunBatchTransferBankCacheInvalidation:
    def test_run_batch_re_runs_when_transfer_bank_hash_differs(self, tmp_path):
        """End-to-end contract: a run_batch with a different
        transfer_bank_hash than the cached file's MUST re-call the API
        rather than silently returning the cached result."""
        bank_a = _bank_yaml(tmp_path, "bank_a", [
            {"id": "1", "question": "Q?", "expected": "A",
             "answer_aliases": [], "difficulty": "easy"},
        ])
        bank_b = _bank_yaml(tmp_path, "bank_b", [
            {"id": "1", "question": "Q?", "expected": "B",
             "answer_aliases": [], "difficulty": "easy"},
        ])

        out_dir = tmp_path / "results"
        cfg_a = ExperimentConfig(
            model_name="scripted-model",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            num_conditioning_turns=1,
            num_transfer_questions=1,
            seed=42,
            transfer_bank=str(bank_a),
            transfer_bank_hash="hash_a",
        )

        async def go(cfg, client):
            out = []
            async for r in run_batch(cfg, client, output_dir=out_dir,
                                     max_concurrent=1):
                out.append(r)
            return out

        client_a = _ScriptedClient(responses=["7", "Answer A."])
        results_a = asyncio.run(go(cfg_a, client_a))
        assert len(results_a) == 1
        assert client_a._i == 2  # 1 conditioning + 1 transfer

        # Pass 2: same everything EXCEPT transfer_bank + hash. Must NOT
        # serve from cache; must re-call the API.
        cfg_b = ExperimentConfig(
            model_name="scripted-model",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            num_conditioning_turns=1,
            num_transfer_questions=1,
            seed=42,
            transfer_bank=str(bank_b),
            transfer_bank_hash="hash_b",
        )
        client_b = _ScriptedClient(responses=["7", "Answer B."])
        results_b = asyncio.run(go(cfg_b, client_b))
        assert len(results_b) == 1
        assert client_b._i == 2, (
            "expected fresh API calls for new transfer_bank_hash; "
            f"client made {client_b._i} calls"
        )

    def test_run_batch_resumes_from_cache_when_hash_matches(self, tmp_path):
        """Inverse: same transfer_bank_hash → cache hits, no API call."""
        bank = _bank_yaml(tmp_path, "bank", [
            {"id": "1", "question": "Q?", "expected": "A",
             "answer_aliases": [], "difficulty": "easy"},
        ])
        out_dir = tmp_path / "results"
        cfg = ExperimentConfig(
            model_name="scripted-model",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            num_conditioning_turns=1,
            num_transfer_questions=1,
            seed=42,
            transfer_bank=str(bank),
            transfer_bank_hash="hash_x",
        )

        async def go(client):
            out = []
            async for r in run_batch(cfg, client, output_dir=out_dir,
                                     max_concurrent=1):
                out.append(r)
            return out

        # Pass 1: write a result.
        c1 = _ScriptedClient(responses=["7", "first."])
        asyncio.run(go(c1))
        assert c1._i == 2

        # Pass 2: same config → cache hit → zero API calls.
        c2 = _ScriptedClient(responses=["NEVER", "CALLED"])
        asyncio.run(go(c2))
        assert c2._i == 0
