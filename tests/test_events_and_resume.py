"""Tests for structured event log and resume-on-partial-failure.

Spec: affect-battery-compute-guardrails::compute-guardrails requirements
1 (Resume-on-partial-failure) and 7 (Structured event log).
"""

import asyncio
import json
from pathlib import Path

import pytest

from src.conditioning.prompts import Condition
from src.models import DryRunClient
from src.runner import (
    EventEmitter,
    ExperimentConfig,
    ExperimentType,
    is_valid_cached_result,
    run_batch,
    save_result,
)


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class TestEventEmitter:
    def test_emit_appends_jsonl(self, tmp_path):
        emitter = EventEmitter(tmp_path / "events.jsonl")
        emitter.emit("run_started", run_name="abc", seed=42)
        emitter.emit("run_completed", run_name="abc", elapsed_s=1.25)

        lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        e1 = json.loads(lines[0])
        assert e1["event"] == "run_started"
        assert e1["run_name"] == "abc"
        assert "timestamp" in e1
        e2 = json.loads(lines[1])
        assert e2["event"] == "run_completed"

    def test_all_events_have_timestamp(self, tmp_path):
        emitter = EventEmitter(tmp_path / "events.jsonl")
        emitter.emit("batch_preflight", runs_to_execute=10)
        line = (tmp_path / "events.jsonl").read_text().strip()
        event = json.loads(line)
        # ISO-8601 UTC: ends in +00:00 or Z, contains T
        assert "T" in event["timestamp"]

    def test_emit_creates_parent_dir(self, tmp_path):
        emitter = EventEmitter(tmp_path / "nested" / "sub" / "events.jsonl")
        emitter.emit("test_event")
        assert (tmp_path / "nested" / "sub" / "events.jsonl").exists()


# ---------------------------------------------------------------------------
# is_valid_cached_result
# ---------------------------------------------------------------------------


def _write_valid_result(tmp_path, run_number: int = 0):
    """Helper: produce a schema-valid, checksum-valid result file."""
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=1,
        seed=42,
    )
    client = DryRunClient(responses=["42", "42", "42", "42", "42",
                                      "Paris", "Jane", "Gold", "1989", "53"] * 2)

    async def go():
        from src.runner import run_single
        result = await run_single(config, client, run_number)
        return save_result(result, tmp_path)

    return asyncio.run(go())


class TestIsValidCachedResult:
    def test_valid_file_passes(self, tmp_path):
        path = _write_valid_result(tmp_path)
        assert is_valid_cached_result(path) is True

    def test_missing_file_fails(self, tmp_path):
        assert is_valid_cached_result(tmp_path / "does_not_exist.json") is False

    def test_tampered_checksum_fails(self, tmp_path):
        path = _write_valid_result(tmp_path)
        data = json.loads(path.read_text())
        data["transfer_responses"] = ["tampered"]
        path.write_text(json.dumps(data))
        assert is_valid_cached_result(path) is False

    def test_missing_required_field_fails(self, tmp_path):
        path = _write_valid_result(tmp_path)
        data = json.loads(path.read_text())
        del data["checksum"]
        path.write_text(json.dumps(data))
        assert is_valid_cached_result(path) is False


# ---------------------------------------------------------------------------
# run_batch resume
# ---------------------------------------------------------------------------


def _canned() -> list[str]:
    return ["42"] * 100


async def _collect_batch(config, client, output_dir):
    results = []
    async for r in run_batch(config, client, output_dir=output_dir):
        results.append(r)
    return results


class _CountingClient(DryRunClient):
    def __init__(self):
        super().__init__(responses=_canned())
        self.call_count = 0

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        self.call_count += 1
        return await super().complete(messages, temperature, max_tokens)


def test_run_batch_skips_cached_runs(tmp_path):
    """Spec scenario: batch of 5 with 3 cached -> API calls for only 2."""
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=5,
        num_conditioning_turns=2,
        num_transfer_questions=2,
        seed=42,
    )
    for run_num in (0, 2, 4):
        _write_valid_result(tmp_path, run_num)

    client = _CountingClient()
    results = asyncio.run(_collect_batch(config, client, tmp_path))

    assert len(results) == 5
    # 2 missing runs * (2 conditioning + 2 transfer) = 8 API calls.
    assert client.call_count == 8


def test_run_batch_emits_skipped_event(tmp_path):
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=2,
        num_conditioning_turns=1,
        num_transfer_questions=1,
        seed=42,
    )
    _write_valid_result(tmp_path, 0)
    asyncio.run(_collect_batch(config, DryRunClient(responses=_canned()), tmp_path))

    events_path = tmp_path / "events.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    event_types = [e["event"] for e in events]
    assert "run_skipped_cached" in event_types


def test_run_batch_emits_started_and_completed(tmp_path):
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=2,
        num_conditioning_turns=1,
        num_transfer_questions=1,
        seed=42,
    )
    asyncio.run(_collect_batch(config, DryRunClient(responses=_canned()), tmp_path))

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    started = [e for e in events if e["event"] == "run_started"]
    completed = [e for e in events if e["event"] == "run_completed"]
    assert len(started) == 2
    assert len(completed) == 2


def test_tampered_cached_result_is_re_executed(tmp_path):
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=1,
        num_conditioning_turns=1,
        num_transfer_questions=1,
        seed=42,
    )
    path = _write_valid_result(tmp_path, 0)
    data = json.loads(path.read_text())
    data["transfer_responses"] = ["tampered"]
    path.write_text(json.dumps(data))

    client = _CountingClient()
    asyncio.run(_collect_batch(config, client, tmp_path))

    assert client.call_count > 0
    fresh_data = json.loads(path.read_text())
    assert "tampered" not in fresh_data["transfer_responses"]
