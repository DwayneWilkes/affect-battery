"""Tests for BatchBudget + pre-flight estimate.

Spec: affect-battery-compute-guardrails::compute-guardrails requirements
2 (Batch budget cap) and 5 (Pre-flight cost estimate).
"""

import asyncio
import json

from src.conditioning.prompts import Condition
from src.models import DryRunClient, ModelClient
from src.runner import (
    BatchBudget,
    ExperimentConfig,
    ExperimentType,
    run_batch,
    run_single,
    save_result,
)


class _Counter(DryRunClient):
    def __init__(self):
        super().__init__(responses=["42"] * 100)
        self.call_count = 0

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        self.call_count += 1
        return await super().complete(messages, temperature, max_tokens)


def _config(num_runs=10, conditioning=1, transfer=1):
    return ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=num_runs,
        num_conditioning_turns=conditioning,
        num_transfer_questions=transfer,
        seed=42,
    )


async def _collect(config, client, tmp_path, **kwargs):
    results = []
    async for r in run_batch(config, client, output_dir=tmp_path, max_concurrent=1, **kwargs):
        results.append(r)
    return results


class TestBatchBudget:
    def test_budget_defaults_to_unbounded(self):
        b = BatchBudget()
        assert b.max_api_calls is None
        assert b.cost_per_call is None

    def test_budget_accepts_max_calls(self):
        b = BatchBudget(max_api_calls=100)
        assert b.max_api_calls == 100

    def test_budget_accepts_cost_per_call(self):
        b = BatchBudget(max_api_calls=100, cost_per_call=0.002)
        assert b.cost_per_call == 0.002


class TestBudgetCapHonored:
    def test_budget_stops_batch_early(self, tmp_path):
        """Budget of 6 calls with 2 calls/run = 3 runs' worth."""
        config = _config(num_runs=10, conditioning=1, transfer=1)  # 2 calls/run
        client = _Counter()
        results = asyncio.run(_collect(
            config, client, tmp_path,
            budget=BatchBudget(max_api_calls=6),
        ))
        assert client.call_count <= 6, f"Overshot budget: {client.call_count}"
        assert len(results) <= 3

    def test_budget_exceeded_event_emitted(self, tmp_path):
        config = _config(num_runs=10, conditioning=1, transfer=1)
        client = _Counter()
        asyncio.run(_collect(
            config, client, tmp_path,
            budget=BatchBudget(max_api_calls=4),
        ))
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert any(e["event"] == "batch_budget_exceeded" for e in events)

    def test_cached_skips_do_not_consume_budget(self, tmp_path):
        """Spec scenario: 1000 runs, 800 cached, budget 500 -> all 200 missing execute."""
        # Scale down: 10 runs, 8 cached, budget of 4 calls (= 2 missing runs).
        config = _config(num_runs=10, conditioning=1, transfer=1)

        async def seed(n):
            dry = DryRunClient(responses=["42"] * 10)
            for i in range(n):
                res = await run_single(config, dry, i)
                save_result(res, tmp_path)

        asyncio.run(seed(8))

        client = _Counter()
        results = asyncio.run(_collect(
            config, client, tmp_path,
            budget=BatchBudget(max_api_calls=4),  # 2 runs worth
        ))
        # All 8 cached + both 2 missing execute.
        assert len(results) == 10
        assert client.call_count == 4


class TestPreflightEvent:
    def test_preflight_reports_execution_count(self, tmp_path):
        config = _config(num_runs=10, conditioning=1, transfer=1)

        async def seed(n):
            dry = DryRunClient(responses=["42"] * 10)
            for i in range(n):
                res = await run_single(config, dry, i)
                save_result(res, tmp_path)

        asyncio.run(seed(4))

        client = _Counter()
        asyncio.run(_collect(config, client, tmp_path))

        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        preflights = [e for e in events if e["event"] == "batch_preflight"]
        assert len(preflights) == 1
        e = preflights[0]
        assert e["runs_total"] == 10
        assert e["runs_cached"] == 4
        assert e["runs_to_execute"] == 6

    def test_preflight_reports_cost_when_configured(self, tmp_path):
        config = _config(num_runs=5, conditioning=1, transfer=1)
        client = _Counter()
        asyncio.run(_collect(
            config, client, tmp_path,
            budget=BatchBudget(max_api_calls=100, cost_per_call=0.01),
        ))
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        preflight = next(e for e in events if e["event"] == "batch_preflight")
        # 5 runs * 2 calls/run = 10 expected calls * $0.01 = $0.10 estimated
        assert "estimated_cost_usd" in preflight
        assert preflight["estimated_cost_usd"] > 0

    def test_preflight_emitted_before_any_api_call(self, tmp_path):
        """The preflight event must appear before the first run_started in
        events.jsonl."""
        config = _config(num_runs=3, conditioning=1, transfer=1)
        client = _Counter()
        asyncio.run(_collect(config, client, tmp_path))

        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        first_preflight_idx = next(i for i, e in enumerate(events) if e["event"] == "batch_preflight")
        first_start_idx = next(i for i, e in enumerate(events) if e["event"] == "run_started")
        assert first_preflight_idx < first_start_idx


class TestBatchCompleted:
    def test_batch_completed_emitted_at_end(self, tmp_path):
        config = _config(num_runs=2, conditioning=1, transfer=1)
        client = _Counter()
        asyncio.run(_collect(config, client, tmp_path))
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "batch_completed"
