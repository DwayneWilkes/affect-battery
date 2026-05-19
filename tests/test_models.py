"""Tests for model clients."""

import asyncio
from src.models import DryRunClient, UsageRecord, capture_call_usage


class TestDryRunClient:
    def test_cycles_responses(self):
        client = DryRunClient(responses=["a", "b", "c"])
        results = [asyncio.run(client.complete([])) for _ in range(5)]
        assert results == ["a", "b", "c", "a", "b"]

    def test_model_name(self):
        client = DryRunClient(model="test-model")
        assert client.model_name == "test-model"


class _FakeClientWithUsageLog:
    def __init__(self):
        self.usage_log = []

    def record(self, prompt: int, completion: int, reasoning: int | None = None):
        self.usage_log.append(UsageRecord(
            model="x", prompt_tokens=prompt, completion_tokens=completion,
            reasoning_tokens=reasoning,
        ))


class TestCaptureCallUsage:
    def test_returns_none_when_client_has_no_usage_log(self):
        client = DryRunClient(model="x")
        assert capture_call_usage(client, before_idx=0) is None

    def test_returns_none_when_no_new_records_since_before(self):
        client = _FakeClientWithUsageLog()
        client.record(100, 50)
        # before_idx points past the existing record
        assert capture_call_usage(client, before_idx=1) is None

    def test_aggregates_records_appended_since_before(self):
        client = _FakeClientWithUsageLog()
        client.record(100, 50)  # pre-existing
        before = len(client.usage_log)
        client.record(60, 30, reasoning=120)
        client.record(40, 20, reasoning=80)
        usage = capture_call_usage(client, before)
        assert usage["n_calls"] == 2
        assert usage["prompt_tokens"] == 100  # 60 + 40
        assert usage["completion_tokens"] == 50  # 30 + 20
        assert usage["reasoning_tokens"] == 200  # 120 + 80

    def test_treats_none_reasoning_as_zero(self):
        client = _FakeClientWithUsageLog()
        before = len(client.usage_log)
        client.record(100, 50, reasoning=None)
        usage = capture_call_usage(client, before)
        assert usage["reasoning_tokens"] == 0
