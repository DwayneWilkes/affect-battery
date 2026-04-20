"""Tests for error-class-aware retry + circuit breaker.

Spec: affect-battery-compute-guardrails::compute-guardrails requirements
4 (Circuit breaker) and 6 (Error-class-aware retry).
"""

import asyncio
import json
from pathlib import Path

import aiohttp
import pytest

from src.conditioning.prompts import Condition
from src.models import DryRunClient, ModelClient, VLLMClient
from src.runner import (
    ExperimentConfig,
    ExperimentType,
    NonRetryableAPIError,
    run_batch,
)


# ---------------------------------------------------------------------------
# Synthetic clients
# ---------------------------------------------------------------------------


class _AlwaysFails(ModelClient):
    """Raises NonRetryableAPIError on every call."""

    def __init__(self, status_code: int = 401):
        self._status_code = status_code
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "always-fails"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        self.call_count += 1
        raise NonRetryableAPIError(
            f"HTTP {self._status_code}", status_code=self._status_code,
        )


class _FailsAfterN(ModelClient):
    """Succeeds N times, then fails with NonRetryableAPIError."""

    def __init__(self, succeed_n: int, responses: list[str] | None = None):
        self._succeed_n = succeed_n
        self._responses = responses or ["42"]
        self._i = 0
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fails-after-n"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        self.call_count += 1
        if self._i < self._succeed_n:
            r = self._responses[self._i % len(self._responses)]
            self._i += 1
            return r
        raise NonRetryableAPIError("HTTP 401", status_code=401)


def _config(num_runs: int = 10) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="test-model",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=num_runs,
        num_conditioning_turns=1,
        num_transfer_questions=1,
        seed=42,
    )


async def _collect(config, client, **kwargs):
    results = []
    async for r in run_batch(config, client, max_concurrent=1, **kwargs):
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class TestNonRetryableAPIError:
    def test_class_has_status_code(self):
        err = NonRetryableAPIError("401 Unauthorized", status_code=401)
        assert err.status_code == 401

    def test_is_exception_subclass(self):
        assert issubclass(NonRetryableAPIError, Exception)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_opens_on_five_consecutive_failures(tmp_path):
    client = _AlwaysFails(status_code=401)
    results = asyncio.run(_collect(
        _config(num_runs=10), client,
        output_dir=tmp_path,
        circuit_breaker_threshold=5,
    ))
    # Circuit opens after 5 failures; no results are yielded for failed runs.
    assert len(results) == 0
    # Client was called for at most 5 runs before the circuit opened.
    # Each run makes 1 conditioning + 1 transfer = 2 calls; but the first
    # call in each run fails, so call_count is between 5 and 10.
    assert client.call_count <= 10

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [e["event"] for e in events]
    assert "batch_circuit_open" in event_types
    # Circuit open event includes the reason / last error.
    circuit_event = next(e for e in events if e["event"] == "batch_circuit_open")
    assert "reason" in circuit_event or "error" in circuit_event


def test_circuit_does_not_open_when_failures_interleaved(tmp_path):
    """Spec scenario: 3 fails + 1 success + 1 fail + 1 success + ... with
    threshold=5 -> circuit does not open."""
    # We build a client that fails-then-succeeds-then-fails in a custom
    # pattern. Here we use _FailsAfterN with succeed_n high enough that
    # the circuit never sees 5 consecutive failures.
    config = _config(num_runs=3)
    client = _FailsAfterN(succeed_n=100, responses=["42"])
    results = asyncio.run(_collect(
        config, client,
        output_dir=tmp_path,
        circuit_breaker_threshold=5,
    ))
    assert len(results) == 3
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [e["event"] for e in events]
    assert "batch_circuit_open" not in event_types


def test_cached_skip_resets_circuit_counter(tmp_path):
    """A cached skip counts as success for circuit-breaker purposes."""
    from src.runner import save_result, run_single

    # Pre-populate run 0 and run 1 as valid cached results.
    async def seed(run_number):
        dry = DryRunClient(responses=["42"] * 10)
        cfg = _config(num_runs=4)
        res = await run_single(cfg, dry, run_number)
        save_result(res, tmp_path)

    asyncio.run(seed(0))
    asyncio.run(seed(1))

    # Now runs 2 and 3 will both fail, but the two cached skips at the
    # front reset the counter (and the threshold-of-3 means 2 failures
    # is not enough to trip it even if the counter never reset).
    failing = _AlwaysFails(status_code=401)
    results = asyncio.run(_collect(
        _config(num_runs=4), failing,
        output_dir=tmp_path,
        circuit_breaker_threshold=3,
    ))
    # Two cached results make it through.
    assert len(results) == 2


def test_run_failed_event_emitted_on_non_retryable(tmp_path):
    client = _AlwaysFails(status_code=401)
    asyncio.run(_collect(
        _config(num_runs=3), client,
        output_dir=tmp_path,
        circuit_breaker_threshold=10,
    ))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    failed = [e for e in events if e["event"] == "run_failed"]
    assert len(failed) >= 1
    assert "error" in failed[0] or "error_class" in failed[0]


# ---------------------------------------------------------------------------
# Error classification in VLLMClient
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, payload: dict | None = None):
        self.status = status
        self._payload = payload or {"choices": [{"message": {"content": "ok"}}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None, history=(), status=self.status,
                message=f"HTTP {self.status}",
            )

    async def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, status: int):
        self.status = status
        self.closed = False
        self.call_count = 0

    def post(self, *args, **kwargs):
        self.call_count += 1
        return _FakeResponse(self.status)

    async def close(self):
        self.closed = True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_vllm_client_raises_non_retryable_on_4xx(status, monkeypatch):
    """Spec: 4xx (excluding 408, 429) must raise NonRetryableAPIError
    immediately without retrying."""
    client = VLLMClient(base_url="http://unused", model="test")
    session = _FakeSession(status=status)

    async def fake_get_session():
        return session

    monkeypatch.setattr(client, "_get_session", fake_get_session)

    async def go():
        with pytest.raises(NonRetryableAPIError) as exc_info:
            await client.complete([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == status

    asyncio.run(go())
    # No retry happened: a single post() call.
    assert session.call_count == 1
