"""Tests for OpenAIClient usage-token tracking.

Reasoning models (gpt-5.x, o-series) bill on hidden reasoning tokens
that the response content doesn't expose. Without per-call usage
tracking the cost of a calibration run is unknowable from local data.

The client appends a `UsageRecord` to `client.usage_log` after every
successful `complete()` call, capturing prompt_tokens,
completion_tokens, and reasoning_tokens (when present). Failed calls
that never produce a response do NOT append a record.

`client.usage_summary()` aggregates across the log. Tests cover the
contract, not specific token counts (those depend on real model
behavior).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_openai_response(
    content: str,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    reasoning_tokens: int | None = None,
):
    """Build a MagicMock that looks like openai.ChatCompletion's shape."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    # OpenAI exposes reasoning_tokens via completion_tokens_details.reasoning_tokens
    # for reasoning models; when absent the attribute is None.
    if reasoning_tokens is not None:
        details = MagicMock()
        details.reasoning_tokens = reasoning_tokens
        resp.usage.completion_tokens_details = details
    else:
        resp.usage.completion_tokens_details = None
    return resp


class TestOpenAIUsageTracking:
    @pytest.mark.asyncio
    async def test_successful_call_appends_usage_record(self):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_inst.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(
                    "answer", prompt_tokens=120, completion_tokens=80,
                    reasoning_tokens=200,
                )
            )
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5.4-nano", api_key="test")
            assert client.usage_log == []
            await client.complete([{"role": "user", "content": "Q?"}])
            assert len(client.usage_log) == 1
            rec = client.usage_log[0]
            assert rec.prompt_tokens == 120
            assert rec.completion_tokens == 80
            assert rec.reasoning_tokens == 200
            assert rec.model == "gpt-5.4-nano"

    @pytest.mark.asyncio
    async def test_reasoning_tokens_none_when_absent(self):
        """Non-reasoning models don't return completion_tokens_details;
        reasoning_tokens should be None, not 0 — None means 'not reported',
        0 means 'reported as zero'."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_inst.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(
                    "answer", prompt_tokens=50, completion_tokens=30,
                    reasoning_tokens=None,
                )
            )
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-4o", api_key="test")
            await client.complete([{"role": "user", "content": "Q"}])
            assert client.usage_log[0].reasoning_tokens is None

    @pytest.mark.asyncio
    async def test_multiple_calls_accumulate(self):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_inst.chat.completions.create = AsyncMock(
                side_effect=[
                    _mock_openai_response("a", 100, 50, 100),
                    _mock_openai_response("b", 110, 40, 120),
                    _mock_openai_response("c", 90, 60, 80),
                ]
            )
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5.4-nano", api_key="test")
            for _ in range(3):
                await client.complete([{"role": "user", "content": "Q"}])
            assert len(client.usage_log) == 3

    @pytest.mark.asyncio
    async def test_usage_summary_aggregates_totals(self):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_inst.chat.completions.create = AsyncMock(
                side_effect=[
                    _mock_openai_response("a", 100, 50, 100),
                    _mock_openai_response("b", 200, 100, 200),
                ]
            )
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5.4-nano", api_key="test")
            await client.complete([{"role": "user", "content": "Q1"}])
            await client.complete([{"role": "user", "content": "Q2"}])

            summary = client.usage_summary()
            assert summary["n_calls"] == 2
            assert summary["prompt_tokens"] == 300
            assert summary["completion_tokens"] == 150
            assert summary["reasoning_tokens"] == 300
            assert summary["model"] == "gpt-5.4-nano"

    @pytest.mark.asyncio
    async def test_usage_summary_empty_log_returns_zero_totals(self):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI"):
            client = OpenAIClient(model="gpt-5.4-nano", api_key="test")
            summary = client.usage_summary()
            assert summary["n_calls"] == 0
            assert summary["prompt_tokens"] == 0
            assert summary["completion_tokens"] == 0
            assert summary["reasoning_tokens"] == 0

    @pytest.mark.asyncio
    async def test_usage_summary_estimates_cost_with_pricing(self):
        """Given pricing in $ per 1M tokens, summary returns USD estimate.
        OpenAI's `reasoning_tokens` field is a breakdown of `completion_tokens`,
        not a separate billable line — the output rate applies to
        `completion_tokens` only, and reasoning is already inside that count."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_inst.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response(
                    "x", prompt_tokens=1_000_000, completion_tokens=1_000_000,
                    reasoning_tokens=500_000,
                )
            )
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5.4-nano", api_key="test")
            await client.complete([{"role": "user", "content": "Q"}])

            # $0.05 per 1M input, $0.40 per 1M output → $0.05 + $0.40 = $0.45
            # Reasoning tokens (500K) are a subset of the 1M completion total,
            # not added separately. Cost = prompt × in_rate + completion × out_rate.
            summary = client.usage_summary(
                input_usd_per_million=0.05,
                output_usd_per_million=0.40,
            )
            assert summary["estimated_usd"] == pytest.approx(0.45, rel=1e-6)

    @pytest.mark.asyncio
    async def test_usage_log_isolated_per_client_instance(self):
        """Each client has its own usage_log — class-level state would
        cross-contaminate concurrent runs."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_inst.chat.completions.create = AsyncMock(
                return_value=_mock_openai_response("x", 10, 5, None)
            )
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            c1 = OpenAIClient(model="gpt-5.4-nano", api_key="t")
            c2 = OpenAIClient(model="gpt-5.4-nano", api_key="t")
            await c1.complete([{"role": "user", "content": "Q"}])
            assert len(c1.usage_log) == 1
            assert len(c2.usage_log) == 0
