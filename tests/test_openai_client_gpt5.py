"""OpenAIClient must adapt parameter names per model family.

OpenAI deprecated `max_tokens` in favor of `max_completion_tokens`
for gpt-5.x and o-series (reasoning) models. Older chat models
(gpt-4, gpt-4o, gpt-3.5) still accept `max_tokens`. The client
must inspect the model id and pick the right kwarg.

Spec: affect-battery-proposal-realignment :: model-adapter.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIClientMaxTokensParam:
    @pytest.mark.asyncio
    async def test_gpt5_uses_max_completion_tokens(self):
        """gpt-5.4-nano (and any gpt-5.x) must use max_completion_tokens,
        NOT max_tokens. The API rejects max_tokens for these models."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "hi"
            mock_inst.chat.completions.create = AsyncMock(return_value=resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5.4-nano", api_key="x")
            await client.complete(
                messages=[{"role": "user", "content": "Q"}],
                temperature=0.5,
                max_tokens=256,
            )

            kwargs = mock_inst.chat.completions.create.call_args.kwargs
            assert "max_completion_tokens" in kwargs, (
                f"gpt-5.4-nano must use max_completion_tokens, "
                f"got: {list(kwargs.keys())}"
            )
            assert kwargs["max_completion_tokens"] == 256
            # The legacy parameter MUST NOT also be passed (the API
            # rejects requests with both).
            assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        "gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-mini",
        "gpt-5.4-nano", "gpt-5.4-pro",
    ])
    async def test_all_gpt5_variants_use_max_completion_tokens(self, model):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "ok"
            mock_inst.chat.completions.create = AsyncMock(return_value=resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model=model, api_key="x")
            await client.complete(
                messages=[{"role": "user", "content": "Q"}],
                max_tokens=128,
            )
            kwargs = mock_inst.chat.completions.create.call_args.kwargs
            assert "max_completion_tokens" in kwargs, (
                f"{model}: kwargs were {list(kwargs.keys())}"
            )
            assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_legacy_gpt4o_still_uses_max_tokens(self):
        """Older chat models (gpt-4o, gpt-4-turbo) still take the
        legacy max_tokens parameter. Must not get the renamed kwarg."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "hi"
            mock_inst.chat.completions.create = AsyncMock(return_value=resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-4o", api_key="x")
            await client.complete(
                messages=[{"role": "user", "content": "Q"}],
                max_tokens=200,
            )

            kwargs = mock_inst.chat.completions.create.call_args.kwargs
            assert "max_tokens" in kwargs
            assert kwargs["max_tokens"] == 200
            assert "max_completion_tokens" not in kwargs

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["o3", "o3-mini", "o4-mini"])
    async def test_o_series_uses_max_completion_tokens(self, model):
        """o-series reasoning models also use max_completion_tokens."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "ok"
            mock_inst.chat.completions.create = AsyncMock(return_value=resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model=model, api_key="x")
            await client.complete(
                messages=[{"role": "user", "content": "Q"}],
                max_tokens=200,
            )

            kwargs = mock_inst.chat.completions.create.call_args.kwargs
            assert "max_completion_tokens" in kwargs
            assert "max_tokens" not in kwargs


class TestOpenAIClientModelNameProperty:
    """Sanity: model_name property reflects whatever was passed in."""

    def test_gpt5_nano_model_name(self):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI"):
            client = OpenAIClient(model="gpt-5.4-nano", api_key="x")
            assert client.model_name == "gpt-5.4-nano"
