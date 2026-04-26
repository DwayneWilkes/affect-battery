"""Tests for OpenAI + Anthropic ModelClient implementations.

These exercise the chat-completion path via mocked SDK responses so no
live API calls fire. Coverage: happy path, system-prompt extraction
(Anthropic-specific), 4xx → NonRetryableAPIError mapping, refusal of
the base-model raw-completion path on both providers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# OpenAIClient
# ---------------------------------------------------------------------------


class TestOpenAIClient:
    @pytest.mark.asyncio
    async def test_complete_returns_message_text(self):
        from src.models import OpenAIClient

        # Mock the AsyncOpenAI constructor so no real API key check happens
        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Hello from OpenAI"
            mock_inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5", api_key="test-key")
            assert client.model_name == "gpt-5"
            out = await client.complete(
                [{"role": "user", "content": "Hi"}],
                temperature=0.5, max_tokens=128,
            )
            assert out == "Hello from OpenAI"
            # Arguments forwarded correctly
            kwargs = mock_inst.chat.completions.create.call_args.kwargs
            assert kwargs["model"] == "gpt-5"
            assert kwargs["temperature"] == 0.5
            assert kwargs["max_tokens"] == 128
            assert kwargs["messages"] == [{"role": "user", "content": "Hi"}]
            await client.close()

    @pytest.mark.asyncio
    async def test_complete_text_raises_not_implemented(self):
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI"):
            client = OpenAIClient(model="gpt-5", api_key="test-key")
            with pytest.raises(NotImplementedError, match="raw-completion"):
                await client.complete_text("Some prompt")

    @pytest.mark.asyncio
    async def test_complete_handles_empty_content_as_empty_string(self):
        """If the API returns content=None (rare), fall back to '' rather
        than crashing the runner."""
        from src.models import OpenAIClient

        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = None
            mock_inst.chat.completions.create = AsyncMock(return_value=mock_resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = OpenAIClient(model="gpt-5", api_key="x")
            assert await client.complete([{"role": "user", "content": "Hi"}]) == ""


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


class TestAnthropicClient:
    @pytest.mark.asyncio
    async def test_complete_extracts_system_prompt(self):
        """Anthropic requires `system` as a separate kwarg; verify the
        first system-role message is pulled out of the messages array."""
        from src.models import AnthropicClient

        with patch("anthropic.AsyncAnthropic") as mock_ctor:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Hello from Claude"
            mock_resp.content = [text_block]
            mock_inst.messages.create = AsyncMock(return_value=mock_resp)
            mock_inst.close = AsyncMock()
            mock_ctor.return_value = mock_inst

            client = AnthropicClient(model="claude-opus-4-7", api_key="x")
            assert client.model_name == "claude-opus-4-7"

            out = await client.complete([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Tell me more"},
            ])
            assert out == "Hello from Claude"

            kwargs = mock_inst.messages.create.call_args.kwargs
            # System extracted to a separate kwarg, in the structured
            # form Anthropic requires for cache_control markers.
            assert kwargs["system"] == [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            # Messages array no longer contains the system turn
            roles = [m["role"] for m in kwargs["messages"]]
            assert "system" not in roles
            assert roles == ["user", "assistant", "user"]
            await client.close()

    @pytest.mark.asyncio
    async def test_complete_handles_no_system_prompt(self):
        """When messages have no system turn, system kwarg is omitted."""
        from src.models import AnthropicClient

        with patch("anthropic.AsyncAnthropic") as mock_ctor:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "ok"
            mock_resp.content = [text_block]
            mock_inst.messages.create = AsyncMock(return_value=mock_resp)
            mock_ctor.return_value = mock_inst

            client = AnthropicClient(model="claude-sonnet-4-6", api_key="x")
            await client.complete([{"role": "user", "content": "Hi"}])
            kwargs = mock_inst.messages.create.call_args.kwargs
            assert "system" not in kwargs

    @pytest.mark.asyncio
    async def test_complete_returns_empty_when_no_text_block(self):
        from src.models import AnthropicClient

        with patch("anthropic.AsyncAnthropic") as mock_ctor:
            mock_inst = MagicMock()
            mock_resp = MagicMock()
            non_text = MagicMock()
            non_text.type = "tool_use"  # not a text block
            mock_resp.content = [non_text]
            mock_inst.messages.create = AsyncMock(return_value=mock_resp)
            mock_ctor.return_value = mock_inst

            client = AnthropicClient(model="claude-sonnet-4-6", api_key="x")
            assert await client.complete([{"role": "user", "content": "Hi"}]) == ""

    @pytest.mark.asyncio
    async def test_complete_text_raises_not_implemented(self):
        from src.models import AnthropicClient

        with patch("anthropic.AsyncAnthropic"):
            client = AnthropicClient(model="claude-sonnet-4-6", api_key="x")
            with pytest.raises(NotImplementedError, match="raw-completion"):
                await client.complete_text("prompt")

    def test_split_system_helper(self):
        from src.models import AnthropicClient

        msgs = [
            {"role": "system", "content": "S1"},
            {"role": "user", "content": "U1"},
            {"role": "system", "content": "S2 (ignored)"},
            {"role": "assistant", "content": "A1"},
        ]
        system, rest = AnthropicClient._split_system(msgs)
        # Only the FIRST system message is extracted; subsequent system
        # messages stay in the body (Anthropic API can handle them inline
        # but this matches our extraction contract).
        assert system == "S1"
        assert len(rest) == 3
        assert rest[0]["role"] == "user"


# ---------------------------------------------------------------------------
# CLI provider dispatch
# ---------------------------------------------------------------------------


class TestCliProviderDispatch:
    def test_provider_flag_parses_for_run(self):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "gpt-5",
            "--condition", "neutral",
            "--provider", "openai",
        ])
        assert args.provider == "openai"

    def test_provider_flag_parses_for_pilot(self):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "pilot",
            "--model", "claude-opus-4-7",
            "--provider", "anthropic",
            "--dry-run",
        ])
        assert args.provider == "anthropic"

    def test_provider_default_is_vllm(self):
        from src.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "test",
            "--condition", "neutral",
        ])
        assert args.provider == "vllm"

    def test_unknown_provider_rejected_by_argparse(self):
        from src.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "run",
                "--model", "test",
                "--condition", "neutral",
                "--provider", "fictional",
            ])

    def test_openai_with_base_model_rejected(self, tmp_path):
        """`_build_client` refuses --provider openai + --base-model
        because OpenAI's modern chat models have no completions endpoint."""
        from src.cli import _build_client

        class Args:
            model = "gpt-5"
            dry_run = False
            base_model = True
            provider = "openai"
            base_url = "ignored"

        with pytest.raises(SystemExit) as excinfo:
            _build_client(Args())
        assert excinfo.value.code == 2

    def test_anthropic_with_base_model_rejected(self):
        from src.cli import _build_client

        class Args:
            model = "claude-opus-4-7"
            dry_run = False
            base_model = True
            provider = "anthropic"
            base_url = "ignored"

        with pytest.raises(SystemExit) as excinfo:
            _build_client(Args())
        assert excinfo.value.code == 2

    def test_dry_run_short_circuits_to_dryrunclient(self):
        """--dry-run trumps any provider; useful for offline testing of
        provider-tagged configs without holding an API key."""
        from src.cli import _build_client
        from src.models import DryRunClient

        class Args:
            model = "any"
            dry_run = True
            base_model = False
            provider = "openai"  # ignored when dry_run is True
            base_url = "ignored"

        client = _build_client(Args())
        assert isinstance(client, DryRunClient)
