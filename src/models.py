"""Model clients for the Affect Battery harness."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)


# HTTP status codes that indicate a client-side configuration error or
# a resource that will never appear: retrying wastes calls and delays the
# operator's feedback loop. 408 (request timeout) and 429 (rate limit) are
# excluded because they are transient.
NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404, 422})


class NonRetryableAPIError(Exception):
    """Raised when the API returns a status that must halt the batch
    rather than be retried (auth, schema, missing resource)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ModelClient(ABC):
    """Abstract base class for model inference."""
    
    @property
    @abstractmethod
    def model_name(self) -> str: ...
    
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str: ...


class VLLMClient(ModelClient):
    """Client for vLLM-served models (OpenAI-compatible API)."""
    
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._session: aiohttp.ClientSession | None = None
    
    @property
    def model_name(self) -> str:
        return self._model
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        session = await self._get_session()
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        for attempt in range(3):
            try:
                async with session.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except aiohttp.ClientResponseError as e:
                if e.status in NON_RETRYABLE_STATUSES:
                    raise NonRetryableAPIError(
                        f"HTTP {e.status}: {e.message}", status_code=e.status,
                    ) from e
                wait = 2 ** attempt
                log.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
                await asyncio.sleep(wait)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = 2 ** attempt
                log.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
                await asyncio.sleep(wait)

        raise RuntimeError(f"Failed after 3 attempts for {self._model}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class VLLMCompletionClient(ModelClient):
    """Client for base models via vLLM completions API (not chat)."""
    
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._session: aiohttp.ClientSession | None = None
    
    @property
    def model_name(self) -> str:
        return self._model
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """For base models, messages is ignored. Use complete_text() instead."""
        raise NotImplementedError("Use complete_text() for base models")
    
    async def complete_text(
        self, prompt: str, temperature: float = 0.7,
        max_tokens: int = 1024, stop: list[str] | None = None,
    ) -> str:
        session = await self._get_session()
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        
        for attempt in range(3):
            try:
                async with session.post(
                    f"{self._base_url}/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["text"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = 2 ** attempt
                log.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s.")
                await asyncio.sleep(wait)
        
        raise RuntimeError(f"Failed after 3 attempts for {self._model}")
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class DryRunClient(ModelClient):
    """Returns canned responses for testing."""
    
    def __init__(self, responses: list[str] | None = None, model: str = "dry-run"):
        self._responses = responses or ["The answer is 42."]
        self._model = model
        self._index = 0
    
    @property
    def model_name(self) -> str:
        return self._model
    
    async def complete(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response

    async def complete_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
    ) -> str:
        """Base-model completion path (vLLM /v1/completions analog).

        Returns the same canned-response cycle as `complete()` so the
        DryRunClient transparently supports both chat and base-model
        runners (used by the base-model feasibility probe + base-model
        batch runs).
        """
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response


class OpenAIClient(ModelClient):
    """Client for OpenAI chat-completion models (gpt-4o, gpt-5, o1, etc.).

    Uses the official `openai` SDK with async transport. The API key is
    read from the `OPENAI_API_KEY` environment variable when not passed
    explicitly. Implements `complete()` for chat models; raises on
    `complete_text()` because newer OpenAI models do not expose the raw
    completions endpoint for the few-shot scaffold path used by base
    models.
    """

    def __init__(self, model: str, api_key: str | None = None):
        from openai import AsyncOpenAI  # type: ignore[import]

        self._model = model
        self._client = AsyncOpenAI(api_key=api_key) if api_key else AsyncOpenAI()

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """gpt-5.x and o-series (reasoning) models deprecated `max_tokens`
        in favor of `max_completion_tokens`. The API rejects the legacy
        kwarg on these models; we have to use the new one.

        Older chat models (gpt-4, gpt-4o, gpt-3.5) still accept
        `max_tokens` and not `max_completion_tokens`.
        """
        lo = model.lower()
        if lo.startswith("gpt-5"):
            return True
        # o-series reasoning models. Match strict prefix to avoid
        # false-positives like 'gpt-4o' (which contains 'o').
        for prefix in ("o1", "o3", "o4", "o5"):
            if lo == prefix or lo.startswith(f"{prefix}-"):
                return True
        return False

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        params: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        # Per-model-family max-tokens parameter naming. gpt-5.x and
        # o-series use max_completion_tokens; legacy models use
        # max_tokens. Sending the wrong one yields a 400.
        if self._uses_max_completion_tokens(self._model):
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        from openai import (  # type: ignore[import]
            APIStatusError, AuthenticationError, RateLimitError,
        )
        import asyncio as _asyncio
        # OpenAI 429 handling has two distinct branches:
        #   - rate_limit_exceeded: too many RPM/TPM. SDK already retries
        #     2x; if those failed, sustained-rate retry with longer
        #     backoff (5s → 15s → 45s) usually clears once the rolling
        #     window resets. After max_extra_retries we give up via
        #     NonRetryableAPIError so the circuit breaker can react.
        #   - insufficient_quota: account is out of credit. Not
        #     retryable; surface a billing-page hint.
        # Other 4xx (auth, bad input) are NonRetryableAPIError on first
        # hit. 5xx propagate so the SDK's transport-level retry kicks in.
        max_extra_retries = 3
        last_exception: Exception | None = None
        resp = None
        for attempt in range(max_extra_retries + 1):
            try:
                resp = await self._client.chat.completions.create(**params)
                break
            except RateLimitError as e:
                last_exception = e
                error_str = str(e).lower()
                if (
                    "insufficient_quota" in error_str
                    or "exceeded your current quota" in error_str
                ):
                    raise NonRetryableAPIError(
                        f"OpenAI account out of credit (insufficient_quota). "
                        f"Top up at platform.openai.com/settings/organization/billing "
                        f"before re-running. Original: {e}",
                        status_code=429,
                    ) from e
                if attempt >= max_extra_retries:
                    raise NonRetryableAPIError(
                        f"OpenAI rate-limit sustained after "
                        f"{max_extra_retries} client-side retries (SDK "
                        f"already retried 2x before this). Lower "
                        f"--rate-limit-rps or --max-concurrent. Original: {e}",
                        status_code=429,
                    ) from e
                # Honor Retry-After header when present; otherwise
                # exponential backoff (5s, 15s, 45s).
                retry_after_sec: float | None = None
                response = getattr(e, "response", None)
                if response is not None and hasattr(response, "headers"):
                    raw = response.headers.get("retry-after")
                    if raw is not None:
                        try:
                            retry_after_sec = float(raw)
                        except ValueError:
                            pass
                wait = retry_after_sec if retry_after_sec else (5 * (3 ** attempt))
                await _asyncio.sleep(wait)
            except AuthenticationError as e:
                raise NonRetryableAPIError(
                    f"OpenAI auth failure: {e}", status_code=401,
                ) from e
            except APIStatusError as e:
                if 400 <= getattr(e, "status_code", 500) < 500:
                    raise NonRetryableAPIError(
                        f"OpenAI 4xx: {e}", status_code=e.status_code,
                    ) from e
                raise
        if resp is None:
            # Defensive: shouldn't happen given the explicit raises above.
            raise NonRetryableAPIError(
                f"OpenAI request failed after {max_extra_retries} retries; "
                f"last error: {last_exception}",
                status_code=429,
            )
        return resp.choices[0].message.content or ""

    async def complete_text(self, *_a, **_kw) -> str:
        raise NotImplementedError(
            "OpenAI's modern chat models do not expose a raw-completion "
            "endpoint; use complete() with chat-format messages."
        )

    async def close(self) -> None:
        await self._client.close()


class AnthropicClient(ModelClient):
    """Client for Anthropic chat models (Claude Sonnet, Opus, Haiku).

    Anthropic's API separates the system prompt from the conversation
    `messages` array; we extract the first system-role message and pass
    the rest as `messages`. The API key is read from `ANTHROPIC_API_KEY`
    when not passed explicitly. Raises on `complete_text()` — Anthropic
    has no raw-completion endpoint.
    """

    def __init__(self, model: str, api_key: str | None = None):
        from anthropic import AsyncAnthropic  # type: ignore[import]

        self._model = model
        self._client = (
            AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()
        )

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """Pull the first system-role message out of `messages`; return
        (system, remaining_messages). When no system message is present,
        returns ("", messages)."""
        system = ""
        rest: list[dict] = []
        for m in messages:
            if m.get("role") == "system" and not system:
                system = m.get("content", "")
            else:
                rest.append({"role": m["role"], "content": m["content"]})
        return system, rest

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        system, chat_messages = self._split_system(messages)
        kwargs: dict = {
            "model": self._model,
            "messages": chat_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            # Anthropic prompt caching: mark the system prompt as
            # cacheable. Below the model's minimum-cacheable-prefix
            # threshold (4096 tokens for Haiku 4.5 / Opus 4.7+; 2048
            # for Sonnet 4.6) the marker is a no-op — the API
            # processes the request without caching, no error. Above
            # threshold it cuts cached-input cost to ~10% of base
            # input. Forward-compat: kicks in automatically if the
            # protocol's system prompt grows beyond the threshold.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as e:
            from anthropic import APIStatusError, AuthenticationError  # type: ignore[import]
            if isinstance(e, AuthenticationError):
                raise NonRetryableAPIError(
                    f"Anthropic auth failure: {e}", status_code=401,
                ) from e
            if isinstance(e, APIStatusError) and 400 <= getattr(e, "status_code", 500) < 500:
                raise NonRetryableAPIError(
                    f"Anthropic 4xx: {e}", status_code=e.status_code,
                ) from e
            raise
        # Anthropic returns a list of content blocks; the first text block
        # carries the response for our chat-completion use case.
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    async def complete_text(self, *_a, **_kw) -> str:
        raise NotImplementedError(
            "Anthropic does not expose a raw-completion endpoint; use "
            "complete() with chat-format messages."
        )

    async def close(self) -> None:
        await self._client.close()
