"""Model clients for the Affect Battery harness."""

import asyncio
import json
import logging
import re
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass

import aiohttp

log = logging.getLogger(__name__)


# HTTP status codes that indicate a client-side configuration error or
# a resource that will never appear: retrying wastes calls and delays the
# operator's feedback loop. 408 (request timeout) and 429 (rate limit) are
# excluded because they are transient.
NON_RETRYABLE_STATUSES = frozenset({400, 401, 403, 404, 422})


@dataclass
class UsageRecord:
    """One API call's billable token counts.

    `reasoning_tokens` is None for non-reasoning models (gpt-4o, etc.)
    that don't itemize hidden reasoning. For reasoning models (gpt-5.x,
    o-series), it's the count from `usage.completion_tokens_details.reasoning_tokens`,
    billed at the same rate as visible completion tokens.
    """
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None = None


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
        # Per-call usage records appended after each successful response.
        # See `UsageRecord` and `usage_summary()`.
        self.usage_log: list[UsageRecord] = []

    @property
    def model_name(self) -> str:
        return self._model

    def usage_summary(
        self,
        *,
        input_usd_per_million: float | None = None,
        output_usd_per_million: float | None = None,
    ) -> dict:
        """Aggregate per-call usage records.

        Output cost (when pricing is supplied) bills `completion_tokens`
        at the output rate. `reasoning_tokens` from
        `completion_tokens_details` is a breakdown of `completion_tokens`,
        not an additional billable line item; it is reported separately
        for visibility but is already included in the completion total.
        """
        n_calls = len(self.usage_log)
        prompt = sum(r.prompt_tokens for r in self.usage_log)
        completion = sum(r.completion_tokens for r in self.usage_log)
        reasoning = sum(
            (r.reasoning_tokens or 0) for r in self.usage_log
        )
        out: dict = {
            "model": self._model,
            "n_calls": n_calls,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
        }
        if input_usd_per_million is not None and output_usd_per_million is not None:
            out["estimated_usd"] = (
                prompt * input_usd_per_million / 1_000_000
                + completion * output_usd_per_million / 1_000_000
            )
        return out

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
        self._record_usage(resp)
        return resp.choices[0].message.content or ""

    def _record_usage(self, resp) -> None:
        """Pull token counts off the response and append a UsageRecord.

        Tolerant of usage missing entirely (some local-proxy SDKs strip
        it) — silently skips rather than crashing.
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0)
        completion = getattr(usage, "completion_tokens", 0)
        reasoning: int | None = None
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", None)
        self.usage_log.append(UsageRecord(
            model=self._model,
            prompt_tokens=int(prompt),
            completion_tokens=int(completion),
            reasoning_tokens=int(reasoning) if reasoning is not None else None,
        ))

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


# -----------------------------------------------------------------------------
# Claude Code CLI backend
# -----------------------------------------------------------------------------
# Subprocess-backed chat-completion adapter for the `claude` CLI.


class ClaudeCodeError(Exception):
    """Base class for ClaudeCodeClient failures."""


class ClaudeCodeNotAvailableError(ClaudeCodeError):
    """Raised when the `claude` binary is not on PATH."""


class ClaudeCodeAuthError(ClaudeCodeError):
    """Raised when `claude auth status` exits non-zero."""


class ClaudeCodeTimeoutError(ClaudeCodeError):
    """Raised when a `complete()` call exceeds the configured timeout."""


class ClaudeCodeProtocolError(ClaudeCodeError):
    """Raised when the CLI's stream-JSON output is missing the result event."""


class ClaudeCodeUnsupportedParameterError(ClaudeCodeError):
    """Raised in strict-params mode when temperature or max_tokens is non-default."""


_CLAUDE_CODE_DEFAULT_TEMPERATURE = 1.0

# Auth source values returned by _detect_claude_auth_source. Closed set;
# downstream code reads these as identifier strings (recorded in the
# pilot manifest as `inference_auth_source`).
AUTH_SUBSCRIPTION = "subscription"
AUTH_API = "api"
AUTH_UNKNOWN = "unknown"


def _format_messages_as_stream_json(messages: list[dict]) -> bytes:
    """Serialize messages as newline-delimited stream-JSON for `claude --input-format stream-json`.

    Each message becomes one line: {"type": <role>, "message": {"role":
    <role>, "content": [{"type": "text", "text": <content>}]}}. The content
    is wrapped as a content-block array because the CLI does a tool_use
    check via .some() that requires the array shape.
    """
    lines = []
    for m in messages:
        content_blocks = [{"type": "text", "text": m["content"]}]
        lines.append(json.dumps({
            "type": m["role"],
            "message": {"role": m["role"], "content": content_blocks},
        }))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_stream_json_response(stdout: str) -> tuple[str | None, dict]:
    """Parse newline-delimited stream-JSON output from `claude --output-format stream-json`.

    Returns (result_text, metadata) where result_text is the assistant's
    final message and metadata is a dict of {total_cost_usd, duration_ms,
    num_turns, session_id} extracted from the terminal `result` event.
    """
    result_text: str | None = None
    metadata: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "result" and event.get("subtype") == "success":
            result_text = event.get("result")
            for k in ("total_cost_usd", "duration_ms", "num_turns", "session_id"):
                if k in event:
                    metadata[k] = event[k]
            break
    return result_text, metadata


# Anchor on the "Login method:" status line so generic tokens like
# "API" inside surrounding banner text cannot misclassify the source.
_AUTH_SUBSCRIPTION_RE = re.compile(
    r"login method[^\n]*\b(claude max|claude pro|subscription|claude account)\b",
    re.IGNORECASE,
)
_AUTH_API_RE = re.compile(
    r"login method[^\n]*\b(console|api key|anthropic console)\b",
    re.IGNORECASE,
)


def _detect_claude_auth_source() -> tuple[str, str]:
    """Probe `claude auth status --text` and classify.

    Returns (auth_source, raw_text). auth_source is one of
    AUTH_SUBSCRIPTION, AUTH_API, or AUTH_UNKNOWN. Raises
    ClaudeCodeNotAvailableError if the binary is missing,
    ClaudeCodeAuthError if the probe exits non-zero or its output cannot
    be decoded.
    """
    try:
        result = subprocess.run(
            ["claude", "auth", "status", "--text"],
            capture_output=True, timeout=10,
        )
    except FileNotFoundError as e:
        raise ClaudeCodeNotAvailableError(
            "`claude` binary not on PATH. Install with: "
            "curl -fsSL https://claude.ai/install.sh | bash"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise ClaudeCodeAuthError(
            f"`claude auth status` timed out after 10s: {e}"
        ) from e
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise ClaudeCodeAuthError(
            f"`claude auth status` exited {result.returncode}: "
            f"{(stdout + stderr).strip()[:300]}"
        )
    text = stdout + stderr
    if _AUTH_SUBSCRIPTION_RE.search(text):
        return AUTH_SUBSCRIPTION, text
    if _AUTH_API_RE.search(text):
        return AUTH_API, text
    return AUTH_UNKNOWN, text


class ClaudeCodeClient(ModelClient):
    """Chat-completion adapter that delegates to the `claude` CLI subprocess.

    The CLI does not accept --temperature or --max-tokens flags, so this
    client raises ClaudeCodeUnsupportedParameterError when callers pass
    non-default values (default temperature: 1.0; default max_tokens:
    None). Set strict_params=False to accept non-defaults with a manifest
    flag instead of raising.

    Auth source is probed once at construction. Subscription auth omits
    --bare; API auth includes --bare; unknown auth omits --bare (fail-safe
    for the more common Claude Max/Pro path) and emits a one-time stderr
    warning naming the unrecognized text.
    """

    def __init__(
        self,
        model: str,
        *,
        strict_params: bool = True,
        timeout: float = 120.0,
    ):
        self._model = model
        self._strict_params = strict_params
        self._timeout = timeout
        self.total_cost_usd: float = 0.0
        # (temperature, max_tokens) for each lenient-mode call whose
        # parameters deviated from the CLI's defaults.
        self.unhonored_calls: list[tuple[float, int | None]] = []

        self.auth_source, raw_text = _detect_claude_auth_source()

        cli_args = [
            "-p",
            "--tools", "",
            "--max-turns", "1",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--no-session-persistence",
        ]
        if self.auth_source == AUTH_API:
            cli_args.append("--bare")
        elif self.auth_source == AUTH_UNKNOWN:
            print(
                f"[ClaudeCodeClient] WARNING: unrecognized `claude auth status` "
                f"text; --bare omitted to preserve subscription auth in case "
                f"the user is on Claude Max/Pro. Auth probe text: "
                f"{raw_text.strip()[:200]!r}",
                file=sys.stderr,
            )
        self._argv = ["claude", *cli_args, "--model", self._model]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def params_unhonored(self) -> bool:
        """True if any call in this client's lifetime ran under
        strict_params=False with a non-default temperature or max_tokens."""
        return bool(self.unhonored_calls)

    def manifest_metadata(self) -> dict:
        """Inference-backend metadata for the pilot manifest writer.

        Returns the auth source, accumulated cost, and (when present)
        the list of unhonored parameter deviations. Empty dict if no
        call has been made yet.
        """
        meta: dict = {
            "inference_auth_source": self.auth_source,
            "inference_total_cost_usd": round(float(self.total_cost_usd), 6),
        }
        if self.unhonored_calls:
            meta["inference_unhonored_calls"] = [
                {"temperature": t, "max_tokens": m}
                for t, m in self.unhonored_calls
            ]
        return meta

    async def complete(
        self,
        messages: list[dict],
        temperature: float = _CLAUDE_CODE_DEFAULT_TEMPERATURE,
        max_tokens: int | None = None,
    ) -> str:
        is_default = (
            temperature == _CLAUDE_CODE_DEFAULT_TEMPERATURE
            and max_tokens is None
        )
        if not is_default:
            if self._strict_params:
                offending = (
                    f"temperature={temperature}"
                    if temperature != _CLAUDE_CODE_DEFAULT_TEMPERATURE
                    else f"max_tokens={max_tokens}"
                )
                raise ClaudeCodeUnsupportedParameterError(
                    f"ClaudeCodeClient: {offending} is not honored by the "
                    f"`claude` CLI (no --temperature / --max-tokens flag). "
                    f"Set strict_params=False to accept this deviation; the "
                    f"manifest will record the (temperature, max_tokens) "
                    f"under inference_unhonored_calls."
                )
            log.warning(
                "ClaudeCodeClient: temperature=%s, max_tokens=%s passed but "
                "not honored by the `claude` CLI. Manifest will record this "
                "deviation under inference_unhonored_calls.",
                temperature, max_tokens,
            )
            self.unhonored_calls.append((temperature, max_tokens))

        stdin_bytes = _format_messages_as_stream_json(messages)

        proc = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(
                proc.communicate(stdin_bytes),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            try:
                # Drain pipes so the child doesn't deadlock on a full
                # stdout buffer; communicate() also awaits the process
                # exit. wait() alone can hang on >64KB stdout.
                await asyncio.wait_for(proc.communicate(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            raise ClaudeCodeTimeoutError(
                f"ClaudeCodeClient: `claude` did not return within "
                f"{self._timeout}s; subprocess killed."
            ) from e

        result_text, metadata = _parse_stream_json_response(
            stdout.decode("utf-8", errors="replace")
        )
        if result_text is None:
            raise ClaudeCodeProtocolError(
                f"ClaudeCodeClient: no `result` event in stream-JSON output. "
                f"stdout snippet: {stdout.decode('utf-8', errors='replace')[:300]!r}"
            )

        cost = metadata.get("total_cost_usd")
        if cost is not None:
            try:
                self.total_cost_usd += float(cost)
            except (TypeError, ValueError):
                pass

        return result_text

    async def complete_text(self, *_a, **_kw) -> str:
        raise NotImplementedError(
            "ClaudeCodeClient does not support base-model completions; "
            "use --provider vllm or anthropic for that path."
        )
