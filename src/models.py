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
        runners (used by  base-model feasibility probe + Task
        3.3 base-model batch runs).
        """
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response
