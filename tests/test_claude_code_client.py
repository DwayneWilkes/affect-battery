"""ClaudeCodeClient: subprocess-backed chat-completion adapter for the
`claude` CLI.

The diagnostic at scripts/diagnostics/check_claude_cli_multiturn.py
verified the CLI's stream-json semantics, --max-turns 1 short-circuit,
and auth detection. This client lifts those helpers into the production
code path and conforms to the ModelClient ABC so any experiment runner
can use --provider claude_code without modification.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import dataclass

import pytest


@dataclass
class _CapturedSubprocessCall:
    argv: list[str]
    stdin: bytes
    proc: "_MockProcess | None" = None


class _MockProcess:
    """Stand-in for asyncio.subprocess.Process."""

    def __init__(self, stdout: bytes, stderr: bytes, returncode: int = 0, *, hang: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False

    async def communicate(self, stdin: bytes | None = None):
        if self._hang:
            await asyncio.sleep(60)
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True

    async def wait(self):
        return self.returncode


class _SubprocessRecorder:
    """Captures subprocess invocations and allows configuring canned responses."""

    def __init__(self):
        self.calls: list[_CapturedSubprocessCall] = []
        self._stdout = b""
        self._stderr = b""
        self._returncode = 0
        self._hang = False

    def set_response(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False):
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._hang = hang

    def __getitem__(self, idx):
        return self.calls[idx]

    def __len__(self):
        return len(self.calls)


@pytest.fixture
def mock_claude_subprocess(monkeypatch):
    """Patch asyncio.create_subprocess_exec to return a configurable mock."""
    recorder = _SubprocessRecorder()

    async def fake_create(*args, **kwargs):
        proc = _MockProcess(recorder._stdout, recorder._stderr, recorder._returncode, hang=recorder._hang)
        recorder.calls.append(_CapturedSubprocessCall(argv=list(args), stdin=b"", proc=proc))
        original_communicate = proc.communicate

        async def communicate_with_capture(stdin: bytes | None = None):
            recorder.calls[-1].stdin = stdin or b""
            return await original_communicate(stdin)

        proc.communicate = communicate_with_capture
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create)
    return recorder


@pytest.fixture
def mock_claude_auth(monkeypatch):
    """Patch subprocess.run for `claude auth status --text`.

    Default: subscription auth, claude on PATH. Override via setter
    `set_auth(text, returncode=0)` and `set_binary_present(present)`. When
    binary_present is False, the auth probe raises FileNotFoundError to
    mirror the real subprocess behavior with a missing binary.
    """
    state = {
        "auth_text": b"Login method: Claude Max account (user@example.com)",
        "auth_returncode": 0,
        "binary_present": True,
    }
    original_run = subprocess.run

    def fake_run(argv, **kwargs):
        if argv[:2] == ["claude", "auth"]:
            if not state["binary_present"]:
                raise FileNotFoundError(
                    "[Errno 2] No such file or directory: 'claude'"
                )

            class _R:
                stdout = state["auth_text"]
                stderr = b""
                returncode = state["auth_returncode"]
            return _R()
        return original_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    def set_auth(text: str, returncode: int = 0):
        state["auth_text"] = text.encode("utf-8") if isinstance(text, str) else text
        state["auth_returncode"] = returncode

    def set_binary_present(present: bool):
        state["binary_present"] = present

    state["set_auth"] = set_auth
    state["set_binary_present"] = set_binary_present
    return state


def _result_event(text: str = "OK", cost: float | None = 0.001) -> bytes:
    """Build a stream-json stdout buffer with one assistant + one result event."""
    events = [
        {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}},
        {"type": "result", "subtype": "success", "result": text, "total_cost_usd": cost, "duration_ms": 100, "num_turns": 1, "session_id": "s1"},
    ]
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")


# -----------------------------------------------------------------------------
# Task 1.1: module imports
# -----------------------------------------------------------------------------


def test_module_imports():
    from src.models import (
        ClaudeCodeClient,
        ClaudeCodeNotAvailableError,
        ClaudeCodeAuthError,
        ClaudeCodeTimeoutError,
        ClaudeCodeProtocolError,
        ClaudeCodeUnsupportedParameterError,
    )
    assert ClaudeCodeClient is not None
    for exc in (
        ClaudeCodeNotAvailableError, ClaudeCodeAuthError, ClaudeCodeTimeoutError,
        ClaudeCodeProtocolError, ClaudeCodeUnsupportedParameterError,
    ):
        assert issubclass(exc, Exception)


# -----------------------------------------------------------------------------
# Task 1.2: complete() returns assistant string
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_returns_assistant_string(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    mock_claude_subprocess.set_response(stdout=_result_event("OK", cost=0.001))
    client = ClaudeCodeClient(model="sonnet")
    result = await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    assert result == "OK"


# -----------------------------------------------------------------------------
# Task 1.3: stream-json input formatting
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_json_input_format(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    mock_claude_subprocess.set_response(stdout=_result_event())
    client = ClaudeCodeClient(model="sonnet")
    await client.complete(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
        temperature=1.0, max_tokens=None,
    )
    stdin_bytes = mock_claude_subprocess[-1].stdin
    lines = [json.loads(line) for line in stdin_bytes.decode().strip().split("\n")]
    assert len(lines) == 3
    assert lines[0]["type"] == "user"
    assert lines[0]["message"]["role"] == "user"
    assert lines[0]["message"]["content"] == [{"type": "text", "text": "a"}]
    assert lines[1]["type"] == "assistant"
    assert lines[1]["message"]["content"] == [{"type": "text", "text": "b"}]
    assert lines[2]["type"] == "user"
    assert lines[2]["message"]["content"] == [{"type": "text", "text": "c"}]


# -----------------------------------------------------------------------------
# Task 1.4: stream-json output parsing
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_parses_result_event(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    mock_claude_subprocess.set_response(stdout=_result_event("expected text"))
    client = ClaudeCodeClient(model="sonnet")
    result = await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    assert result == "expected text"


@pytest.mark.asyncio
async def test_missing_result_event_raises(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient, ClaudeCodeProtocolError
    # stdout has system + assistant events but no result event
    bad_stdout = (
        json.dumps({"type": "system", "subtype": "init"}) + "\n" +
        json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}]}}) + "\n"
    ).encode()
    mock_claude_subprocess.set_response(stdout=bad_stdout)
    client = ClaudeCodeClient(model="sonnet")
    with pytest.raises(ClaudeCodeProtocolError) as exc:
        await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    assert "system" in str(exc.value) or "assistant" in str(exc.value) or "result" in str(exc.value).lower()


# -----------------------------------------------------------------------------
# Task 1.5: cost accumulation
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_cost_usd_accumulates(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    client = ClaudeCodeClient(model="sonnet")
    mock_claude_subprocess.set_response(stdout=_result_event("a", cost=0.001))
    await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    mock_claude_subprocess.set_response(stdout=_result_event("b", cost=0.002))
    await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    assert abs(client.total_cost_usd - 0.003) < 1e-9


@pytest.mark.asyncio
async def test_null_cost_does_not_increment(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    client = ClaudeCodeClient(model="sonnet")
    mock_claude_subprocess.set_response(stdout=_result_event("a", cost=None))
    await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    assert client.total_cost_usd == 0.0


# -----------------------------------------------------------------------------
# Task 1.6: generation-parameter rejection
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_default_temperature_raises_in_strict_mode(mock_claude_auth):
    from src.models import ClaudeCodeClient, ClaudeCodeUnsupportedParameterError
    client = ClaudeCodeClient(model="sonnet")
    with pytest.raises(ClaudeCodeUnsupportedParameterError) as exc:
        await client.complete([{"role": "user", "content": "q"}], temperature=0.7, max_tokens=None)
    msg = str(exc.value)
    assert "temperature" in msg.lower()
    assert "0.7" in msg


@pytest.mark.asyncio
async def test_non_default_max_tokens_raises_in_strict_mode(mock_claude_auth):
    from src.models import ClaudeCodeClient, ClaudeCodeUnsupportedParameterError
    client = ClaudeCodeClient(model="sonnet")
    with pytest.raises(ClaudeCodeUnsupportedParameterError) as exc:
        await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=512)
    msg = str(exc.value)
    assert "max_tokens" in msg.lower()
    assert "512" in msg


@pytest.mark.asyncio
async def test_strict_params_false_records_unhonored_calls(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    client = ClaudeCodeClient(model="sonnet", strict_params=False)

    mock_claude_subprocess.set_response(stdout=_result_event())
    await client.complete([{"role": "user", "content": "q"}], temperature=0.7, max_tokens=512)
    mock_claude_subprocess.set_response(stdout=_result_event())
    await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    mock_claude_subprocess.set_response(stdout=_result_event())
    await client.complete([{"role": "user", "content": "q"}], temperature=0.5, max_tokens=None)

    # First and third calls deviated; second call used defaults.
    assert client.unhonored_calls == [(0.7, 512), (0.5, None)]
    assert client.params_unhonored is True
    meta = client.manifest_metadata()
    assert meta["inference_unhonored_calls"] == [
        {"temperature": 0.7, "max_tokens": 512},
        {"temperature": 0.5, "max_tokens": None},
    ]


@pytest.mark.asyncio
async def test_strict_params_false_compliant_calls_leave_no_record(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient
    mock_claude_subprocess.set_response(stdout=_result_event())
    client = ClaudeCodeClient(model="sonnet", strict_params=False)
    await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    assert client.unhonored_calls == []
    assert client.params_unhonored is False
    assert "inference_unhonored_calls" not in client.manifest_metadata()


# -----------------------------------------------------------------------------
# Task 1.7: auth source detection
# -----------------------------------------------------------------------------


def test_subscription_auth_detected(mock_claude_auth):
    from src.models import ClaudeCodeClient
    mock_claude_auth["set_auth"]("Login method: Claude Max account (user@example.com)")
    client = ClaudeCodeClient(model="sonnet")
    assert client.auth_source == "subscription"
    assert "--bare" not in client._argv


def test_api_auth_detected(mock_claude_auth):
    from src.models import ClaudeCodeClient
    mock_claude_auth["set_auth"]("Login method: Anthropic Console (API key)")
    client = ClaudeCodeClient(model="sonnet")
    assert client.auth_source == "api"
    assert "--bare" in client._argv


def test_subscription_with_api_word_in_banner_stays_subscription(mock_claude_auth):
    """A subscription banner that mentions "API access" must classify as subscription."""
    from src.models import ClaudeCodeClient
    mock_claude_auth["set_auth"](
        "API access included with your subscription.\n"
        "Login method: Claude Max account (user@example.com)"
    )
    client = ClaudeCodeClient(model="sonnet")
    assert client.auth_source == "subscription"
    assert "--bare" not in client._argv


def test_unknown_auth_omits_bare_and_warns(mock_claude_auth, mock_claude_subprocess, capsys):
    from src.models import ClaudeCodeClient
    mock_claude_auth["set_auth"]("some unrecognized auth response from a future CLI version")
    client = ClaudeCodeClient(model="sonnet")
    assert client.auth_source == "unknown"
    assert "--bare" not in client._argv
    err = capsys.readouterr().err.lower()
    assert "unrecognized" in err or "unknown" in err
    assert "--bare" in err


def test_missing_binary_raises(mock_claude_auth):
    from src.models import ClaudeCodeClient, ClaudeCodeNotAvailableError
    mock_claude_auth["set_binary_present"](False)
    with pytest.raises(ClaudeCodeNotAvailableError):
        ClaudeCodeClient(model="sonnet")


def test_auth_status_nonzero_raises(mock_claude_auth):
    from src.models import ClaudeCodeClient, ClaudeCodeAuthError
    mock_claude_auth["set_auth"]("Not logged in", returncode=1)
    with pytest.raises(ClaudeCodeAuthError):
        ClaudeCodeClient(model="sonnet")


# -----------------------------------------------------------------------------
# Task 1.8: timeout
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_kills_subprocess_and_raises(mock_claude_subprocess, mock_claude_auth):
    from src.models import ClaudeCodeClient, ClaudeCodeTimeoutError
    mock_claude_subprocess.set_response(hang=True)
    client = ClaudeCodeClient(model="sonnet", timeout=0.1)
    with pytest.raises(ClaudeCodeTimeoutError):
        await client.complete([{"role": "user", "content": "q"}], temperature=1.0, max_tokens=None)
    # The single recorded call must have had its subprocess killed; without
    # the kill the child process leaks and may deadlock on a full stdout buffer.
    assert len(mock_claude_subprocess) == 1
    proc = mock_claude_subprocess[0].proc
    assert proc.killed is True


# -----------------------------------------------------------------------------
# Model name property
# -----------------------------------------------------------------------------


def test_model_name_returns_configured_selector(mock_claude_auth):
    from src.models import ClaudeCodeClient
    client = ClaudeCodeClient(model="opus")
    assert client.model_name == "opus"
