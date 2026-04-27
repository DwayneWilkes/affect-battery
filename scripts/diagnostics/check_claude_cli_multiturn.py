"""Verify Claude Code CLI accepts stream-json multi-turn input the way the
affect-battery harness needs it to.

The affect-battery harness's ModelClient.complete() is shaped like a
stateless chat-completion API: caller passes a list of messages, callee
returns the next assistant message. The CLI's ``-p --input-format
stream-json`` mode is the closest match. This diagnostic verifies the
CLI invocation actually behaves as a chat-completion proxy.

Tests:
  1. stream-json input is accepted and parsed
  2. The response field contains the model's next assistant message
  3. Multi-turn context is preserved (model recalls prior turns)
  4. ``--no-session-persistence`` flag is accepted
  5. Generation parameter flags (--temperature, --max-tokens) presence
     is reported (informational; missing flags mean defaults apply)

Each test is short (<30s) and uses a deterministic prompt designed to
discriminate pass/fail clearly.

Usage:
    python scripts/diagnostics/check_claude_cli_multiturn.py
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys


PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


async def run_claude(
    stdin_input: str, args: list[str], timeout: float = 60
) -> tuple[int, str, str]:
    """Run claude with the given args and stdin. Returns (rc, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        "claude", *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_input.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"timed out after {timeout}s"
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def format_messages_as_stream_json(messages: list[dict]) -> str:
    """Serialize a [{role, content}] list as newline-delimited stream-json events.

    The exact stream-json schema the CLI expects isn't fully documented; this
    diagnostic uses the shape inferred from the docs and reports back if it
    isn't accepted. If this format fails, the alternative shapes to try are:
      - {"role": "user", "content": "..."}                  (raw chat-message)
      - {"type":"message","message":{"role":"...",...}}     (wrapped)
      - {"type":"user","content":"..."}                     (flat)
    """
    lines = []
    for m in messages:
        lines.append(json.dumps({
            "type": m["role"],
            "message": {"role": m["role"], "content": m["content"]},
        }))
    return "\n".join(lines) + "\n"


CLI_BASE_ARGS = [
    "-p",
    "--bare",
    "--tools", "",
    "--max-turns", "1",
    "--input-format", "stream-json",
    "--output-format", "json",
    "--no-session-persistence",
]


async def test_basic_stream_json() -> tuple[str, str]:
    """Send a single user message via stream-json; verify a response comes back."""
    stdin = format_messages_as_stream_json([
        {"role": "user", "content": "Reply with exactly the two characters 'OK' and nothing else."},
    ])
    rc, stdout, stderr = await run_claude(stdin, CLI_BASE_ARGS)
    if rc != 0:
        return FAIL, f"non-zero exit ({rc}): {stderr.strip()[:300]}"
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return FAIL, f"output is not valid JSON: {stdout[:200]!r}"
    result = data.get("result", "")
    if not isinstance(result, str) or not result:
        return FAIL, f"missing or empty .result field: {data}"
    if "OK" in result.upper():
        return PASS, f"stream-json input + JSON output works (result snippet: {result[:80]!r})"
    return WARN, (
        f"stream-json parsed but unexpected response: {result[:120]!r}. "
        "May indicate the model is wrapping the answer in commentary; consider --append-system-prompt."
    )


async def test_multiturn_context() -> tuple[str, str]:
    """Verify the model receives prior turns as real conversation context.

    If the CLI flattens the input into a single prompt (transcript-imitation
    failure mode), the model will see a transcript with the secret word but
    treat the entire input as a flat document. If the CLI threads the messages
    as real multi-turn input, the model is conditioned on the secret word
    being established in turn 1 and recalls it on cue.

    A weaker contrast than the ideal but still informative.
    """
    stdin = format_messages_as_stream_json([
        {
            "role": "user",
            "content": (
                "Remember the secret word: ORANGE. I will ask later. "
                "For this turn, reply with only 'noted'."
            ),
        },
        {"role": "assistant", "content": "noted"},
        {
            "role": "user",
            "content": "What is two plus two? Reply with only the digit.",
        },
        {"role": "assistant", "content": "4"},
        {
            "role": "user",
            "content": (
                "Now, what was the secret word I asked you to remember? "
                "Reply with only the word in uppercase."
            ),
        },
    ])
    rc, stdout, stderr = await run_claude(stdin, CLI_BASE_ARGS)
    if rc != 0:
        return FAIL, f"non-zero exit ({rc}): {stderr.strip()[:300]}"
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return FAIL, f"output is not valid JSON: {stdout[:200]!r}"
    result = data.get("result", "").upper()
    if "ORANGE" in result:
        return PASS, "multi-turn context preserved (model recalled secret from turn 1)"
    return FAIL, (
        f"multi-turn context NOT preserved (got: {result[:120]!r}). "
        "The CLI may not be threading stream-json events as a real multi-turn "
        "conversation. Try alternative stream-json schemas before building."
    )


async def test_help_for_generation_flags() -> tuple[str | None, str]:
    """Scan ``claude --help`` for temperature and max-tokens flags.

    Returns (None, multi_line_message) so the caller can print findings as a
    block rather than a single status line.
    """
    rc, stdout, stderr = await run_claude("", ["--help"], timeout=15)
    help_text = (stdout + "\n" + stderr).lower()

    findings = []
    if "--temperature" in help_text:
        findings.append(f"  {PASS} --temperature flag found in --help")
    else:
        findings.append(
            f"  {WARN} --temperature flag NOT in --help. "
            "If absent, CLI defaults will apply; experimental temperature settings "
            "(typically 0.7) cannot be honored. Methods write-up should note this."
        )
    if "--max-tokens" in help_text or "--max_tokens" in help_text:
        findings.append(f"  {PASS} --max-tokens flag found in --help")
    else:
        findings.append(
            f"  {WARN} --max-tokens flag NOT in --help. "
            "If absent, defaults apply (likely 4096+). exp3b's 512-token cap and "
            "exp3c's 256-token cap will not be honored."
        )
    return None, "\n".join(findings)


async def test_short_circuit_check() -> tuple[str, str]:
    """Test that --max-turns 1 actually short-circuits the agent loop.

    If --max-turns 1 doesn't behave as a hard cap, the model may produce
    multiple agentic turns (tool calls, internal reasoning steps) and the
    'response' we get back may not be the model's first chat-completion
    output. We verify by asking a question whose answer is a single word
    and checking the response is short.
    """
    stdin = format_messages_as_stream_json([
        {
            "role": "user",
            "content": "What color is the sky on a clear day? Answer with one word.",
        },
    ])
    rc, stdout, stderr = await run_claude(stdin, CLI_BASE_ARGS)
    if rc != 0:
        return FAIL, f"non-zero exit ({rc}): {stderr.strip()[:200]}"
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return FAIL, f"output not JSON: {stdout[:200]!r}"
    result = data.get("result", "")
    word_count = len(result.split())
    if word_count <= 5:
        return PASS, f"--max-turns 1 short-circuits as expected (response: {result[:60]!r})"
    return WARN, (
        f"response has {word_count} words; expected <=5 for a one-word answer. "
        "The agent loop may still be running multiple turns; check --max-turns behavior."
    )


async def main() -> int:
    if not shutil.which("claude"):
        print(f"{FAIL} claude CLI not on PATH")
        print()
        print("Install with: curl -fsSL https://claude.ai/install.sh | bash")
        return 2

    print("==================================================")
    print("  Claude Code CLI multi-turn diagnostic")
    print("==================================================")
    print()
    print("Verifies the CLI can serve as a stateless chat-completion proxy")
    print("for the affect-battery harness. Each test is <30s.")
    print()

    tests = [
        ("Basic stream-json input + JSON output", test_basic_stream_json),
        ("--max-turns 1 short-circuits agent loop", test_short_circuit_check),
        ("Multi-turn context preservation", test_multiturn_context),
        ("Generation parameter flag availability", test_help_for_generation_flags),
    ]

    results = []
    for name, test in tests:
        print(f"-- {name}")
        try:
            result = await test()
            if result[0] is None:
                print(result[1])
            else:
                status, msg = result
                print(f"  {status} {msg}")
            results.append((name, result))
        except Exception as e:
            print(f"  {FAIL} unexpected exception: {type(e).__name__}: {e}")
            results.append((name, (FAIL, str(e))))
        print()

    failures = sum(1 for _, r in results if r[0] == FAIL)
    warnings = sum(1 for _, r in results if r[0] == WARN)

    print("==================================================")
    print(f"  Summary: {failures} failures, {warnings} warnings")
    print("==================================================")
    if failures > 0:
        print()
        print("One or more critical checks failed. The CLI may not be ready")
        print("for the affect-battery harness in its current form. Common")
        print("next steps:")
        print("  - Try alternative stream-json schemas (see")
        print("    format_messages_as_stream_json in this file)")
        print("  - Check `claude --help` for an `--input-format` schema page")
        print("  - Open an issue at https://github.com/anthropics/claude-code-issues")
        return 1

    print()
    print("All critical checks passed. The CLI is ready for the next step:")
    print("  - For subscription-billed scripted use: run `claude setup-token`")
    print("  - To build the integration: implement ClaudeCliClient in src/models.py")
    if warnings > 0:
        print()
        print(f"{warnings} warning(s) above. Review before relying on the CLI for")
        print("publication-grade results; warnings about temperature / max-tokens")
        print("indicate experimental parameter settings will not be honored.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
