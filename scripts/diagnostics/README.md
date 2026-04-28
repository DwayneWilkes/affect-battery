# Claude Code CLI as an inference backend

This directory contains two diagnostic scripts and the operator-facing
documentation for routing affect-battery experiments through the
`claude` CLI via `--provider claude_code`. The integration adapter
itself lives at `src/models.py::ClaudeCodeClient`; this README covers
the diagnostics, the operator workflow, and the methodological caveats
operators need to know before relying on the CLI path.

## TL;DR

```bash
# 1. Verify the CLI can serve as a chat-completion backend.
bash scripts/diagnostics/check_claude_cli.sh
python scripts/diagnostics/check_claude_cli_multiturn.py

# 2. Run any experiment with --provider claude_code.
uv run --active python -m src.cli run --provider claude_code --model sonnet ...
```

See *Using Claude Code as the inference backend* below for the full
workflow, manifest fields, and error semantics.

## Why two scripts

The shape of the affect-battery harness imposes two requirements on any
chat-completion backend:

1. **Stateless multi-turn semantics**: each `complete()` call passes a full
   message history and receives the next assistant message. No server-side
   conversation state is required.
2. **Subscription-billed scripted use**: the operator may not have an API
   key but does have a Claude Pro or Max subscription, and wants exploration
   runs to bill against that subscription.

`check_claude_cli.sh` verifies basic capability and authentication.
`check_claude_cli_multiturn.py` verifies the multi-turn semantics work in
the specific shape the harness needs.

## Run order

```bash
# 1. Basic capability and auth check.
bash scripts/diagnostics/check_claude_cli.sh

# 2. Multi-turn semantics check (only if step 1 passes).
python scripts/diagnostics/check_claude_cli_multiturn.py
```

## What each result means

### Bash script results

- **PASS on all checks**: the CLI is installed, authenticated, and the
  `-p` mode produces output. Proceed to the Python diagnostic.
- **FAIL on auth**: run `claude auth login` (subscription) or
  `claude auth login --console` (API). Re-run the script.
- **FAIL on `-p` invocation**: the CLI may be installed but unable to
  reach Anthropic. Check network and re-run.
- **WARN on subscription detection**: informational; the script can't
  always tell subscription from API auth. Run `claude auth status` to
  confirm.

### Python diagnostic results

- **PASS on all four tests**: the CLI is ready for the affect-battery
  harness. Run experiments via `--provider claude_code`.
- **FAIL on basic stream-json**: the CLI may use a different stream-json
  schema than this script expects. Inspect the script's
  `format_messages_as_stream_json` function and try alternative shapes.
- **FAIL on multi-turn context**: the CLI accepts stream-json input but
  doesn't thread it as a real multi-turn conversation. The harness cannot
  use this CLI as-is; consider the Anthropic API instead.
- **WARN on `--temperature` / `--max-tokens` absence**: the CLI defaults
  apply for those parameters; experimental settings (temperature=0.7,
  per-experiment max_tokens caps) cannot be honored via the CLI. Methods
  write-ups for any results obtained via this path should note this
  deviation from the registered design.

## Setting up subscription-billed scripted access

If both diagnostics pass and you want to run the harness against your
Claude subscription rather than an API key:

```bash
# Generate a long-lived OAuth token (prints to terminal; capture and store
# securely).
claude setup-token

# Export it for subsequent invocations.
export CLAUDE_CODE_OAUTH_TOKEN=<token>

# The harness's ClaudeCodeClient picks up this token automatically
# when no API key is set.
```

The token is tied to your subscription quota. Pro and Max plans have
different weekly quotas; large pilots may exceed even Max plan limits and
need to be split across weeks or moved to API billing.

## Methodological note

The CLI's `-p` mode runs an agent loop internally. We use `--tools ""` and
`--max-turns 1` to short-circuit this and use the CLI as a raw
chat-completion proxy. This is a valid use of the tool but slightly off
the documented happy path. If a future Claude Code release changes the
behavior of these flags, the harness's CLI provider may need updating;
re-run the diagnostic after upgrading the CLI to confirm.

For publication-grade results, prefer the Anthropic API path
(`--provider anthropic` with an API key) where temperature, max_tokens,
and other generation parameters are under explicit control.
`ClaudeCodeClient` runs in strict mode by default and raises on any
non-default `temperature` or `max_tokens`, so a runner that depends on
those settings cannot accidentally generate publication-grade data
through the CLI path. The CLI path suits exploration and feasibility
checks where the CLI's defaults are acceptable.

## CLI architecture findings (verified against Claude Code 2.1.120)

The diagnostics surface four constraints that any `ClaudeCodeClient`
implementation must respect:

1. **`--bare` skips OAuth and keychain reads.** Subscription auth (Claude
   Pro/Max via `claude auth login`) breaks under `--bare` because the
   token lives in the keychain. The integration must omit `--bare` when
   the auth source is subscription. `--bare` is fine when an API key is
   present in `ANTHROPIC_API_KEY`.
2. **`--input-format stream-json` requires `--output-format stream-json
   --verbose`.** The CLI rejects mixed input/output formats and silently
   drops events without `--verbose`.
3. **Assistant messages in stream-json input must use content-block
   arrays.** `{"role":"assistant","content":"..."}` fails with
   `H.message.content.some is not a function`. Use
   `{"role":"assistant","content":[{"type":"text","text":"..."}]}`. User
   messages tolerate string content but the array form is the safer
   choice; the diagnostic uses arrays uniformly.
4. **`--temperature` and `--max-tokens` are not exposed at the CLI
   surface.** Generation defaults apply for any harness call routed
   through the CLI provider. Methods documentation should record the
   defaults rather than the experimental settings.

## Using Claude Code as the inference backend

`--provider claude_code` routes any experiment through the
`ClaudeCodeClient` adapter in `src/models.py`. The adapter spawns a
`claude` subprocess per `complete()` call, formats messages as
stream-JSON, parses the result event, and conforms to the same
`ModelClient` ABC as `OpenAIClient` and `AnthropicClient`.

### Quick start

```bash
# 1. Prerequisites: both diagnostics pass; either subscription or API auth.
bash scripts/diagnostics/check_claude_cli.sh
python scripts/diagnostics/check_claude_cli_multiturn.py

# 2. Run an experiment.
uv run --active python -m src.cli run \
    --experiment exp1a \
    --provider claude_code \
    --model sonnet \
    --condition strong_positive \
    --num-runs 5 \
    --bank arithmetic_easy_v1 \
    --output-dir results/exp1a_smoke \
    --skip-prereg-gate --skip-power-gate
```

`--model` accepts CLI selectors (`sonnet`, `opus`, `haiku`) or full
model IDs. Per-call timeout defaults to 120s.

### Auth and `--bare`

Auth source is detected once at construction by parsing `claude auth
status --text`. The adapter then chooses whether to pass `--bare` to
the subprocess based on the source:

| Auth source detected | `--bare` | Behavior |
|---|---|---|
| `subscription` (Claude Pro/Max) | omitted | OAuth/keychain reads happen; subscription billing applies |
| `api` (Anthropic Console) | included | Skips OAuth/keychain reads; faster scripted use |
| `unknown` (unrecognized text) | omitted | Fail-safe: preserves subscription auth if the user is on Max/Pro. A one-time stderr warning prints the unrecognized text |

For subscription scripted use, run `claude setup-token` and export
`CLAUDE_CODE_OAUTH_TOKEN` so a fresh shell can authenticate without an
interactive login.

### Generation parameters

The CLI does not expose `--temperature` or `--max-tokens`. The adapter
enforces this at the boundary:

| Mode | Behavior on non-default `temperature` or `max_tokens` |
|---|---|
| `strict_params=True` (default) | Raises `ClaudeCodeUnsupportedParameterError` before subprocess launch. The error message names the offending parameter and value. |
| `strict_params=False` | Logs a WARNING with the offending value, appends `(temperature, max_tokens)` to `client.unhonored_calls`, and proceeds with the CLI's defaults. |

Pre-registered experiments that commit to specific generation
parameters cannot be replicated under this backend without a
pre-registration amendment that documents the deviation.

### Pilot manifest fields

When a pilot run uses `--provider claude_code`, the manifest writer
calls `client.manifest_metadata()` and merges its return value into
`manifest.yaml`. Fields:

| Field | Always present | Value |
|---|---|---|
| `inference_auth_source` | yes | `subscription`, `api`, or `unknown` |
| `inference_total_cost_usd` | yes | sum of per-call `total_cost_usd` from the CLI's result events; `0.0` under subscription auth (CLI reports `null`) |
| `inference_unhonored_calls` | only when at least one lenient-mode call deviated | list of `{temperature, max_tokens}` records, one per deviating call |

A run that produced no deviations omits `inference_unhonored_calls`
entirely; a run with one deviation among many compliant calls records
exactly that one. Operators reading the manifest see precisely which
parameter values were dropped, not a sticky boolean.

### Error handling

| Exception | Raised when |
|---|---|
| `ClaudeCodeNotAvailableError` | `claude` binary not on PATH at construction |
| `ClaudeCodeAuthError` | `claude auth status` exits non-zero or times out |
| `ClaudeCodeTimeoutError` | a `complete()` call exceeds the configured timeout (default 120s); subprocess is killed and pipes drained before raising |
| `ClaudeCodeProtocolError` | the CLI's stream-JSON output contains no `result` event with `subtype="success"` |
| `ClaudeCodeUnsupportedParameterError` | strict mode + non-default `temperature` or `max_tokens` |

All five subclass `ClaudeCodeError`. Catching the base class covers
every adapter-specific failure mode.
