# Claude Code CLI diagnostics

These scripts verify that the Claude Code CLI can serve as a chat-completion
backend for the affect-battery harness. Run them before attempting to use
`--provider claude-cli` against a real experiment, especially if you intend
to run on a Claude subscription rather than an Anthropic API key.

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
  harness. Build the `ClaudeCliClient` integration.
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

# The harness's ClaudeCliClient (once built) will pick up this token
# automatically when no API key is set.
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
and other generation parameters are under explicit control. The CLI path
is well-suited to exploration and feasibility checks where rough
parameter control is acceptable.

## CLI architecture findings (verified against Claude Code 2.1.120)

The diagnostics surface four constraints that any `ClaudeCliClient`
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
