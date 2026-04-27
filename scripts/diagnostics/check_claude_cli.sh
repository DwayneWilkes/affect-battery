#!/usr/bin/env bash
# Verify Claude Code CLI works for scripted, subscription-billed exploration.
#
# Reports pass/fail for each capability without modifying any state.
# Run this before building the affect-battery ClaudeCliClient integration.
#
# Usage:
#   bash scripts/diagnostics/check_claude_cli.sh
#
# Exit codes:
#   0 - all checks passed; CLI is ready for affect-battery use
#   1 - one or more checks failed; see output for details
#   2 - claude CLI not installed (fatal)

set -uo pipefail

PASS="[PASS]"
FAIL="[FAIL]"
WARN="[WARN]"
INFO="[INFO]"

results=()
fail_count=0
warn_count=0

report() {
    local status="$1"
    local message="$2"
    results+=("$status $message")
    if [[ "$status" == "$FAIL" ]]; then
        fail_count=$((fail_count + 1))
    fi
    if [[ "$status" == "$WARN" ]]; then
        warn_count=$((warn_count + 1))
    fi
}

# ---- Check 1: claude CLI installed ----
if ! command -v claude >/dev/null 2>&1; then
    echo "$FAIL claude CLI not found on PATH"
    echo
    echo "Install with one of:"
    echo "  curl -fsSL https://claude.ai/install.sh | bash       # macOS / Linux / WSL"
    echo "  brew install --cask claude-code                       # macOS / Linux (Homebrew)"
    echo "  winget install Anthropic.ClaudeCode                   # Windows"
    echo
    echo "Then re-run this script."
    exit 2
fi
version=$(claude --version 2>/dev/null | head -1 || echo "unknown")
report "$PASS" "claude CLI installed (version: $version)"

# ---- Check 2: authentication ----
# claude auth status exits 0 when logged in, 1 when not. We capture both
# stdout and exit code rather than grepping for English phrases (which
# break across versions).
auth_output=$(claude auth status --text 2>&1)
auth_rc=$?
auth_source=""
if [[ $auth_rc -ne 0 ]]; then
    report "$FAIL" "claude is not authenticated (auth status exit $auth_rc)"
    echo
    echo "Sign in with:"
    echo "  claude auth login              # subscription billing (Claude Pro/Max)"
    echo "  claude auth login --console    # API billing via Anthropic Console"
    echo
else
    report "$PASS" "claude is authenticated"
    # Detect auth source from the well-known 'Login method:' line:
    #   "Login method: Claude Max account"     -> subscription
    #   "Login method: Anthropic Console"      -> API key
    if echo "$auth_output" | grep -qiE "Login method:.*(Max|Pro|subscription|Claude account)"; then
        auth_source="subscription"
        report "$INFO" "Auth source: Claude subscription (Pro/Max)"
    elif echo "$auth_output" | grep -qiE "Login method:.*(Console|API)"; then
        auth_source="api"
        report "$INFO" "Auth source: Anthropic Console (API billing)"
    else
        report "$INFO" "Auth source: see 'claude auth status' for details"
    fi
fi

# ---- Check 3: basic --print invocation ----
# CRITICAL: --bare skips OAuth/keychain reads, so it requires an API key
# in $ANTHROPIC_API_KEY and will fail under subscription auth. We pick the
# invocation flags based on the detected auth source.
if [[ "$auth_source" == "subscription" ]]; then
    # Subscription path: do NOT pass --bare (it would skip OAuth).
    invocation_flags=(-p --max-turns 1 --tools "")
    invocation_label="claude -p (subscription path, no --bare)"
else
    # API path or unknown: --bare is fine and faster for scripted use.
    invocation_flags=(-p --bare --max-turns 1 --tools "")
    invocation_label="claude -p --bare (API path)"
fi
print_response=$(echo "Reply with only the digit 4 and nothing else." | \
    timeout 60 claude "${invocation_flags[@]}" 2>&1 || echo "")
if [[ -z "$print_response" ]]; then
    report "$FAIL" "$invocation_label produced no output"
elif echo "$print_response" | grep -qiE "not logged in|please run /login|unauthor"; then
    report "$FAIL" "$invocation_label returned an auth error: ${print_response:0:120}"
    echo
    echo "If your auth source is subscription, --bare cannot be used here"
    echo "(it skips OAuth reads). The diagnostic should have detected this;"
    echo "rerun 'claude auth status --text' to confirm auth source."
    echo
else
    snippet=$(echo "$print_response" | head -c 80 | tr '\n' ' ')
    report "$PASS" "$invocation_label works (response: $snippet)"
fi

# ---- Check 4: long-lived OAuth token availability ----
# We only check that the command exists; we don't actually run setup-token
# (which would print a token to the terminal and we don't want to leak it).
if claude setup-token --help >/dev/null 2>&1; then
    report "$PASS" "claude setup-token available (for CI / scripted subscription auth)"
else
    report "$WARN" "claude setup-token may not be available; check 'claude --help'"
fi

# ---- Check 5: model selection ----
# Use the same auth-source-aware invocation flags as Check 3.
model_response=$(echo "Reply with only the word ok and nothing else." | \
    timeout 60 claude "${invocation_flags[@]}" --model sonnet 2>&1 || echo "")
if [[ -z "$model_response" ]]; then
    report "$WARN" "--model sonnet returned empty response"
elif echo "$model_response" | grep -qiE "not logged in|please run /login|unauthor"; then
    report "$WARN" "--model sonnet returned an auth error: ${model_response:0:80}"
else
    report "$PASS" "--model sonnet works"
fi

# ---- Check 6: structured output ----
json_response=$(echo "Reply with only the digit 7." | \
    timeout 60 claude "${invocation_flags[@]}" --output-format json 2>&1 || echo "")
if [[ -n "$json_response" ]] && echo "$json_response" | grep -q '"result"'; then
    report "$PASS" "--output-format json works (response includes .result field)"
else
    snippet=$(echo "$json_response" | head -c 120 | tr '\n' ' ')
    report "$WARN" "--output-format json may not be working; got: $snippet"
fi

# ---- Print results ----
echo
echo "=================================================="
echo "  Claude Code CLI diagnostic results"
echo "=================================================="
for r in "${results[@]}"; do
    echo "$r"
done
echo "=================================================="
echo "Summary: $fail_count failures, $warn_count warnings"
echo

if [[ $fail_count -gt 0 ]]; then
    echo "One or more critical checks failed. The CLI is not ready for"
    echo "affect-battery use until these are resolved."
    exit 1
fi

echo "All critical checks passed. Run the multi-turn diagnostic next:"
echo "  python scripts/diagnostics/check_claude_cli_multiturn.py"
exit 0
