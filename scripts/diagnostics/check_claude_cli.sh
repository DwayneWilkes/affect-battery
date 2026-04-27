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
auth_output=$(claude auth status --text 2>/dev/null || echo "")
if [[ -z "$auth_output" ]]; then
    report "$FAIL" "claude auth status returned no output (not logged in?)"
    echo
    echo "Sign in with:"
    echo "  claude auth login              # subscription billing (Claude Pro/Max)"
    echo "  claude auth login --console    # API billing via Anthropic Console"
    echo
elif echo "$auth_output" | grep -qi "logged in\|authenticated\|signed in"; then
    report "$PASS" "claude is authenticated"
    if echo "$auth_output" | grep -qi "console\|api"; then
        report "$INFO" "Auth source appears to be Anthropic Console (API billing)"
    elif echo "$auth_output" | grep -qi "claude\|subscription\|pro\|max"; then
        report "$INFO" "Auth source appears to be Claude subscription"
    else
        report "$INFO" "Auth source: see 'claude auth status' for details"
    fi
else
    report "$FAIL" "claude appears unauthenticated"
    echo
    echo "$auth_output"
    echo
fi

# ---- Check 3: basic --print invocation ----
print_response=$(echo "Reply with only the digits 4 and nothing else." | \
    timeout 30 claude -p --bare --max-turns 1 --tools "" 2>/dev/null || echo "")
if [[ -z "$print_response" ]]; then
    report "$FAIL" "claude -p basic invocation produced no output"
else
    snippet=$(echo "$print_response" | head -c 80 | tr '\n' ' ')
    report "$PASS" "claude -p basic invocation works (response: $snippet)"
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
model_response=$(echo "Reply with only the word ok and nothing else." | \
    timeout 30 claude -p --bare --max-turns 1 --tools "" --model sonnet 2>/dev/null || echo "")
if [[ -n "$model_response" ]]; then
    report "$PASS" "--model sonnet works"
else
    report "$WARN" "--model sonnet returned empty response"
fi

# ---- Check 6: structured output ----
json_response=$(echo "Reply with only the digit 7." | \
    timeout 30 claude -p --bare --max-turns 1 --tools "" --output-format json 2>/dev/null || echo "")
if [[ -n "$json_response" ]] && echo "$json_response" | grep -q '"result"'; then
    report "$PASS" "--output-format json works (response includes .result field)"
else
    report "$WARN" "--output-format json may not be working; got: ${json_response:0:120}"
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
