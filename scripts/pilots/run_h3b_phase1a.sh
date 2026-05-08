#!/usr/bin/env bash
# Run the H3b Phase 1A 20-pass single-turn calibrated replication.
#
# Pre-registration: docs/preregistrations/h3b_2026-05-07.md
#
# Each of 20 passes writes to its own subdirectory under the output
# base, so parallel passes do not collide on filename. The pre-reg
# locks the (item_id, level) cell as the unit of analysis, not pass
# order, so corpora aggregated across `pass_*/level_M/` directories
# are equivalent to a sequential run.
#
# Usage:
#   bash scripts/pilots/run_h3b_phase1a.sh \
#     --prereg-commit DwayneWilkes/affect-battery@<squash-sha> \
#     [--output-base results/h3b_2026-05-07] \
#     [--max-parallel 20]
#
# Environment:
#   OPENAI_API_KEY must be set.

set -euo pipefail

OUTPUT_BASE="results/h3b_2026-05-07"
PREREG_COMMIT=""
MAX_PARALLEL=20
N_PASSES=20
SEED=42
BANK="configs/banks/h3b_calibrated_v1.yaml"
RUNNER_CONFIG="configs/exp3a_runner_h3b_2026-05-07.yaml"
DRY_RUN=0

usage() {
    cat <<'USAGE'
Run the H3b Phase 1A 20-pass single-turn calibrated replication.

Usage:
  bash scripts/pilots/run_h3b_phase1a.sh \
    --prereg-commit <owner/repo>@<sha> \
    [--output-base PATH] \
    [--max-parallel N] \
    [--n-passes N] \
    [--seed N]

Required:
  --prereg-commit       Pre-registration commit reference, e.g.
                        DwayneWilkes/affect-battery@<squash-sha>.

Optional:
  --output-base PATH    Output root (default: results/h3b_2026-05-07).
  --max-parallel N      Max concurrent passes in flight (default: 20).
  --n-passes N          Number of passes (default: 20).
  --seed N              Sampler seed (default: 42).
  --dry-run             Pass --dry-run through to affect-battery (canned
                        responses, no API calls; bypasses pre-reg/power
                        gates). For wiring/E2E sanity checks.

Pre-registration: docs/preregistrations/h3b_2026-05-07.md
Environment:      OPENAI_API_KEY must be set.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prereg-commit) PREREG_COMMIT="$2"; shift 2;;
        --output-base)   OUTPUT_BASE="$2";   shift 2;;
        --max-parallel)  MAX_PARALLEL="$2";  shift 2;;
        --n-passes)      N_PASSES="$2";      shift 2;;
        --seed)           SEED="$2";          shift 2;;
        --dry-run)        DRY_RUN=1;          shift;;
        -h|--help)       usage; exit 0;;
        *) echo "unknown flag: $1" >&2; usage >&2; exit 1;;
    esac
done

if [[ -z "$PREREG_COMMIT" ]]; then
    echo "ERROR: --prereg-commit is required" >&2
    echo "  format: <owner/repo>@<sha>, e.g. DwayneWilkes/affect-battery@abc1234" >&2
    exit 1
fi
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY env var must be set" >&2
    exit 1
fi
if [[ ! -f "$BANK" ]]; then
    echo "ERROR: bank YAML not found at $BANK (run from repo root)" >&2
    exit 1
fi
if [[ ! -f "$RUNNER_CONFIG" ]]; then
    echo "ERROR: runner config not found at $RUNNER_CONFIG (run from repo root)" >&2
    exit 1
fi

# Bash 5.1+ is required for `wait -n -p VAR` (used in the dispatch
# loop to identify which background job was reaped). BASH_VERSINFO is
# the structured version array — checking [0]/[1] avoids parsing the
# human-readable string. Validated up-front, before any side effects,
# so a pre-5.1 bash exits cleanly without leaving an output dir or
# manifest behind.
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo "ERROR: bash $BASH_VERSION lacks 'wait -n -p VAR' (need 5.1+)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_BASE"

# Derive the expected per-pass cell count from the canonical artifacts so
# the count check survives bank or intensity-level changes without script
# edits. Bank items are counted by `^- id:` lines (yaml dump format used
# by build_calibrated_bank.py); intensity levels come from the runner
# config's `intensity_levels: [...]` list.
N_BANK_ITEMS=$(grep -cE '^- id:' "$BANK" || true)
N_LEVELS=$(grep -E '^intensity_levels:' "$RUNNER_CONFIG" | grep -oE '[0-9]+' | wc -l || true)
if [[ "$N_BANK_ITEMS" -lt 1 || "$N_LEVELS" -lt 1 ]]; then
    echo "ERROR: could not derive item or level counts (items=$N_BANK_ITEMS, levels=$N_LEVELS)" >&2
    exit 1
fi
EXPECTED_CELLS_PER_PASS=$((N_BANK_ITEMS * N_LEVELS))

MANIFEST="$OUTPUT_BASE/run_manifest.txt"
{
    echo "H3b Phase 1A: single-turn calibrated replication (20 within-subjects passes)"
    echo "prereg:                docs/preregistrations/h3b_2026-05-07.md"
    echo "prereg_commit:         $PREREG_COMMIT"
    echo "bank:                  $BANK"
    echo "runner_config:         $RUNNER_CONFIG"
    echo "n_bank_items:          $N_BANK_ITEMS"
    echo "n_levels:              $N_LEVELS"
    echo "expected_cells/pass:   $EXPECTED_CELLS_PER_PASS"
    echo "n_passes:              $N_PASSES"
    echo "max_parallel:          $MAX_PARALLEL"
    echo "seed:                  $SEED"
    echo "started_utc:           $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$MANIFEST"

# Bounded-concurrency dispatch. Cap concurrent in-flight passes at
# MAX_PARALLEL via `wait -n`. Each pass writes to its own subdir so
# filenames don't collide.
#
# Pass-skipping discipline: a pass is skipped only when its output
# directory holds the full expected_cells_per_pass count of result
# JSONs. Partial directories (crashed mid-pass) are re-dispatched; the
# runner's per-cell cache layer (documented in CLAUDE.md) resumes from
# the missing cells without re-billing the API for completed ones.
#
# Failure handling: each background PID is tracked alongside its pass
# number. The first pass to fail triggers SIGTERM on every other
# in-flight pass before the script exits; this stops API budget burn
# the moment we know the pre-reg-locked count cannot be reached.
declare -A IN_FLIGHT=()       # pid → pass number, for currently-running passes
declare -i FAILED_COUNT=0
declare -a FAILED_PASSES=()

# Cell files match `<digits>.json` under `level_*/neutral/`. Centralise
# the find filter so the dispatch-loop "skip complete" check (line ~190)
# and the post-run total enforcement (bottom of script) cannot drift
# apart and silently miscount.
find_cell_files() {
    find "$1" -path '*/level_*/neutral/*.json' -name '[0-9]*.json' 2>/dev/null
}

SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')

terminate_inflight() {
    # Send SIGTERM to every still-running pass and its descendants,
    # via the pass's process group. Each pass is dispatched as
    # `setsid affect-battery ... &`; once setsid() runs the program
    # leads its own session/pgroup and pgid==pid, so signaling the
    # group catches every descendant the kernel has linked to it.
    #
    # Race window: between `&` returning $! and setsid() executing
    # in the child, the child's pgid is still $SELF_PGID (inherited).
    # Reading pgid live and comparing to $SELF_PGID lets us avoid
    # `kill -- -$SELF_PGID` (which would signal the wrapper itself)
    # and instead signal the lone PID, which is pre-fork so has no
    # descendants yet.
    local pid pgid
    for pid in "${!IN_FLIGHT[@]}"; do
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [[ -z "$pgid" ]]; then
            continue  # process already gone
        fi
        if [[ "$pgid" == "$SELF_PGID" ]]; then
            kill -TERM "$pid" 2>/dev/null || true
        else
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
    done
}

trap_handler() {
    # External SIGINT/SIGTERM: tear down in-flight passes, then exit
    # with the conventional 128+signal code. Per-signal traps let us
    # distinguish 130 (SIGINT) from 143 (SIGTERM); automation that
    # classifies shutdown reason by exit code relies on the convention.
    local sig="$1"
    terminate_inflight
    wait 2>/dev/null || true
    case "$sig" in
        INT)  exit 130;;
        TERM) exit 143;;
        *)    exit 1;;
    esac
}
trap 'trap_handler INT' INT
trap 'trap_handler TERM' TERM

# Account a pass's exit code: 0 = success (no-op), anything else =
# failure. Called from both the wait -n branch and the final drain.
account_status() {
    local pass="$1" rc="$2"
    if [[ "$rc" -ne 0 ]]; then
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_PASSES+=("$pass")
    fi
}

for pass in $(seq 1 "$N_PASSES"); do
    pass_dir=$(printf "%s/pass_%02d" "$OUTPUT_BASE" "$pass")
    existing_cells=0
    if [[ -d "$pass_dir" ]]; then
        existing_cells=$(find_cell_files "$pass_dir" | wc -l)
    fi
    if [[ "$existing_cells" -ge "$EXPECTED_CELLS_PER_PASS" ]]; then
        echo "pass $pass: $existing_cells/$EXPECTED_CELLS_PER_PASS cells complete, skipping"
        continue
    fi
    if [[ "$existing_cells" -gt 0 ]]; then
        echo "pass $pass: $existing_cells/$EXPECTED_CELLS_PER_PASS cells partial, re-dispatching (runner cache resumes)"
    else
        echo "pass $pass: dispatching to $pass_dir"
    fi
    # Dispatch the pass in its own session/process group so cleanup
    # can signal the group atomically (see `terminate_inflight`).
    # Backgrounding `setsid prog` directly (no `(...)& ` wrapper) makes
    # bash's $! point at the program itself: bash forks, the child
    # execs `setsid`, setsid calls setsid() and execs the program in
    # place. PID == pgid == sid for the running pass.
    extra_args=()
    if [[ "$DRY_RUN" -eq 1 ]]; then
        extra_args+=("--dry-run")
    fi
    # exp3a's task items come from --transfer-bank (a YAML path);
    # --bank is unused for exp3a (it would route through the
    # ArithmeticBank loader which our calibrated task bank is not).
    setsid affect-battery run \
        --experiment exp3a \
        --provider openai \
        --model gpt-5.4-nano \
        --transfer-bank "$BANK" \
        --num-runs 18 \
        --seed "$SEED" \
        --temperature 0.7 \
        --runner-config "$RUNNER_CONFIG" \
        --pre-registration-github-commit "$PREREG_COMMIT" \
        --output-dir "$pass_dir" \
        "${extra_args[@]}" &
    IN_FLIGHT[$!]=$pass
    if [[ "${#IN_FLIGHT[@]}" -ge "$MAX_PARALLEL" ]]; then
        # Drain the next finishing background job. `wait -n -p` returns
        # the reaped PID via the named variable so we can attribute the
        # exit status to its pass number. On non-zero exit, terminate
        # remaining passes immediately rather than burning more API
        # budget on a pre-reg that can no longer reach its locked cell
        # count.
        caught_pid=""
        rc=0
        wait -n -p caught_pid "${!IN_FLIGHT[@]}" || rc=$?
        caught_pass="${IN_FLIGHT[$caught_pid]:-?}"
        unset "IN_FLIGHT[$caught_pid]"
        account_status "$caught_pass" "$rc"
        if [[ "$rc" -ne 0 ]]; then
            echo "ERROR: pass $caught_pass failed (exit $rc); terminating remaining in-flight passes" >&2
            terminate_inflight
            break
        fi
    fi
done

# Drain whatever is still in flight. Per-PID `wait` returns the
# child's exit code (or 143 for SIGTERM'd passes). Capture via
# `|| rc=$?` not `if ! wait; then rc=$?` — bash negates the pipeline's
# exit status when `!` is used, so `$?` inside the `then` block is
# always 0 and would silently mis-attribute statuses.
for pid in "${!IN_FLIGHT[@]}"; do
    pass="${IN_FLIGHT[$pid]}"
    rc=0
    wait "$pid" 2>/dev/null || rc=$?
    account_status "$pass" "$rc"
done

trap - INT
trap - TERM

{
    echo "completed_utc:         $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "failed_passes:         $FAILED_COUNT"
} >> "$MANIFEST"

if [[ "$FAILED_COUNT" -gt 0 ]]; then
    echo
    echo "FAILURE: $FAILED_COUNT pass(es) failed: ${FAILED_PASSES[*]:-(killed before completion)}" >&2
    echo "Manifest: $MANIFEST" >&2
    exit 1
fi

echo
echo "All $N_PASSES passes complete. Manifest: $MANIFEST"
echo "Cell-count check:"
total_cells=$(find_cell_files "$OUTPUT_BASE" | wc -l)
expected_total=$((N_PASSES * EXPECTED_CELLS_PER_PASS))
echo "  total result JSONs: $total_cells (expected $expected_total = $N_PASSES passes × $N_BANK_ITEMS items × $N_LEVELS levels)"
if [[ "$total_cells" -ne "$expected_total" ]]; then
    {
        echo "ERROR: cell-count mismatch — got $total_cells, expected $expected_total"
        echo "       Pre-registered analysis assumes $expected_total cells; missing"
        echo "       cells will silently bias the (item_id, level) corpus. Inspect"
        echo "       per-pass cell counts under $OUTPUT_BASE/pass_*/ before any"
        echo "       analysis or amendment."
    } >&2
    {
        echo "cell_count_check:      FAIL ($total_cells/$expected_total)"
    } >> "$MANIFEST"
    exit 1
fi
echo "cell_count_check:      OK ($total_cells/$expected_total)" >> "$MANIFEST"
