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
        --seed)          SEED="$2";          shift 2;;
        -h|--help)       usage; exit 0;;
        *) echo "unknown flag: $1" >&2; usage; exit 1;;
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

mkdir -p "$OUTPUT_BASE"

# Derive the expected per-pass cell count from the canonical artifacts so
# the count check survives bank or intensity-level changes without script
# edits. Bank items are counted by `^- id:` lines (yaml dump format used
# by build_calibrated_bank.py); intensity levels come from the runner
# config's `intensity_levels: [...]` list.
N_BANK_ITEMS=$(grep -cE '^- id:' "$BANK")
N_LEVELS=$(grep -E '^intensity_levels:' "$RUNNER_CONFIG" | grep -oE '[0-9]+' | wc -l)
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
in_flight=0
for pass in $(seq 1 "$N_PASSES"); do
    pass_dir=$(printf "%s/pass_%02d" "$OUTPUT_BASE" "$pass")
    existing_cells=0
    if [[ -d "$pass_dir" ]]; then
        existing_cells=$(find "$pass_dir" -path "*/level_*/neutral/*.json" 2>/dev/null | wc -l)
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
    (
        affect-battery run \
            --experiment exp3a \
            --provider openai \
            --model gpt-5.4-nano \
            --bank "$BANK" \
            --num-runs 18 \
            --seed "$SEED" \
            --temperature 0.7 \
            --runner-config "$RUNNER_CONFIG" \
            --pre-registration-github-commit "$PREREG_COMMIT" \
            --output-dir "$pass_dir"
    ) &
    in_flight=$((in_flight + 1))
    if [[ "$in_flight" -ge "$MAX_PARALLEL" ]]; then
        wait -n
        in_flight=$((in_flight - 1))
    fi
done

# Drain remaining background jobs.
wait

{
    echo "completed_utc:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$MANIFEST"

echo
echo "All $N_PASSES passes complete. Manifest: $MANIFEST"
echo "Cell-count check:"
total_cells=$(find "$OUTPUT_BASE" -path "*/level_*/neutral/*.json" | wc -l)
expected_total=$((N_PASSES * EXPECTED_CELLS_PER_PASS))
echo "  total result JSONs: $total_cells (expected $expected_total = $N_PASSES passes × $N_BANK_ITEMS items × $N_LEVELS levels)"
