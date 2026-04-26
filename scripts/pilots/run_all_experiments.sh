#!/usr/bin/env bash
# Run all 5 smokeable experiments for a given model in sequence.
# (exp3a deferred — requires Krippendorff intensity-pilot pass before
# the runner accepts a pilot_seed_path. See spec for details.)
#
# Each experiment lands in its own pilot dir per the per-experiment
# naming convention in run_anthropic_pilot.sh:
#   results/pilots/<YYYY-MM-DD>_<model_slug>_<experiment>/
#
# Usage:
#   bash scripts/pilots/run_all_experiments.sh                          # Haiku, num-runs=2
#   bash scripts/pilots/run_all_experiments.sh --model claude-sonnet-4-6
#   bash scripts/pilots/run_all_experiments.sh --num-runs 5
#   bash scripts/pilots/run_all_experiments.sh --estimate               # cost preview, no API spend
#   bash scripts/pilots/run_all_experiments.sh --dry-run                # offline smoke
#
# All flags are forwarded to run_anthropic_pilot.sh; check that
# script's --help for the full list.

set -euo pipefail
unset VIRTUAL_ENV  # let direnv pick this project's venv

PROVIDER="anthropic"
MODEL="claude-haiku-4-5"
NUM_RUNS=2
SEED=42
ESTIMATE=""
OVERWRITE=""
SKIP_PREREG=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)     PROVIDER="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --num-runs)     NUM_RUNS="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --estimate)     ESTIMATE="--estimate"; shift ;;
    --overwrite)    OVERWRITE="--overwrite"; shift ;;
    --skip-prereg)  SKIP_PREREG="--skip-prereg"; shift ;;
    --dry-run)      DRY_RUN="--dry-run"; shift ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Five smokeable experiments. exp3a is deliberately omitted; its runner
# refuses to start without a Krippendorff-validated intensity pilot
# seed, which is a separate prerequisite workflow.
EXPERIMENTS=(exp1a exp1b exp2 exp3b exp3c)

# Per-experiment extra flags. exp2 needs --neutral-turns; exp3b/3c need
# --runner-config. Everything else inherits the script defaults.
declare -A EXTRA_FLAGS=(
  [exp1a]=""
  [exp1b]=""
  [exp2]="--neutral-turns 5"
  [exp3b]="--runner-config configs/exp3b_runner.yaml"
  [exp3c]="--runner-config configs/exp3c_runner.yaml"
)

START_TIME=$(date +%s)
FAILED=()
# Per-experiment aggregation: collect cost + wall-clock from the
# [ESTIMATE_SUMMARY] / [RUN_SUMMARY] line each pilot subprocess
# emits. Sum at the end for the headline grand-total.
SUMMARY_LOG="$(mktemp)"
trap "rm -f '${SUMMARY_LOG}'" EXIT

for exp in "${EXPERIMENTS[@]}"; do
  echo ""
  echo "===================================="
  echo "  Pilot: ${exp}"
  echo "===================================="

  # tee the subprocess output to the user's terminal AND to a log we
  # can grep for the machine-readable summary lines after.
  if bash scripts/pilots/run_anthropic_pilot.sh \
        --provider "${PROVIDER}" \
        --model "${MODEL}" \
        --experiment "${exp}" \
        --num-runs "${NUM_RUNS}" \
        --seed "${SEED}" \
        ${EXTRA_FLAGS[$exp]} \
        ${ESTIMATE} \
        ${OVERWRITE} \
        ${SKIP_PREREG} \
        ${DRY_RUN} 2>&1 | tee -a "${SUMMARY_LOG}"; then
    echo "  ✓ ${exp} complete"
  else
    echo "  ✗ ${exp} FAILED" >&2
    FAILED+=("${exp}")
    # Don't bail — the cache layer makes per-experiment retries
    # cheap, and continuing surfaces all failure modes in one run.
  fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Sum cost + wall-clock from the per-experiment summary lines.
# Both [ESTIMATE_SUMMARY] (estimate mode) and [RUN_SUMMARY] (real
# run) use the same `cost_usd=N wall_clock_sec=M` shape.
TOTAL_COST=$(grep -oE '\[(ESTIMATE|RUN)_SUMMARY\] cost_usd=[0-9.]+' "${SUMMARY_LOG}" \
  | sed 's/.*cost_usd=//' \
  | awk '{ s += $1 } END { printf "%.4f", s }')
TOTAL_WALL_SEC=$(grep -oE 'wall_clock_sec=[0-9.]+' "${SUMMARY_LOG}" \
  | sed 's/wall_clock_sec=//' \
  | awk '{ s += $1 } END { printf "%.1f", s }')

echo ""
echo "===================================="
echo "  Multi-experiment pilot summary"
echo "===================================="
echo "  provider:        ${PROVIDER}"
echo "  model:           ${MODEL}"
echo "  num_runs:        ${NUM_RUNS}"
echo "  experiments:     ${#EXPERIMENTS[@]} attempted"
echo "  succeeded:       $((${#EXPERIMENTS[@]} - ${#FAILED[@]}))"
echo "  failed:          ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "    -> ${FAILED[*]}"
fi
# Wall-clock: orchestrator-measured (sequential), and sum of
# per-experiment wall-clocks (which would be the lower bound if
# experiments could run in parallel).
echo "  orchestrator_elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s"
if [[ -n "${TOTAL_WALL_SEC}" && "${TOTAL_WALL_SEC}" != "0.0" ]]; then
  TOTAL_WALL_MIN=$(awk -v s="${TOTAL_WALL_SEC}" 'BEGIN { printf "%.1f", s / 60 }')
  if [[ -n "${ESTIMATE}" ]]; then
    echo "  estimated wall:  ${TOTAL_WALL_MIN}m (sum across experiments; sequential)"
  else
    echo "  per-pilot wall:  ${TOTAL_WALL_MIN}m (sum of per-experiment elapsed)"
  fi
fi
# Cost: sum across experiments. Same line for both --estimate and
# real-run because cmd_pilot's [RUN_SUMMARY] uses the estimator
# post-hoc (we don't track real token usage at runtime today).
if [[ -n "${TOTAL_COST}" && "${TOTAL_COST}" != "0.0000" ]]; then
  if [[ -n "${ESTIMATE}" ]]; then
    echo "  estimated cost:  \$${TOTAL_COST} (sum across experiments)"
  else
    echo "  estimated cost:  \$${TOTAL_COST} (post-hoc estimate; real usage may differ)"
  fi
fi
echo ""

# Aggregate cost / time across pilot manifests when not in --estimate
# or --dry-run mode (those don't produce manifests with real timings).
if [[ -z "${ESTIMATE}" && -z "${DRY_RUN}" ]]; then
  DATE_STAMP=$(date -u +%Y-%m-%d)
  MODEL_SLUG="${MODEL//\//_}"
  echo "  Pilot dirs:"
  for exp in "${EXPERIMENTS[@]}"; do
    DIR="results/pilots/${DATE_STAMP}_${MODEL_SLUG}_${exp}"
    if [[ -d "${DIR}" ]]; then
      echo "    ${DIR}/"
    fi
  done
fi

# Exit non-zero if any experiment failed so CI catches it.
if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
