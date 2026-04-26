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
PARALLEL=""
MAX_CONCURRENT=""  # if unset, run_anthropic_pilot.sh's default applies (16)
RATE_LIMIT_RPS=""  # if unset, run_anthropic_pilot.sh's default applies (20)

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
    --parallel)     PARALLEL="1"; shift ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --rate-limit-rps) RATE_LIMIT_RPS="$2"; shift 2 ;;
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

# Build optional concurrency flags so empty values don't pass through
# as empty strings (the CLI argparser would reject them).
CONCURRENCY_FLAGS=""
if [[ -n "${MAX_CONCURRENT}" ]]; then
  CONCURRENCY_FLAGS+="--max-concurrent ${MAX_CONCURRENT} "
fi
if [[ -n "${RATE_LIMIT_RPS}" ]]; then
  CONCURRENCY_FLAGS+="--rate-limit-rps ${RATE_LIMIT_RPS} "
fi

START_TIME=$(date +%s)
FAILED=()
# Per-experiment aggregation: each pilot subprocess emits an
# [ESTIMATE_SUMMARY] / [RUN_SUMMARY] line we grep for at the end.
# In parallel mode each subprocess writes to its own log; in
# sequential mode all output goes to one shared log.
LOG_DIR="$(mktemp -d)"
trap "rm -rf '${LOG_DIR}'" EXIT

# Per-experiment runner. Captures stdout+stderr to a per-exp log and
# echoes a status marker on completion. Designed to be backgrounded.
run_one_experiment() {
  local exp="$1"
  local log="${LOG_DIR}/${exp}.log"
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
        ${DRY_RUN} \
        ${CONCURRENCY_FLAGS} > "${log}" 2>&1; then
    echo "  ✓ ${exp} complete (log: ${log})"
    return 0
  else
    echo "  ✗ ${exp} FAILED (log: ${log})" >&2
    return 1
  fi
}

if [[ -n "${PARALLEL}" ]]; then
  echo ""
  echo "Running ${#EXPERIMENTS[@]} experiments IN PARALLEL."
  echo "Per-experiment output streams to ${LOG_DIR}/<exp>.log;"
  echo "wall-clock will be bounded by the slowest experiment, not the sum."
  echo ""
  declare -A PIDS
  for exp in "${EXPERIMENTS[@]}"; do
    echo "  ▶ launching ${exp}..."
    run_one_experiment "${exp}" &
    PIDS[$exp]=$!
  done
  echo ""
  echo "  waiting for all experiments to complete..."
  for exp in "${EXPERIMENTS[@]}"; do
    if ! wait "${PIDS[$exp]}"; then
      FAILED+=("${exp}")
    fi
  done
  # After all done, dump the per-experiment logs sequentially so the
  # user sees the full picture without interleaved chaos.
  for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "===================================="
    echo "  Output: ${exp}"
    echo "===================================="
    cat "${LOG_DIR}/${exp}.log"
  done
else
  # Sequential mode (original behavior). Stream live to terminal + log.
  for exp in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "===================================="
    echo "  Pilot: ${exp}"
    echo "===================================="
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
          ${DRY_RUN} \
          ${CONCURRENCY_FLAGS} 2>&1 | tee "${LOG_DIR}/${exp}.log"; then
      echo "  ✓ ${exp} complete"
    else
      echo "  ✗ ${exp} FAILED" >&2
      FAILED+=("${exp}")
    fi
  done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Sum cost + wall-clock from the per-experiment log files.
TOTAL_COST=$(cat "${LOG_DIR}"/*.log 2>/dev/null \
  | grep -oE '\[(ESTIMATE|RUN)_SUMMARY\] cost_usd=[0-9.]+' \
  | sed 's/.*cost_usd=//' \
  | awk '{ s += $1 } END { printf "%.4f", s }')
TOTAL_WALL_SEC=$(cat "${LOG_DIR}"/*.log 2>/dev/null \
  | grep -oE 'wall_clock_sec=[0-9.]+' \
  | sed 's/wall_clock_sec=//' \
  | awk '{ s += $1 } END { printf "%.1f", s }')
# In parallel mode, the per-experiment max is the bound on real
# wall-clock (not the sum, which is what the experiments would have
# taken sequentially).
MAX_WALL_SEC=$(cat "${LOG_DIR}"/*.log 2>/dev/null \
  | grep -oE 'wall_clock_sec=[0-9.]+' \
  | sed 's/wall_clock_sec=//' \
  | awk 'BEGIN { m = 0 } { if ($1 > m) m = $1 } END { printf "%.1f", m }')

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
echo "  mode:            $([[ -n "${PARALLEL}" ]] && echo "parallel" || echo "sequential")"
echo "  orchestrator_elapsed: $((ELAPSED / 60))m $((ELAPSED % 60))s"
if [[ -n "${TOTAL_WALL_SEC}" && "${TOTAL_WALL_SEC}" != "0.0" ]]; then
  TOTAL_WALL_MIN=$(awk -v s="${TOTAL_WALL_SEC}" 'BEGIN { printf "%.1f", s / 60 }')
  MAX_WALL_MIN=$(awk -v s="${MAX_WALL_SEC}" 'BEGIN { printf "%.1f", s / 60 }')
  if [[ -n "${PARALLEL}" ]]; then
    if [[ -n "${ESTIMATE}" ]]; then
      echo "  estimated wall:  ${MAX_WALL_MIN}m parallel  (vs ${TOTAL_WALL_MIN}m sequential)"
    else
      echo "  per-pilot wall:  max ${MAX_WALL_MIN}m  (sum across: ${TOTAL_WALL_MIN}m)"
    fi
  else
    if [[ -n "${ESTIMATE}" ]]; then
      echo "  estimated wall:  ${TOTAL_WALL_MIN}m sequential  (would be ~${MAX_WALL_MIN}m with --parallel)"
    else
      echo "  per-pilot wall:  ${TOTAL_WALL_MIN}m (sum of per-experiment elapsed)"
    fi
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
  PILOT_ROOT="results/pilots/${DATE_STAMP}_${MODEL_SLUG}"
  echo "  Pilot root:    ${PILOT_ROOT}/"
  echo "  Per-experiment dirs:"
  for exp in "${EXPERIMENTS[@]}"; do
    DIR="${PILOT_ROOT}/${exp}"
    if [[ -d "${DIR}" ]]; then
      echo "    ${DIR}/"
    fi
  done
fi

# Exit non-zero if any experiment failed so CI catches it.
if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
