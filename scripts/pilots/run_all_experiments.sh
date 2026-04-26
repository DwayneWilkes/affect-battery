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
# Default aggregate rate-limit budget. In sequential mode this passes
# through to one pilot. In --parallel mode the orchestrator divides
# this across N experiments to keep aggregate at this ceiling. Tuned
# for tier-1/tier-2 accounts (~200-500 RPM); tier-3+ can override.
RATE_LIMIT_RPS=8

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
#
# Critical for --parallel: each subprocess has its own client-side
# rate limiter; they don't share state. So if you ask for 20 RPS per
# pilot and run 5 in parallel, you get 100 RPS aggregate to the
# provider — way over typical tier-1/tier-2 RPM caps. Auto-divide
# the rate budget across N parallel experiments so the aggregate
# stays at the user-set ceiling. Concurrency cap (sockets) is
# per-process so doesn't need division.
EFFECTIVE_RATE_LIMIT_RPS="${RATE_LIMIT_RPS}"
if [[ -n "${PARALLEL}" && -n "${RATE_LIMIT_RPS}" ]]; then
  N_EXP=${#EXPERIMENTS[@]}
  EFFECTIVE_RATE_LIMIT_RPS=$(awk -v r="${RATE_LIMIT_RPS}" -v n="${N_EXP}" \
    'BEGIN { printf "%.2f", r / n }')
  echo ""
  echo "  [parallel] dividing --rate-limit-rps ${RATE_LIMIT_RPS} across "
  echo "  ${N_EXP} parallel experiments → ${EFFECTIVE_RATE_LIMIT_RPS} RPS each"
  echo "  (aggregate stays at ${RATE_LIMIT_RPS} RPS)"
fi
CONCURRENCY_FLAGS=""
if [[ -n "${MAX_CONCURRENT}" ]]; then
  CONCURRENCY_FLAGS+="--max-concurrent ${MAX_CONCURRENT} "
fi
if [[ -n "${EFFECTIVE_RATE_LIMIT_RPS}" ]]; then
  CONCURRENCY_FLAGS+="--rate-limit-rps ${EFFECTIVE_RATE_LIMIT_RPS} "
fi

START_TIME=$(date +%s)
FAILED=()
# Per-experiment aggregation: each pilot subprocess emits an
# [ESTIMATE_SUMMARY] / [RUN_SUMMARY] line we grep for at the end.
# In parallel mode each subprocess writes to its own log; in
# sequential mode all output goes to one shared log.
LOG_DIR="$(mktemp -d)"
# Track child PIDs so SIGINT can forward to all of them. Bash's
# default backgrounded-job handling does NOT reliably forward
# SIGINT — the orchestrator gets it, the children don't, and the
# user has to kill -9 to stop them.
declare -a CHILD_PIDS=()

cleanup_on_signal() {
  local signal_name="${1:-INT}"
  echo "" >&2
  echo "[orchestrator] ${signal_name} received; forwarding to ${#CHILD_PIDS[@]} child experiments..." >&2
  for pid in "${CHILD_PIDS[@]}"; do
    kill -INT "${pid}" 2>/dev/null || true
  done
  # Give them up to 10s to drain in-flight API calls cleanly.
  local deadline=$(( $(date +%s) + 10 ))
  while [[ $(date +%s) -lt ${deadline} ]]; do
    local alive=0
    for pid in "${CHILD_PIDS[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$(( alive + 1 ))
      fi
    done
    if [[ ${alive} -eq 0 ]]; then break; fi
    sleep 0.5
  done
  # Anyone still alive after grace gets SIGKILL.
  for pid in "${CHILD_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[orchestrator] PID ${pid} didn't drain; force-killing." >&2
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  rm -rf "${LOG_DIR}" 2>/dev/null
  exit 130
}

trap 'cleanup_on_signal INT' INT
trap 'cleanup_on_signal TERM' TERM
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
    # Also track in CHILD_PIDS so cleanup_on_signal can find these
    # without needing to declare -A access from inside the trap.
    CHILD_PIDS+=("$!")
  done
  echo ""
  echo "  waiting for all experiments to complete (Ctrl-C to abort)..."
  echo "  (heartbeat below shows progress every 30s; full per-experiment"
  echo "   logs at ${LOG_DIR}/<exp>.log — \`tail -F ${LOG_DIR}/*.log\` to live-tail)"
  echo ""

  # Heartbeat: periodically print which experiments are still running,
  # along with the last line of each one's log so the user can see
  # progress without parsing 5 interleaved tqdm streams.
  heartbeat_loop() {
    while true; do
      sleep 30
      local now elapsed mins secs
      now=$(date +%s)
      elapsed=$(( now - START_TIME ))
      mins=$(( elapsed / 60 ))
      secs=$(( elapsed % 60 ))
      local still_running=()
      for e in "${EXPERIMENTS[@]}"; do
        if kill -0 "${PIDS[$e]}" 2>/dev/null; then
          still_running+=("${e}")
        fi
      done
      if [[ ${#still_running[@]} -eq 0 ]]; then return 0; fi
      echo "  [+${mins}m${secs}s] running: ${still_running[*]}"
      for e in "${still_running[@]}"; do
        # Grab the last non-blank line from each log; tqdm progress
        # bars use \r so we strip CRs to get the last update.
        local last
        last=$(tr '\r' '\n' < "${LOG_DIR}/${e}.log" 2>/dev/null \
               | grep -v '^$' | tail -1 | head -c 100)
        if [[ -n "${last}" ]]; then
          echo "    ${e}: ${last}"
        fi
      done
    done
  }
  heartbeat_loop &
  HEARTBEAT_PID=$!
  CHILD_PIDS+=("${HEARTBEAT_PID}")  # also kill on SIGINT

  for exp in "${EXPERIMENTS[@]}"; do
    if ! wait "${PIDS[$exp]}"; then
      FAILED+=("${exp}")
    fi
  done

  # Stop the heartbeat now that all experiments have completed.
  kill "${HEARTBEAT_PID}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" 2>/dev/null || true
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
