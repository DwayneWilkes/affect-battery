#!/usr/bin/env bash
# Run a single-experiment pilot end-to-end against any supported provider
# (anthropic, openai, vllm). Default provider is anthropic for legacy
# parity, but --provider openai / --provider vllm work too.
#
# Steps:
#   1. Verify the appropriate API key is set for the chosen --provider
#   2. Verify a pre-registration tag was created (or accept --skip-prereg)
#   3. Run affect-battery pilot via the configured provider
#   4. Run affect-battery analyze on the produced results
#
# Usage:
#   bash scripts/pilots/run_pilot.sh
#   bash scripts/pilots/run_pilot.sh --model claude-opus-4-7
#   bash scripts/pilots/run_pilot.sh --provider openai --model gpt-5.4-nano
#   bash scripts/pilots/run_pilot.sh --dry-run     # offline smoke test
#   bash scripts/pilots/run_pilot.sh --skip-prereg # pilot under existing pre-reg
#
# Output: results/pilots/<YYYY-MM-DD>_<model_slug>/<experiment>/

set -euo pipefail

# Unset any inherited VIRTUAL_ENV so uv silently uses this project's
# .venv instead of warning about the mismatch when the parent shell
# has another project's venv activated. This is the right behavior
# anyway: project venvs should win over ambient activation, otherwise
# running this script from inside another project's shell would pick
# up the wrong dependency set.
unset VIRTUAL_ENV

# Default to Haiku for cheapest validation. Override with --model
# claude-sonnet-4-6 or --model claude-opus-4-7 once the harness is
# verified end-to-end on Haiku.
PROVIDER="anthropic"
MODEL="claude-haiku-4-5"
EXPERIMENT="exp1a"
NUM_RUNS=5
SEED=42
DRY_RUN=""
SKIP_PREREG=""
PREREG_TAG=""
RUNNER_CONFIG=""
NEUTRAL_TURNS=0
OVERWRITE=""
ESTIMATE=""
# Concurrency + rate limit. Defaults tuned conservatively for tier-1
# and tier-2 API accounts (~200-500 RPM on small models). At 8 RPS
# the aggregate stays under 480 RPM, comfortable for tier-2 OpenAI
# (500 RPM gpt-5.4-nano) and tier-2 Anthropic (1000+ RPM Haiku 4.5).
# Tier-3+ accounts can override via --rate-limit-rps to 30+ and
# saturate higher RPM ceilings. The SDKs handle transient 429s via
# built-in retry + Retry-After; our wrapper retries sustained
# rate-limit errors up to 3 more times before raising.
#
# In --parallel mode, the orchestrator auto-divides this rate budget
# across the N parallel experiments so the aggregate to the provider
# stays at the user-set ceiling. (Each subprocess has its own
# client-side limiter; they don't share state.)
MAX_CONCURRENT=16
RATE_LIMIT_RPS=8
# Default to the alias-aware TriviaQA hard subset so frontier models don't
# saturate on the legacy 6-item hardcoded pool. Override with
# --transfer-bank '' to use the legacy pool, or with another bank path.
TRANSFER_BANK="configs/banks/exp1a_factual_qa_hard_v1.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider)       PROVIDER="$2"; shift 2 ;;
    --model)          MODEL="$2"; shift 2 ;;
    --experiment)     EXPERIMENT="$2"; shift 2 ;;
    --num-runs)       NUM_RUNS="$2"; shift 2 ;;
    --seed)           SEED="$2"; shift 2 ;;
    --dry-run)        DRY_RUN="--dry-run"; shift ;;
    --skip-prereg)    SKIP_PREREG="1"; shift ;;
    --prereg-tag)     PREREG_TAG="$2"; shift 2 ;;
    --transfer-bank)  TRANSFER_BANK="$2"; shift 2 ;;
    --runner-config)  RUNNER_CONFIG="$2"; shift 2 ;;
    --neutral-turns)  NEUTRAL_TURNS="$2"; shift 2 ;;
    --overwrite)      OVERWRITE="--overwrite"; shift ;;
    --estimate)       ESTIMATE="--estimate"; shift ;;
    --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
    --rate-limit-rps) RATE_LIMIT_RPS="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# exp3a/3b/3c require a per-experiment runner config (intensity_levels,
# prompts, items). Surface the requirement up-front rather than letting
# the CLI exit with a less-clear error mid-pilot.
case "${EXPERIMENT}" in
  exp3a|exp3b|exp3c)
    if [[ -z "${RUNNER_CONFIG}" ]]; then
      echo "error: --experiment ${EXPERIMENT} requires --runner-config <yaml>" >&2
      echo "       (intensity_levels for exp3a, prompts for exp3b, items for exp3c)" >&2
      exit 2
    fi
    ;;
esac

# Date stamp: honor PILOT_DATE_STAMP env var if the orchestrator (or
# the user) has set one. This is critical for multi-step orchestrators
# that cross midnight UTC — without it, sub-pilots launched on different
# sides of midnight write to different pilot dirs, scattering related
# data (e.g. exp2's N-sweep) across two roots and breaking analysis.
DATE_STAMP="${PILOT_DATE_STAMP:-$(date -u +%Y-%m-%d)}"
# Pilot directory is `<date>_<model_slug>_<experiment>/` so each
# (date, model, experiment) triple lives in its own dir — no collisions
# between pilots of different experiments or models. Slashes in the
# model name (e.g. 'meta-llama/Llama-3') are replaced with underscores
# so the path stays one level deep.
MODEL_SLUG="${MODEL//\//_}"
if [[ -n "${DRY_RUN}" ]]; then
  # Segregate dry-run output so its canned responses cannot pollute the
  # real-run cache via the (config, run_number) cache key.
  OUTPUT_DIR="results/pilots/${DATE_STAMP}_${MODEL_SLUG}_dryrun/${EXPERIMENT}"
else
  OUTPUT_DIR="results/pilots/${DATE_STAMP}_${MODEL_SLUG}/${EXPERIMENT}"
fi

# ---- Pre-flight ----

# --estimate computes cost + wall-clock without making API calls;
# no API key needed. --dry-run uses canned responses; same.
if [[ -z "${DRY_RUN}" && -z "${ESTIMATE}" ]]; then
  case "${PROVIDER}" in
    anthropic)  KEY_VAR="ANTHROPIC_API_KEY" ;;
    openai)     KEY_VAR="OPENAI_API_KEY" ;;
    *)          KEY_VAR="" ;;  # vllm or unknown; no key check
  esac
  if [[ -n "${KEY_VAR}" && -z "${!KEY_VAR:-}" ]]; then
    echo "error: ${KEY_VAR} not set. Export it from your shell profile" >&2
    echo "       or pass --dry-run for an offline smoke test." >&2
    exit 2
  fi
fi

# Resolve a pre-registration tag if the user didn't supply one.
if [[ -z "${PREREG_TAG}" && -z "${SKIP_PREREG}" && -z "${DRY_RUN}" && -z "${ESTIMATE}" ]]; then
  # Pick the most recent prereg-* tag, if any.
  PREREG_TAG=$(git tag --list 'prereg-*' --sort=-creatordate | head -1)
  if [[ -z "${PREREG_TAG}" ]]; then
    echo "error: no prereg-* tag found. Create one first with:" >&2
    echo "  python -m scripts.create_prereg_tag --tag prereg-${DATE_STAMP}" >&2
    echo "or pass --skip-prereg / --dry-run." >&2
    exit 2
  fi
fi

# Convert prereg tag -> owner/repo@sha for the CLI flag.
# Power-report gate is intentionally skipped here: pilots run BEFORE
# the variance probe that grounds the power report, so requiring one
# would be circular. The pre-reg gate stays armed (the methodology
# IS the pre-reg) so pilot runs are still locked to a specific
# commit. If pilot results are later promoted to primary without an
# amendment, the analysis pipeline emits the
# `pre_registration_violation: pilot_promoted_to_primary` audit code.
PREREG_FLAG=""
if [[ -n "${PREREG_TAG}" ]]; then
  ORIGIN_URL=$(git remote get-url origin)
  # Match git@github.com:owner/repo.git or https://github.com/owner/repo.git
  if [[ "${ORIGIN_URL}" =~ github\.com[:/]([^/]+)/([^/.]+) ]]; then
    OWNER="${BASH_REMATCH[1]}"
    REPO="${BASH_REMATCH[2]}"
  else
    echo "error: cannot parse origin URL (${ORIGIN_URL})" >&2
    exit 2
  fi
  TAG_SHA=$(git rev-parse "${PREREG_TAG}^{commit}")
  PREREG_FLAG="--pre-registration-github-commit ${OWNER}/${REPO}@${TAG_SHA} --skip-power-gate"
  echo "Using pre-registration: ${PREREG_TAG} -> ${OWNER}/${REPO}@${TAG_SHA:0:12}"
  echo "Power-report gate skipped (pilots run before the variance probe)."
fi

if [[ -n "${SKIP_PREREG}" ]]; then
  PREREG_FLAG="--skip-prereg-gate --skip-power-gate"
  echo "WARNING: skipping BOTH pre-registration + power gates (explicit pilot bypass)"
fi

# ---- Run ----

echo ""
echo "Running ${PROVIDER} pilot:"
echo "  model:         ${MODEL}"
echo "  experiment:    ${EXPERIMENT}"
echo "  num_runs:      ${NUM_RUNS}"
echo "  seed:          ${SEED}"
echo "  neutral_turns: ${NEUTRAL_TURNS} (only used by exp2)"
echo "  output_dir:    ${OUTPUT_DIR}"
echo "  transfer_bank: ${TRANSFER_BANK:-<legacy hardcoded pool>}"
echo "  runner_config: ${RUNNER_CONFIG:-<none, only required for exp3a/3b/3c>}"
echo "  overwrite:     ${OVERWRITE:-no (resume-by-default)}"
echo "  concurrency:   max=${MAX_CONCURRENT}, rate=${RATE_LIMIT_RPS} RPS"
echo ""

# Note: `affect-battery pilot` runs all 7 conditions × num_runs.
# Total API calls ≈ num_runs × 7 conditions × ~10 turns = ~350 calls
# at the default num_runs=5, ~650 tokens per call.
# Cost at default model:
#   Haiku 4.5: ~$0.80/M input + $4/M output  -> ~$0.40-0.80 total
#   Sonnet 4.6: ~$3/M input + $15/M output   -> ~$2-3 total
#   Opus 4.7: ~$15/M input + $75/M output    -> ~$10-15 total
# Cost-per-call default below is calibrated for Haiku; override with
# --model claude-sonnet-4-6 / --model claude-opus-4-7 (and the script
# will scale --cost-per-call accordingly via the case statement above).

# Scale cost-per-call by model tier so the budget guardrail stays
# honest across providers + tiers. Conservative blended estimates
# (input + output for ~500-token avg call). The --estimate path
# uses proper input/output split via the Python tier resolver; this
# is just the runtime budget cap.
case "${MODEL}" in
  *opus*|*gpt-5.5*|*5.5-pro*|*5.4-pro*) COST_PER_CALL="0.080" ;;  # frontier
  *sonnet*|*gpt-5.4)                    COST_PER_CALL="0.015" ;;  # mid
  *haiku*|*gpt-5.4-mini|*5.4-mini)      COST_PER_CALL="0.003" ;;  # small
  *gpt-5.4-nano|*5.4-nano|*nano*)       COST_PER_CALL="0.0008" ;; # nano
  *)                                    COST_PER_CALL="0.015" ;;  # conservative default
esac

# Forward optional flags only when set; the CLI argparser would reject
# empty string values, so we build flag strings conditionally.
TRANSFER_BANK_FLAG=""
if [[ -n "${TRANSFER_BANK}" ]]; then
  TRANSFER_BANK_FLAG="--transfer-bank ${TRANSFER_BANK}"
fi
RUNNER_CONFIG_FLAG=""
if [[ -n "${RUNNER_CONFIG}" ]]; then
  RUNNER_CONFIG_FLAG="--runner-config ${RUNNER_CONFIG}"
fi
# For --estimate, omit --cost-per-call so the Python tier resolution
# can pick the right per-model default and label it 'exact'/'tier'
# in the estimate output. For real runs, the bash default is a
# meaningful budget guard.
COST_PER_CALL_FLAG=""
if [[ -z "${ESTIMATE}" ]]; then
  COST_PER_CALL_FLAG="--cost-per-call ${COST_PER_CALL}"
fi

uv run affect-battery pilot \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --experiment "${EXPERIMENT}" \
  --num-runs "${NUM_RUNS}" \
  --seed "${SEED}" \
  --neutral-turns "${NEUTRAL_TURNS}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-concurrent "${MAX_CONCURRENT}" \
  --rate-limit-rps "${RATE_LIMIT_RPS}" \
  --budget-max-calls 500 \
  ${COST_PER_CALL_FLAG} \
  ${TRANSFER_BANK_FLAG} \
  ${RUNNER_CONFIG_FLAG} \
  ${OVERWRITE} \
  ${ESTIMATE} \
  ${DRY_RUN} \
  ${PREREG_FLAG}

# ---- Analyze ----
# Skip when --estimate (no real pilot ran, no results to analyze).

if [[ -z "${ESTIMATE}" ]]; then
  echo ""
  echo "Running analysis on pilot results..."
  uv run affect-battery analyze \
    --results-dir "${OUTPUT_DIR}" \
    --model "${MODEL}"

  echo ""
  echo "Pilot complete. Reports under ${OUTPUT_DIR}/reports/"
  echo "Top-level: ${OUTPUT_DIR}/reports/AGGREGATE_REPORT.md"
  echo "Manifest:  ${OUTPUT_DIR}/manifest.yaml"
fi
