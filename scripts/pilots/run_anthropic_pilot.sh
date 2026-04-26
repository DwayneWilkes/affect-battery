#!/usr/bin/env bash
# Run a contemporary-Anthropic pilot end-to-end.
#
# Steps:
#   1. Verify ANTHROPIC_API_KEY is set
#   2. Verify a pre-registration tag was created (or accept --skip-prereg)
#   3. Run affect-battery pilot via --provider anthropic
#   4. Run affect-battery analyze on the produced results
#
# Usage:
#   bash scripts/pilots/run_anthropic_pilot.sh
#   bash scripts/pilots/run_anthropic_pilot.sh --model claude-opus-4-7
#   bash scripts/pilots/run_anthropic_pilot.sh --dry-run     # offline smoke test
#   bash scripts/pilots/run_anthropic_pilot.sh --skip-prereg # pilot under existing pre-reg
#
# Output: results/pilots/<YYYY-MM-DD>_<model_slug>_<experiment>/

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
# Default to the alias-aware TriviaQA hard subset so frontier models don't
# saturate on the legacy 6-item hardcoded pool. Override with
# --transfer-bank '' to use the legacy pool, or with another bank path.
TRANSFER_BANK="configs/banks/exp1a_factual_qa_hard_v1.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
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

DATE_STAMP=$(date -u +%Y-%m-%d)
# Pilot directory is `<date>_<model_slug>_<experiment>/` so each
# (date, model, experiment) triple lives in its own dir — no collisions
# between pilots of different experiments or models. Slashes in the
# model name (e.g. 'meta-llama/Llama-3') are replaced with underscores
# so the path stays one level deep.
MODEL_SLUG="${MODEL//\//_}"
if [[ -n "${DRY_RUN}" ]]; then
  # Segregate dry-run output so its canned responses cannot pollute the
  # real-run cache via the (config, run_number) cache key.
  OUTPUT_DIR="results/pilots/${DATE_STAMP}_${MODEL_SLUG}_${EXPERIMENT}_dryrun"
else
  OUTPUT_DIR="results/pilots/${DATE_STAMP}_${MODEL_SLUG}_${EXPERIMENT}"
fi

# ---- Pre-flight ----

if [[ -z "${DRY_RUN}" ]]; then
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "error: ANTHROPIC_API_KEY not set. Export it from your shell profile" >&2
    echo "       or pass --dry-run for an offline smoke test." >&2
    exit 2
  fi
fi

# Resolve a pre-registration tag if the user didn't supply one.
if [[ -z "${PREREG_TAG}" && -z "${SKIP_PREREG}" && -z "${DRY_RUN}" ]]; then
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
echo "Running Anthropic pilot:"
echo "  model:         ${MODEL}"
echo "  experiment:    ${EXPERIMENT}"
echo "  num_runs:      ${NUM_RUNS}"
echo "  seed:          ${SEED}"
echo "  neutral_turns: ${NEUTRAL_TURNS} (only used by exp2)"
echo "  output_dir:    ${OUTPUT_DIR}"
echo "  transfer_bank: ${TRANSFER_BANK:-<legacy hardcoded pool>}"
echo "  runner_config: ${RUNNER_CONFIG:-<none, only required for exp3a/3b/3c>}"
echo "  overwrite:     ${OVERWRITE:-no (resume-by-default)}"
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

# Scale cost-per-call by model so the budget guardrail stays honest.
case "${MODEL}" in
  *opus*)    COST_PER_CALL="0.080" ;;  # rough mid-estimate
  *sonnet*)  COST_PER_CALL="0.015" ;;
  *haiku*)   COST_PER_CALL="0.003" ;;
  *)         COST_PER_CALL="0.015" ;;  # conservative default
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

uv run affect-battery pilot \
  --provider anthropic \
  --model "${MODEL}" \
  --experiment "${EXPERIMENT}" \
  --num-runs "${NUM_RUNS}" \
  --seed "${SEED}" \
  --neutral-turns "${NEUTRAL_TURNS}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-concurrent 4 \
  --rate-limit-rps 5 \
  --budget-max-calls 500 \
  --cost-per-call "${COST_PER_CALL}" \
  ${TRANSFER_BANK_FLAG} \
  ${RUNNER_CONFIG_FLAG} \
  ${OVERWRITE} \
  ${DRY_RUN} \
  ${PREREG_FLAG}

# ---- Analyze ----

echo ""
echo "Running analysis on pilot results..."
uv run affect-battery analyze \
  --results-dir "${OUTPUT_DIR}" \
  --model "${MODEL}"

echo ""
echo "Pilot complete. Reports under ${OUTPUT_DIR}/reports/"
echo "Top-level: ${OUTPUT_DIR}/reports/AGGREGATE_REPORT.md"
echo "Manifest:  ${OUTPUT_DIR}/manifest.yaml"
