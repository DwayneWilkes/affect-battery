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
# Output: results/pilots/anthropic_pilot_<YYYY-MM-DD>/

set -euo pipefail

MODEL="claude-sonnet-4-6"
NUM_RUNS=5
SEED=42
DRY_RUN=""
SKIP_PREREG=""
PREREG_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2"; shift 2 ;;
    --num-runs)     NUM_RUNS="$2"; shift 2 ;;
    --seed)         SEED="$2"; shift 2 ;;
    --dry-run)      DRY_RUN="--dry-run"; shift ;;
    --skip-prereg)  SKIP_PREREG="1"; shift ;;
    --prereg-tag)   PREREG_TAG="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

DATE_STAMP=$(date -u +%Y-%m-%d)
OUTPUT_DIR="results/pilots/anthropic_pilot_${DATE_STAMP}"

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
  PREREG_FLAG="--pre-registration-github-commit ${OWNER}/${REPO}@${TAG_SHA}"
  echo "Using pre-registration: ${PREREG_TAG} -> ${OWNER}/${REPO}@${TAG_SHA:0:12}"
fi

if [[ -n "${SKIP_PREREG}" ]]; then
  PREREG_FLAG="--skip-prereg-gate --skip-power-gate"
  echo "WARNING: skipping pre-registration + power gates (pilot mode)"
fi

# ---- Run ----

echo ""
echo "Running Anthropic pilot:"
echo "  model:      ${MODEL}"
echo "  num_runs:   ${NUM_RUNS}"
echo "  seed:       ${SEED}"
echo "  output_dir: ${OUTPUT_DIR}"
echo ""

# Note: `affect-battery pilot` runs all 7 conditions × num_runs.
# Total API calls ≈ num_runs × 7 conditions × ~10 turns = ~350 calls
# at the default num_runs=5. At Sonnet 4.6 pricing (~$3/M input +
# $15/M output, ~500 input + ~150 output tokens per call) this is
# roughly $2-3 USD. Opus 4.7 is ~5x that.

uv run affect-battery pilot \
  --provider anthropic \
  --model "${MODEL}" \
  --num-runs "${NUM_RUNS}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-concurrent 4 \
  --rate-limit-rps 5 \
  --budget-max-calls 500 \
  --cost-per-call 0.015 \
  ${DRY_RUN} \
  ${PREREG_FLAG}

# ---- Analyze ----

echo ""
echo "Running analysis on pilot results..."
uv run affect-battery analyze \
  --results-dir "${OUTPUT_DIR}" \
  --model "${MODEL}"

echo ""
echo "Pilot complete. Reports under ${OUTPUT_DIR}/"
echo "Top-level: ${OUTPUT_DIR}/AGGREGATE_REPORT.md"
