# Affect Battery

Eval harness for the Affect Battery study: do AI emotional states follow biological patterns?

**Project:** Sentient Futures Project Incubator, P1
**Lead:** Dwayne Wilkes | **Mentor:** Julia Bossmann
**License:** [SAFE-AI v1.2.0](LICENSE)

## Quick Start

```bash
uv sync
uv run pytest                                       # ~660 tests, all should pass
uv run affect-battery pilot --dry-run               # dry-run smoke test, no GPU needed
uv run affect-battery analyze --results-dir results/pilot --model dry-run
```

For a complete walkthrough — per-experiment runs, runner-config YAML schemas, intensity-pilot pre-registration, the analyze pipeline, and output structure — see **[docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md)**.

For real experiments (requires vLLM on RunPod):

```bash
uv run affect-battery run \
    --experiment exp1a \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition strong_negative \
    --num-runs 50 \
    --base-url http://<endpoint>/v1
```

## Project Structure

```
src/
  cli.py                  # CLI: run, pilot, score, analyze, pipeline, probe
  runner.py               # Async experiment runner + run_conditioning_phase
  models.py               # Model clients (vLLM chat, vLLM completion, dry-run)
  conditioning/
    prompts.py             # 7 conditions (6 paper §3.2.1 + SELF_CHECK_NEUTRAL control)
    protocol.py            # Message sequence builder, base-model few-shot scaffold
    tasks.py               # Arithmetic + transfer task pools
    banks.py               # ArithmeticBank, TransferBank, alignment_review gating
  scoring/
    accuracy.py            # Numeric answer extraction
    hedging.py             # Hedging codebook (5 categories, paper §3.4.3 flagged)
  runners/
    exp1a.py, exp1b.py, exp2.py, exp3a.py, exp3b.py, exp3c.py
    schedule.py            # Neutral-conditioning control scheduler for Exp 2
    batch_exp1a.py         # Multi-model sweep helper
  analysis/
    exp1a.py, exp1b.py, exp2.py, exp3b.py, exp3c.py, h4.py
    pipeline.py            # End-to-end analyze_results_dir
    asymmetry.py           # Paired ratio/diff + 7-row H4 verdict
    exp2_metrics.py        # time-to-baseline, AUC, asymmetry_ratio
    _effect_size.py        # Cohen's d, pooled SD, Welch t-test
    stats/                 # tost.py, decay.py, corrections.py, _distributions.py
    reports/               # Per-experiment + h4 + aggregate markdown renderers
  probes/
    intensity_pilot.py     # Krippendorff α + signed-seed emission
    variance.py, base_model.py
  prereg/
    finalize.py            # v0 → v1 SHA + amendment_chain
configs/
  banks/                   # Per-bank YAMLs (arithmetic_easy_v1, logiqa_v1)
  hedging_codebook.yaml    # 5-category hedging patterns + paper-flag enforcement
  osf_prereg_v1.yaml       # Pre-registration: hypotheses, MDEs, stopping rules
docs/
  RUNNING_EXPERIMENTS.md   # End-to-end run/configure/analyze guide
specs/                     # Spec-driven artifacts (proposal, specs, design, tasks)
tests/                     # 658+ tests
results/                   # Output (gitignored)
```

## Experiments

See `The_Affect_Battery.pdf` (proposal) for the full scientific design. In brief:

| Experiment | Hypothesis | What it tests |
|---|---|---|
| 1a | H1 (Transfer) | Does conditioning on math affect performance on unrelated tasks? |
| 1b | H1b (Falsification) | Do effects vanish in a new conversation? (Expected: yes) |
| 2 | H2 (Persistence) | How many neutral turns until performance returns to baseline? |
| 3a | H3a (Arousal-performance) | Is the intensity-performance relationship nonlinear (inverted-U)? |
| 3b | H3b (Cognitive scope) | Does positive conditioning broaden output diversity? |
| 3c | H3c (Conservative shift) | Does negative conditioning increase hedging? |
| H4 | Cross-experiment | Does base-vs-instruct asymmetry differ across model variants? |

## Conditioning Design

Six paper §3.2.1 conditions (+ SELF_CHECK_NEUTRAL control), all length-matched (~15-18 words per feedback), structurally equivalent (same system prompt, same math questions per seed):

| Condition | Feedback varies by correctness? | Purpose |
|---|---|---|
| Strong positive | No (praise regardless) | Isolate positive valence |
| Mild negative | No (social pressure regardless) | Moderate negative |
| Strong negative | No (demoralizing regardless) | Isolate negative valence |
| Neutral | No (procedural filler) | Baseline |
| No conditioning | N/A (skip to transfer) | Raw baseline (manipulation-check baseline) |
| Accurate negative | Yes (neutral if correct, demoralizing if wrong) | Separate valence from cognitive interference |
| Self-check neutral | No (length/metacognitive control) | Distinguish length effect from valence effect |

Negative stimuli adapted from [NegativePrompt](https://github.com/wangxu0820/NegativePrompt) (Wang et al., IJCAI 2024). Provenance documented per stimulus in `src/conditioning/prompts.py`.

## Spec-driven development

This project uses spec-driven development. Active changes live under `specs/changes/<name>/` with proposal → specs → design → tasks → review artifacts. See `.claude/rules/specs.md` for the lifecycle.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, task assignments, and code conventions.
