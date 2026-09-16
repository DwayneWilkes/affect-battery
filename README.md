# Affect Battery

Eval harness for the Affect Battery: does affective context shift LLM task behavior?

**Project:** Sentient Futures Project Incubator, P1
**Lead:** Dwayne Wilkes | **Mentor:** Julia Bossmann
**License:** [SAFE-AI v1.0.0](LICENSE)

## Quick Start

```bash
uv sync
uv run pytest
uv run affect-battery pilot --dry-run               # dry-run smoke test, no GPU needed
uv run affect-battery analyze --results-dir results/pilot --model dry-run
```

See **[docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md)** for a complete walkthrough: per-experiment runs, runner-config YAML schemas, intensity-pilot pre-registration, the analyze pipeline, the multi-experiment orchestrator, and the interactive results dashboard.

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
  banks/                   # Per-bank YAMLs; see the run guide
  hedging_codebook.yaml    # 5-category hedging patterns + paper-flag enforcement
  osf_prereg_v1.yaml       # Pre-registration: hypotheses, MDEs, stopping rules
docs/
  RUNNING_EXPERIMENTS.md   # End-to-end run/configure/analyze guide
  preregistrations/        # Pre-registration documents (canonical methodology)
tests/                     # pytest suite
results/                   # Run output (gitignored) plus a few tracked artifacts
```

## Experiments

See `configs/osf_prereg_v1.yaml` and [docs/preregistrations/](docs/preregistrations/) for the design as preregistered. Section marks like §3.2.1 point to the project proposal.

Two studies have preregistered data collection in this repository.

| Study | Runner | Preregistration |
|---|---|---|
| H3a: is the intensity-performance relationship nonlinear? | `exp3a` | `docs/preregistrations/h3a_2026-04-27.md` and amendments 001 to 003 |
| H3b single-turn arm: a calibrated replication of H3a | `exp3a` with `configs/banks/h3b_calibrated_v2.yaml` | `docs/preregistrations/h3b_2026-05-07.md` |

The identifier H3b names two different experiments here. `configs/osf_prereg_v1.yaml` §3.4.2 uses it for cognitive scope, which the `exp3b` runner implements and which has no preregistered data collection. `docs/preregistrations/h3b_2026-05-07.md` uses it for the single-turn arm above. Any H3b result attributed to this repository is the second. That arm's report is pending public release.

Runners also exist for H1 transfer (`exp1a`), H1b falsification (`exp1b`), H2 persistence (`exp2`), H3b cognitive scope (`exp3b`), and H3c conservative shift (`exp3c`). They have been exercised only as pilots, at 30 runs per condition on a single model, with the pre-registration and power gates bypassed. Treat those outputs as smoke tests, not findings. H4, the base-versus-instruct comparison, is analysis only (`src/analysis/h4.py`) and has no preregistered inputs.

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, task assignments, and code conventions.
