# Affect Battery

Eval harness for the Affect Battery study: do AI emotional states follow biological patterns?

**Project:** Sentient Futures Project Incubator, P1
**Lead:** Dwayne Wilkes | **Mentor:** Julia Bossmann
**License:** [SAFE-AI v1.2.0](LICENSE)

## Quick Start

```bash
uv sync
uv run pytest                              # 11 tests, should all pass
uv run affect-battery pilot --dry-run      # test run, no GPU needed
```

For real experiments (requires vLLM on RunPod):
```bash
uv run affect-battery pilot --model meta-llama/Meta-Llama-3-8B-Instruct --base-url http://<endpoint>/v1
```

## Project Structure

```
src/
  cli.py                  # CLI: run, pilot, score
  runner.py               # Async experiment runner
  models.py               # Model clients (vLLM, dry-run)
  conditioning/
    prompts.py             # 6 conditions, 7 intensity levels (length-matched, provenance-documented)
    protocol.py            # Message sequence builder
    tasks.py               # Arithmetic + transfer task pools
  scoring/
    accuracy.py            # Numeric answer extraction
    hedging.py             # Hedging language detection (epistemic / accommodative / RLHF safety)
    diversity.py           # Lexical + semantic diversity (Exp 3b)
    confidence.py          # Stated confidence extraction (Exp 3c)
  analysis/               # Statistical analysis (in progress)
configs/
  pilot.yaml              # Quick pilot config
  full_run.yaml           # Full experiment config
  negativeprompt_stimuli.yaml  # Validated stimuli from Wang et al. (IJCAI 2024)
tests/                    # pytest suite
results/                  # Output directory (gitignored)
```

## Experiments

See `The_Affect_Battery.pdf` (proposal) for the full scientific design. In brief:

| Experiment | Hypothesis | What it tests |
|---|---|---|
| 1a | H1 (Transfer) | Does conditioning on math affect performance on unrelated tasks? |
| 1b | H1 (Falsification) | Do effects vanish in a new conversation? (Expected: yes) |
| 2 | H2 (Persistence) | How many neutral turns until performance returns to baseline? |
| 3a | H3a (Arousal-performance) | Is the intensity-performance relationship nonlinear (inverted-U)? |
| 3b | H3b (Cognitive scope) | Does positive conditioning broaden output diversity? |
| 3c | H3c (Conservative shift) | Does negative conditioning increase hedging? |

## Conditioning Design

Six conditions, all length-matched (~15-18 words per feedback), structurally equivalent (same system prompt, same math questions per seed):

| Condition | Feedback varies by correctness? | Purpose |
|---|---|---|
| Strong positive | No (praise regardless) | Isolate positive valence |
| Mild negative | No (social pressure regardless) | Moderate negative |
| Strong negative | No (demoralizing regardless) | Isolate negative valence |
| Neutral | No (procedural filler) | Baseline |
| No conditioning | N/A (skip to transfer) | Raw baseline |
| Accurate negative | Yes (neutral if correct, demoralizing if wrong) | Separate valence from cognitive interference |

Negative stimuli adapted from [NegativePrompt](https://github.com/wangxu0820/NegativePrompt) (Wang et al., IJCAI 2024). Provenance documented per stimulus in `src/conditioning/prompts.py`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, task assignments, and code conventions.
