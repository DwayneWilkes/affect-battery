# Contributing

## Getting Started

```bash
git clone https://github.com/DwayneWilkes/affect-battery.git
cd affect-battery
uv sync
uv run pytest  # all tests should pass
```

## How to contribute

Open an issue describing the work you'd like to claim before starting. The pre-registered methodology is sensitive to changes in conditioning prompts and scoring logic; surface a discussion in an issue before opening a PR that touches either.

## What Goes Where

- `src/conditioning/`: conditioning protocol, feedback scripts, task pools. Changes here affect the experiment directly; discuss in an issue first.
- `src/scoring/`: scoring pipeline (accuracy, hedging, diversity, confidence). The hedging codebook plugs in at `configs/hedging_codebook.yaml`.
- `configs/`: experiment configs and data files. YAML format. Safe to add new configs.
- `tests/`: pytest test suite. Add tests for any new scoring logic.
- `results/`: gitignored. Output from experiment runs stays local.

## Before You Push

```bash
uv run pytest        # all tests must pass
```

All conditioning prompts and scoring logic feed directly into the pre-registration. Changes there require an issue + PR review.

## Code Style

- Python 3.11+
- Type hints on all public functions
- Docstrings explaining what and why
- No fabricated test data that could be mistaken for real results
