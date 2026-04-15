# Contributing

## Getting Started

```bash
git clone https://github.com/DwayneWilkes/affect-battery.git
cd affect-battery
uv sync
uv run pytest  # should see 11 passing
```

## Task Assignments

See the [Open Task Pool](https://docs.google.com/document/d/1GuWgAhYHXs955oe1nt2M_2YXtrEzjx3sGDDE3sH3XUU) for available work. Claim tasks in Slack before starting.

## What Goes Where

- `src/conditioning/` - Conditioning protocol, feedback scripts, task pools. Changes here affect the experiment directly. Coordinate before editing.
- `src/scoring/` - Scoring pipeline (accuracy, hedging, diversity, confidence). Akshansh's hedging codebook (Ticket 6) plugs in at `configs/hedging_codebook.yaml`.
- `configs/` - Experiment configs and data files. YAML format. Safe to add new configs.
- `tests/` - pytest test suite. Add tests for any new scoring logic.
- `results/` - gitignored. Output from experiment runs stays local.

## Before You Push

```bash
uv run pytest        # all tests must pass
```

Do not push to main without confirming tests pass. Do not modify conditioning prompts or scoring logic without discussing first: these feed directly into the OSF pre-registration.

## Code Style

- Python 3.11+
- Type hints on all public functions
- Docstrings explaining what and why
- No fabricated test data that could be mistaken for real results
