# Affect Battery Eval Harness

## Build & Test
```bash
uv sync
uv run pytest
uv run python -m src.cli --help
```

## Conventions
- Python 3.11+, uv for package management
- pytest for tests, pytest-asyncio for async tests
- Type hints and docstrings on all public functions
- Results saved as JSON with SHA-256 checksums
- Config-driven: YAML configs in configs/
- No fabricated data or results
