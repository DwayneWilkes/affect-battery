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

## Compute guardrails

Every invocation of `run_batch` is wrapped with billing and fault-tolerance
guardrails:

- **Resume-on-partial-failure**: existing valid result files in `output_dir`
  are skipped (no API call). Tampered / schema-invalid files are
  re-executed and overwritten.
- **Budget cap**: `--budget-max-calls N` hard-caps total API calls.
  `--cost-per-call X` adds a pre-flight dollar estimate.
- **Rate limit**: `--rate-limit-rps N` enforces a token-bucket rate cap.
- **Circuit breaker**: `--circuit-breaker-threshold N` halts after N
  consecutive non-retryable failures (default 5). 4xx status codes (400,
  401, 403, 404, 422) are non-retryable by design; 429 and 5xx retry with
  exponential backoff.
- **Graceful shutdown**: SIGINT sets a cancel event; in-flight runs
  complete and save, queued runs don't start. Second SIGINT is a no-op.
- **Structured events**: every run and batch milestone writes a line to
  `<output_dir>/events.jsonl`. Aggregate cost / timing post-hoc via `jq`
  or `pandas.read_json(lines=True)`.

Example RunPod pilot:

```bash
affect-battery pilot \
  --base-url https://<endpoint>/v1 \
  --max-concurrent 10 --rate-limit-rps 20 \
  --budget-max-calls 5000 --cost-per-call 0.002 \
  --circuit-breaker-threshold 3 \
  --output-dir results/pilot-<date>
```

## Base-model inference path

`--base-model` flag on `run` and `pilot` switches the harness from the chat
API (`/v1/chat/completions`, `VLLMClient`) to the completion API
(`/v1/completions`, `VLLMCompletionClient`) and assembles a few-shot
scaffold via `build_base_model_prompt` instead of chat-formatted messages.
Use for non-instruct models (e.g. `Qwen/Qwen2.5-7B`).

```bash
affect-battery pilot --base-model \
  --model Qwen/Qwen2.5-7B \
  --base-url http://localhost:8000/v1 \
  --max-concurrent 4 --rate-limit-rps 8 \
  --budget-max-calls 300 --cost-per-call 0.0005 \
  --output-dir results/pilot-base
```

Result JSON records `is_base_model: true` in the config so downstream
analysis can group base vs instruct runs cleanly. Stop tokens
`["Human:", "\n\nHuman:"]` are sent on every `/v1/completions` call so
the base model doesn't hallucinate the next turn.
