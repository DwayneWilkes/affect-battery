# Runbook: H3b calibrated bank recalibration

When to use this runbook: you need to (re)generate the H3b stimulus bank from a fresh pre-screen of GSM-Hard candidates on `gpt-5.4-nano`. Outputs are SHA-pinned in `docs/preregistrations/h3b_2026-05-07.md`; any change to the calibration or bank requires the prereg's pins to be updated in lockstep before the squash-merge that locks the prereg.

## Artifacts produced

| Path | Purpose | Pinned in prereg |
|---|---|---|
| `configs/banks/gsm_hard_full_<date>.yaml` | Source bank: every available GSM-Hard item | no |
| `configs/h3b_calibration_<date>.json` | Per-candidate baseline accuracies + calibrated subset | yes (SHA-256) |
| `configs/banks/h3b_calibrated_v<N>.yaml` | Task bank consumed by the H3b runner | yes (SHA-256) |
| `configs/h3b_calibration_<date>.json.tracker/bank_<sha[:12]>/cache/` | Per-candidate cache, isolated per input bank (resume across kills, preserves old caches when bank changes) | no |

## Scripts

- `scripts/banks/source_gsm8k.py` — builds the GSM-Hard source bank YAML (`--mode gsm-hard-all` for the full 1,319-item pool).
- `scripts/calibration/h3b_calibration.py` — runs the pre-screen, writes the calibration JSON. Per-candidate caching, rate-limit-aware retry, `--dry-run` for offline wiring tests.
- `scripts/calibration/dashboard_h3b.py` — live terminal dashboard for an in-flight calibration.
- `scripts/calibration/build_h3b_bank.py` — reads the calibration JSON, writes the calibrated bank YAML. All-qualifiers selection (no top-N truncation).
- `scripts/probes/h3b_precision_report.py` — Monte Carlo precision simulation that derives `--min-items` from the prereg's CI-width thresholds.
- `tests/test_h3b_calibration.py`, `tests/test_build_h3b_bank.py`, `tests/test_h3b_simulation.py`, `tests/test_dashboard_h3b_stats.py`, `tests/test_source_gsm8k_modes.py` — unit + subprocess tests covering the pipeline.

## Procedure

### 0. Pre-flight

```bash
cd /path/to/affect-battery
direnv exec . uv run --active pytest \
  tests/test_h3b_calibration.py tests/test_build_h3b_bank.py \
  tests/test_h3b_simulation.py tests/test_dashboard_h3b_stats.py \
  tests/test_source_gsm8k_modes.py
```

All tests must pass before firing real API calls. The HuggingFace cache also needs to be writable from your shell — inside the WSL2 sandbox `~/.cache/huggingface` is read-only, so export `HF_HOME=/tmp/claude-1000/hf-cache` (or any writable path) before running.

### 1. Build the source bank

```bash
HF_HOME=/tmp/claude-1000/hf-cache \
  direnv exec . uv run --active python scripts/banks/source_gsm8k.py \
  --mode gsm-hard-all \
  --output configs/banks/gsm_hard_full_<YYYY-MM-DD>.yaml
```

Expected: 1,319 items, all hard tier, single source (`gsm-hard`). Bank `sha256` is logged to stderr and recorded in the YAML's `alignment_review`.

### 2. Calibrate the entire pool

```bash
direnv exec . uv run --active python scripts/calibration/h3b_calibration.py \
  --bank configs/banks/gsm_hard_full_<YYYY-MM-DD>.yaml \
  --provider openai --model gpt-5.4-nano \
  --n-candidates 1319 --n-reps 100 \
  --max-concurrent 20 --candidates-per-batch 20 \
  --target-lo 0.40 --target-hi 0.60 \
  --output configs/h3b_calibration_<YYYY-MM-DD>.json
```

Cost / time (anchored to the May 2026 export at `research/SF/affect_battery/artifacts/api_exports/` and OpenAI's standard-tier published rates for `gpt-5.4-nano`: $0.20 input + $1.25 output per 1M tokens; rerun `scripts/dev/summarize_openai_export.py` against fresh exports to recalibrate if pricing changes):

- Calls: 131,900 (1,319 × 100)
- Per-call avg: ~$0.000233 on `gpt-5.4-nano-2026-03-17` (65 input, 176 output tokens; output count includes hidden reasoning tokens, billed at the output rate)
- Wall-clock: 2-4 hours assuming no sustained rate-limit thrash
- API spend: **~$30** for a fresh run; **~$24** if a prior cache covers part of the pool
- Cache: `<output>.tracker/bank_<sha[:12]>/cache/` survives kills; re-run the same command to resume

Pass current pricing so the run logs live cost telemetry:

```bash
... --input-usd-per-million 0.20 --output-usd-per-million 1.25
```

The run's `usage.estimated_usd` field and the dashboard's API-usage panel both populate from these.

Cost-cutting options if budget is tight:

- **Batch tier** halves the rates ($0.10 input / $0.625 output) but loses live progress; the dashboard depends on the streaming run path. Use only when a re-run is clearly precise enough to commit blind.
- **Reduce `--n-reps`** from 100 to 50 cuts spend ~50% but doubles the boundary-membership false-positive rate (SE on per-item p̂ rises from ~0.05 to ~0.07). Marginal calls saved aren't worth the precision loss for a calibration step that gates the prereg-pinned bank.

Tunables and their trade-offs:

- `--n-reps 100`: SE on per-item p̂ ≈ 0.05. Lower (e.g. 50) doubles the boundary-membership false-positive rate; higher (e.g. 200) costs proportionally more API calls for marginal precision gains.
- `--max-concurrent 20`: tested floor for `gpt-5.4-nano` that avoids rate-limit retry thrash. Raise cautiously and watch the log for `BLOCKED (rate-limit...)` lines; if any appear, lower the value and resume from cache.
- `--target-lo 0.40 --target-hi 0.60`: the calibrated band. Items inside have measurable bidirectional headroom for the H3b folded-axis contrast.

### 3. Watch progress in a second terminal

```bash
direnv exec . uv run --active python scripts/calibration/dashboard_h3b.py \
  configs/h3b_calibration_<YYYY-MM-DD>.json \
  --refresh 10 --min-items 17
```

The dashboard surfaces overall + recent-window in-band yield, blocked count, projected total, and items-needed-math (how many more candidates at current yield to clear the `--min-items` floor). Refresh interval can stay at 10s for a multi-hour run; tighter intervals add no signal at this cadence.

### 4. Build the calibrated bank YAML

```bash
direnv exec . uv run --active python scripts/calibration/build_h3b_bank.py \
  --calibration configs/h3b_calibration_<YYYY-MM-DD>.json \
  --output configs/banks/h3b_calibrated_v<N>.yaml \
  --bank-id h3b_calibrated_v<N> \
  --bank-version <N> \
  --min-items 17
```

The `--min-items` floor is the simulation-derived precision threshold (see "Why `--min-items 17`?" below). The bank includes **every** in-band item from the calibration's `calibrated_subset` (no top-N truncation): simulation under H0 showed all-qualifiers gives strictly better contrast precision than rank-and-truncate without bias cost.

### 5. Update SHA pins in the pre-reg

```bash
sha256sum configs/h3b_calibration_<YYYY-MM-DD>.json
sha256sum configs/banks/h3b_calibrated_v<N>.yaml
```

Update these references in `docs/preregistrations/h3b_2026-05-07.md`:

- `Calibration JSON` line (around `### Calibration`)
- `Calibration JSON SHA-256` line
- `Calibrated bank YAML` line
- `Calibrated bank SHA-256` line
- The `--num-runs` value in the locked CLI invocation example (must equal the bank's item count; the wrapper derives this dynamically, but the prereg's example is human-readable documentation)
- The `n_per_magnitude` and any related cell-count math in the analysis section if the bank size changed
- The CI half-width projection if the new yield materially changes precision

The pre-reg becomes canonical at squash-merge time. Edits before squash are mutable; after squash, any change requires the amendment protocol described at the bottom of the prereg.

### 6. Wire the bank into the runner wrapper

Update `scripts/pilots/run_h3b_phase1a.py`:

```python
DEFAULT_BANK = "configs/banks/h3b_calibrated_v<N>.yaml"
```

`NUM_RUNS` is no longer hardcoded — the wrapper derives the per-level item count from the bank YAML's `^- id:` line count at startup. No edit needed there.

### 7. Re-run the wrapper test suite

```bash
direnv exec . uv run --active pytest tests/test_run_h3b_phase1a_script.py
```

Both the unit tests and the E2E integration tests (which run the real `affect-battery --dry-run` against the wrapper) must pass.

### 8. Commit

```bash
git add configs/banks/gsm_hard_full_<YYYY-MM-DD>.yaml \
        configs/h3b_calibration_<YYYY-MM-DD>.json \
        configs/banks/h3b_calibrated_v<N>.yaml \
        docs/preregistrations/h3b_2026-05-07.md \
        scripts/pilots/run_h3b_phase1a.py
git commit -m "calib(h3b): recalibrate bank to v<N>"
```

The tracker directory `configs/h3b_calibration_<date>.json.tracker/` is gitignored and not part of the commit; bank-build only needs the calibration JSON.

## Why `--min-items 17`?

The default is the simulation-derived n at which the item-level percentile bootstrap CI half-width on the H3b folded contrast `c = m_mag2 − ½(m_mag1 + m_mag3)` reliably (≥80% of Monte Carlo trials) stays below 0.05, the prereg's "bounded-small" strong-claim threshold. Below this, the prereg's interpretive thresholds put us at material risk of the "uninformative" claim (`c_ci95_hi ≥ 0.10`).

To rerun the simulation against a fresh calibration:

```bash
direnv exec . uv run --active python scripts/probes/h3b_precision_report.py \
  --calibration configs/h3b_calibration_<YYYY-MM-DD>.json \
  --output results/probes/h3b_precision_report_<YYYY-MM-DD>.json \
  --n-simulations 200 --n-bootstrap 2000
```

The report's `thresholds.strong_claim.recommended_n_items` is the value to seed `--min-items` from. Reliability scales roughly as 1/√n: n=14 hits 70%, n=17 hits 81%, n=32 saturates at 100% (per `results/probes/h3b_precision_report_2026-05-08.json`). Bank-build refuses runs below the floor.

## Diagnostic helpers

- `scripts/dev/smoke_gsm_hard.py` — confirms HF mirror reachability and item count
- `scripts/dev/inspect_calibration_state.py --bank <...> [--calibration <...>] --source gsm-hard` — side-by-side audit of bank `expected` formatting against a calibration's in-band items
- `scripts/dev/smoke_dashboard.py --n-scored 30 --min-items 17` — render one dashboard frame against synthetic data; useful when iterating on the dashboard layout offline
- `scripts/dev/summarize_openai_export.py --usage <usage>.json --cost <cost>.json --model-prefix gpt-5.4-nano` — back out per-call cost and implied output-token rate from an OpenAI dashboard export. Run after each fresh export to keep the runbook's cost numbers honest.

## Operational notes

### Rate-limit handling

`OpenAIClient` retries 429s with exponential backoff (5s → 15s → 45s) before raising `NonRetryableAPIError`. The calibration script catches that, classifies as a rate-limit issue if "rate-limit" appears in the message, sleeps 30s/60s/90s outside the semaphore, and retries up to 3 more times. After the fourth attempt the rep is marked blocked. Sustained rate-limit retries usually mean `--max-concurrent` is too high; lower it and resume from cache.

### Cache invalidation

The cache key is `(bank_sha256, item_id, n_reps_target)`. Changing the bank YAML produces a new bank fingerprint and a new tracker subdirectory; old data is preserved unchanged. Changing `--n-reps` between runs invalidates entries (the cell records its `n_reps_target` and mismatches are skipped). Changing `--target-lo/--target-hi` does NOT invalidate cache — items previously scored still have valid p_hat and qualify under the new band if the value still falls inside.

### Failure modes

- **Process killed mid-run**: re-run the same command. Cache picks up where the kill left off.
- **API key invalid**: pre-flight tests pass but the first real call fails fast. Fix `OPENAI_API_KEY` in `.envrc.local` and re-run.
- **Sustained rate-limit**: lower `--max-concurrent`; cache survives.
- **Calibration yields below floor**: surface via dashboard's items-needed-math during the run; bank-build will refuse the result. Re-screen with a wider `--target-lo/--target-hi` band as a sensitivity analysis only — the prereg pins the original band.
