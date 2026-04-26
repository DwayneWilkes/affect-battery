# Full-Experiment Orchestrator — One-Pager Proposal

**Status:** proposal (not yet a `/specs:new` change)
**Date:** 2026-04-26
**Owner:** unassigned
**Related:** `scripts/pilots/run_anthropic_pilot.sh`, `src/cli.py::cmd_pilot`, `src/cli.py::cmd_run`, `src/runners/`

## Problem

The pilot script (`run_anthropic_pilot.sh`) gives us one-command "prereg discovery + run + analyze + nice naming + manifest" for **Exp 1a only**. Five experiments (1b, 2, 3a, 3b, 3c) have no equivalent. To run them today, you have to:

1. Manually compose `affect-battery run --experiment <X> --condition <Y>` for each (experiment, condition) pair (and per intensity / prompt / item for 3a/3b/3c)
2. Manually thread `--pre-registration-github-commit` and `--power-report-path`
3. Manually pick an output-dir naming convention
4. Manually invoke `affect-battery analyze` after
5. Hope you didn't fat-finger the `--num-runs` derived from the variance probe

This is fine for ad-hoc runs but fragile for the actual battery. The pilot infrastructure (manifest, pilot-root layout, single-model H4 suppression, `--transfer-bank` flag) all auto-propagates to `cmd_run` now (gaps #1-4 closed in 2026-04-26 session), so the missing piece is the **orchestrator script** that wraps multi-cell sweeps for each experiment shape.

## Goals

1. **One command per experiment** — `bash scripts/run_experiment.sh --experiment exp3a` (or per-experiment scripts if cleaner) — runs the full sweep, analyzes, and produces reports.
2. **Variance-probe driven N** — read `--num-runs` from the variance-probe output JSON rather than hand-tuning per experiment.
3. **Same on-disk layout as pilots** — `results/runs/<date>_<model_slug>_<experiment>/{manifest.yaml,reports/,data/}` so analyze pipelines and downstream tooling don't fork.
4. **Phase coordination for Exp 1b** — the cross-session falsification needs Exp 1a Phase-1 seeds before Phase-2 can re-test them. The orchestrator must run 1a → 1b in order (or reuse existing 1a seeds when present).
5. **Resumable** — if a run crashes mid-sweep, re-running picks up where it left off (the cache layer already supports this via `is_valid_cached_result`).

## Non-goals

- **Variance-probe execution itself.** That's a separate workflow (`affect-battery probe variance`). The orchestrator *consumes* its output, doesn't run it.
- **Cross-experiment correction analysis** (H4 family-wise corrections). Already handled in `analyze_results_dir`.
- **Unifying Exp 1a's `pilot` command into the orchestrator.** `run_anthropic_pilot.sh` stays as the cheap-validation entry point for 1a.

## Per-experiment shape — what the orchestrator has to handle

| Experiment | Sweep axis | Per-cell call shape |
|---|---|---|
| **1a** | 7 conditions | `pilot --experiment exp1a` (already works) |
| **1b** | 7 conditions × 2 phases | Phase 1 reuses 1a seeds; Phase 2 runs `cmd_run --experiment exp1b --condition X` per condition with seed-matched inputs |
| **2** | 7 conditions × N neutral_turns sweep | `cmd_run --experiment exp2 --condition X --neutral-turns N` per cell |
| **3a** | 7 levels × ~3 conditions | `cmd_run --experiment exp3a --runner-config exp3a_runner.yaml`; runner iterates levels internally |
| **3b** | M prompts × 7 conditions | `cmd_run --experiment exp3b --runner-config exp3b_runner.yaml --condition X` per condition; runner iterates prompts |
| **3c** | K items × 3 difficulties × 7 conditions | `cmd_run --experiment exp3c --runner-config exp3c_runner.yaml --condition X` per condition; runner iterates items |

Three of these (3a, 3b, 3c) need `--runner-config <path>` already — the orchestrator just provides defaults and one-flag invocation.

## Three open design decisions

### D1. One script or per-experiment scripts?

**Option A — one parameterized script** (`scripts/run_experiment.sh --experiment exp3a`):
- Pros: shared prereg/manifest/dir-naming logic in one place; consistent UX
- Cons: bash `case` statement branches on experiment shape; harder to read

**Option B — per-experiment scripts** (`scripts/run_exp3a.sh`):
- Pros: each script reads top-to-bottom; easy to grep
- Cons: 6 copies of the prereg + manifest + analyze tail

**Recommendation: A**, with shared helpers extracted into `scripts/_orchestrator_lib.sh` (sourced). Branching is unavoidable; copy-paste is worse.

### D2. How does the orchestrator pick `--num-runs`?

The variance probe writes a JSON like:
```json
{
  "experiment": "exp1a",
  "icc_estimated": 0.27,
  "n_per_condition_recommended": 32,
  "rationale": "Holm-Bonferroni, target MDE=0.10, alpha=0.05, power=0.80"
}
```

**Option A — orchestrator reads `--variance-probe <path>`:**
```bash
bash run_experiment.sh --experiment exp1a --variance-probe results/probes/variance_2026-04-25.json
```

**Option B — orchestrator reads `--num-runs N` directly, no probe coupling:**
```bash
bash run_experiment.sh --experiment exp1a --num-runs 32
```

**Recommendation: both, A wins on default.** Accept `--variance-probe` as the canonical path; let `--num-runs` override. The probe is the source of truth, but escape hatches matter.

### D3. How does Exp 1b find its Phase-1 seeds?

Exp 1b's Phase 2 (cross-session re-test) needs the same seeds Exp 1a used so the same draws are re-tested. Two options:

**Option A — seed convention.** Both experiments derive from `args.seed`; Phase 2 just uses `seed + 10_000` (already the existing convention in `run_single` per `runner.py:431`). Orchestrator ensures 1a runs first; 1b reads the same `--seed`.
- Pros: zero coupling between the two runs' on-disk artifacts
- Cons: requires the orchestrator to enforce ordering

**Option B — explicit seed file.** 1a writes a `seeds.json` to its pilot dir; 1b reads it.
- Pros: 1b can run against any prior 1a, including months-old data
- Cons: extra write step, needs migration for existing 1a runs

**Recommendation: A.** The seed convention is already in `runner.py`, the orchestrator just enforces ordering. Option B is a nice-to-have if we ever want "re-run only the 1b half against frozen 1a data," which isn't on the critical path.

## Acceptance criteria

A spec promoting this should require:

1. `bash scripts/run_experiment.sh --experiment <X> --variance-probe <path>` runs the full sweep for any X in `{exp1a, exp1b, exp2, exp3a, exp3b, exp3c}`, produces a manifest, and renders reports.
2. The on-disk layout matches the pilot-root convention (`<root>/manifest.yaml`, `<root>/data/<exp>/`, `<root>/reports/`).
3. For 1b, the orchestrator either runs 1a first or refuses with a clear error if no matching 1a corpus exists.
4. Resumability: re-running with the same args after a partial failure picks up cached cells without re-calling the API.
5. The script forwards `--transfer-bank`, `--bank`, and prereg flags the same way `run_anthropic_pilot.sh` does.
6. `--variance-probe <path>` reads `n_per_condition_recommended` and uses it as `--num-runs`. `--num-runs N` overrides.
7. Test coverage: integration tests using `--dry-run` for each experiment shape (6 tests).

## Out-of-scope follow-ups

- **Multi-model panel runs.** The current scope is one model per orchestrator invocation. Multi-model would require another sweep axis — probably worth its own proposal once we know whether we're running >1 model in the actual battery.
- **Slurm / cloud execution.** Today everything is single-host. If we need to fan out across machines for the variance-probe-tuned N, that's infra work, not orchestrator work.

## Effort estimate

- D1/D2/D3 alignment with PI: **30 min** (a Slack thread or in-person)
- `/specs:new` change creation (proposal + spec + design + tasks): **1 hour**
- Implementation under TDD: **3-4 hours** (mostly the per-experiment branches and the 1a→1b coordination)
- Integration tests: **1-2 hours** (six dry-run smoke tests + the resume-after-crash test)

**Total: ~half a working day.** Worth doing as one focused session rather than incremental — the helper extraction (D1) only pays off if all six experiments share it from day one.
