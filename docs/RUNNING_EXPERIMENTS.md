# Running Experiments

End-to-end guide for going from `git pull` to a populated set of per-experiment reports + an aggregate cross-experiment landing page.

## Prerequisites

- `uv sync` to install Python deps (Python 3.12+).
- For real runs: a vLLM endpoint (RunPod or local) serving the four paper §3.1 models. For dry-run smoke tests no GPU is needed.
- Optional: `--runner-config` YAML files for Exp 3a / 3b / 3c (schemas below).

## Workflow at a glance

```
1. Probes (Week 0)         → variance MDE + base-model feasibility
2. Intensity pilot         → Krippendorff α, emit pre-registered seed
3. Manipulation check      → confirm conditioning fires per model
4. Per-experiment runs     → exp1a, exp1b, exp2, exp3a, exp3b, exp3c
5. Analyze                 → seven markdown reports under results/
```

Phase 1-3 are gates. Phase 4 produces result JSONs. Phase 5 stitches them into reports.

## Smoke test (no GPU)

Verify the harness end-to-end with a fake model:

```bash
uv run pytest                                                           # 658 tests pass
uv run affect-battery pilot --dry-run                                   # 5 runs × 7 conditions, results/pilot/
uv run affect-battery analyze --results-dir results/pilot --model dry-run
```

The `analyze` step always produces `results/pilot/AGGREGATE_REPORT.md`. Per-experiment reports only appear when matching corpora exist.

## Real runs (per experiment)

Common flags:
- `--model <name>` (paper §3.1 model, or `gpt-5` / `claude-opus-4-7` etc. when using API providers)
- `--provider {vllm,openai,anthropic}` (default: `vllm`). `openai` and `anthropic` route through their official SDKs; set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in env.
- `--base-url http://<endpoint>/v1` — only relevant for `--provider vllm`
- `--base-model` for the Llama-3-8B base (non-instruct) inference path; uses `/v1/completions` + few-shot scaffold instead of chat. **Not supported with `--provider openai` or `--provider anthropic`** (no raw-completion endpoint on those APIs).
- `--bank arithmetic_easy_v1` (default) or `logiqa_v1` (status: candidate, currently excluded from primary).
- `--seed 42`, `--num-runs 50`, `--temperature 0.7` (defaults match the spec).
- Output goes to `--output-dir results/` by default; per-experiment subdirs are auto-created.

### Exp 1a — within-session transfer (H1)

No extra config needed. Uses `run_batch`; budget / rate-limit / cancel work.

```bash
# vLLM (open-source paper §3.1 panel)
uv run affect-battery run \
    --experiment exp1a \
    --provider vllm \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition strong_negative \
    --num-runs 50 \
    --base-url http://<endpoint>/v1 \
    --output-dir results/exp1a

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-... uv run affect-battery run \
    --experiment exp1a \
    --provider anthropic \
    --model claude-opus-4-7 \
    --condition strong_negative \
    --num-runs 50 \
    --output-dir results/exp1a

# OpenAI
OPENAI_API_KEY=sk-... uv run affect-battery run \
    --experiment exp1a \
    --provider openai \
    --model gpt-5 \
    --condition strong_negative \
    --num-runs 50 \
    --output-dir results/exp1a
```

**Provider notes:**
- `--provider vllm` (default) uses the local/RunPod OpenAI-compatible endpoint at `--base-url`. This is the only provider that supports `--base-model`.
- `--provider openai` uses `openai.AsyncOpenAI` with the `OPENAI_API_KEY` env var. Chat models only.
- `--provider anthropic` uses `anthropic.AsyncAnthropic` with `ANTHROPIC_API_KEY`. Chat models only; the system prompt is automatically extracted from the messages array.
- API model version drift: record the exact model string (e.g., `claude-opus-4-7-2026-04-15`) in your run config so analysis can detect cross-run drift.

**H4 caveat:** the base-vs-instruct asymmetry test requires a base (non-RLHF'd) checkpoint, which only the vLLM Llama-3-8B base provides. When running the API providers, you can either keep that vLLM checkpoint as the H4 anchor, reframe H4 as a cross-provider contrast, or drop H4 (see `configs/osf_prereg_v1.yaml::amendment_chain` entry 2).

Repeat per condition × per model. See `src/runners/batch_exp1a.py::run_exp1a_batch` for the model-sweep helper.

### Exp 1b — cross-session falsification (H1b)

Same shape as Exp 1a; the runner overrides the system prompt with `CROSS_SESSION_SYSTEM_PROMPT` automatically and records distinct `session_1_seed` / `session_2_seed` on the body.

```bash
uv run affect-battery run \
    --experiment exp1b \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition strong_negative \
    --num-runs 50 \
    --base-url http://<endpoint>/v1 \
    --output-dir results/exp1b
```

### Exp 2 — persistence dynamics (H2)

Iterates the N-values sweep; pass `--neutral-turns N` per N value. Produces `Exp2Body` with `n_value` + per-turn accuracies.

```bash
for N in 1 3 5 10; do
  uv run affect-battery run \
      --experiment exp2 \
      --model meta-llama/Meta-Llama-3-8B-Instruct \
      --condition strong_negative \
      --num-runs 50 \
      --neutral-turns "$N" \
      --base-url http://<endpoint>/v1 \
      --output-dir "results/exp2/n_${N}"
done
```

Pair with neutral-conditioning controls (see `src/runners/schedule.py::schedule_exp2_with_controls`).

### Exp 3a — inverted-U intensity (H3a)

**Gated** by the intensity-axis Krippendorff pilot. You MUST run the pilot and emit a signed seed before Exp 3a will start; the runner re-validates the SHA at every invocation and refuses to run if the seed has been touched.

Pilot first (run from a Python shell or notebook with three rater dicts):

```python
from src.probes.intensity_pilot import run_intensity_pilot, emit_seed

ratings = {
    "rater_1": [...],   # one rating per (item, level) cell
    "rater_2": [...],
    "rater_3": [...],
}
result = run_intensity_pilot(ratings)
assert result["decision"] == "proceed"  # else collapse / restructure
emit_seed(
    result,
    axis_id="primary_valence_axis",
    n_levels=7,
    pilot_date="2026-04-25",
    output_path="configs/intensity_pilot_pass_2026-04-25.json",
)
```

Then `--runner-config` schema (`configs/exp3a_runner.yaml`):

```yaml
intensity_levels: [1, 2, 3, 4, 5, 6, 7]
pilot_seed_path: configs/intensity_pilot_pass_2026-04-25.json
```

```bash
uv run affect-battery run \
    --experiment exp3a \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition strong_positive \
    --num-runs 50 \
    --base-url http://<endpoint>/v1 \
    --output-dir results/exp3a \
    --runner-config configs/exp3a_runner.yaml
```

Per-level subdirs are written under `results/exp3a/level_1/`, `level_2/`, etc.

### Exp 3b — cognitive scope (H3b)

`--runner-config` schema (`configs/exp3b_runner.yaml`):

```yaml
n_generations: 10
prompts:
  - id: story_1
    text: "Continue the story: The lighthouse keeper saw a strange light..."
  - id: brainstorm_1
    text: "List 5 unconventional uses for a paperclip."
  - id: creative_problem_1
    text: "How might a community without internet share news?"
```

```bash
uv run affect-battery run \
    --experiment exp3b \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition strong_positive \
    --num-runs 5 \
    --base-url http://<endpoint>/v1 \
    --output-dir results/exp3b \
    --runner-config configs/exp3b_runner.yaml
```

Each (run_num, prompt) yields one result JSON containing `n_generations` parallel completions on the same conditioning history. Generations dispatch via `asyncio.gather` so wall-clock is one round-trip per prompt, not ten.

### Exp 3c — conservative shift (H3c)

`--runner-config` schema (`configs/exp3c_runner.yaml`):

```yaml
items:
  - difficulty: easy
    question: "What is the capital of France?"
    expected: "Paris"
  - difficulty: medium
    question: "Who wrote 'Hamlet'?"
    expected: "Shakespeare"
  - difficulty: hard
    question: "What is the speed of light in m/s?"
    expected: "299792458"
```

`difficulty` MUST be one of `{easy, medium, hard}`. Items can be hand-authored or ingested from TriviaQA / NaturalQuestions.

```bash
uv run affect-battery run \
    --experiment exp3c \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --condition strong_negative \
    --num-runs 5 \
    --base-url http://<endpoint>/v1 \
    --output-dir results/exp3c \
    --runner-config configs/exp3c_runner.yaml
```

Items dispatch concurrently via `asyncio.gather`; refusal heuristic flags responses starting with `I cannot` / `I can't` / etc.

## Analysis (reports)

After runs complete:

```bash
uv run affect-battery analyze \
    --results-dir results/ \
    --model "Llama-3-8B-Instruct"
```

This produces (when corpora exist):

| Report | Source |
|---|---|
| `results/exp1a_report.md` | Exp 1a corpus → per-condition Cohen's d + Holm correction |
| `results/exp1b_report.md` | Exp 1a + Exp 1b → three-way comparison + TOST equivalence |
| `results/exp2_report.md` | Exp 2 corpus → decay fits (exp / linear) + recovery metrics + §10 caveat |
| `results/exp3b_report.md` | Exp 3b corpus → embedding variance + n-gram ratio + §10 caveat |
| `results/exp3c_report.md` | Exp 3c corpus → hedging rate + refusal rate + §10 caveat |
| `results/h4_report.md` | Exp 1a corpus + manipulation check → per-model verdicts + 2x2 joint table |
| `results/AGGREGATE_REPORT.md` | All of the above + Holm-corrected primary family p-values |

Family-wise corrections (Holm) run automatically across H1, H1b directional, H1b TOST, H4. The aggregate report includes the corrected p-values table.

## Pre-registration

The runner accepts **two equivalent pre-registration vehicles**. Pick whichever fits the timeline; the result-file schema records which was used so reviewers can verify either way.

### Option A — GitHub commit (fast, no third-party dependency)

The methodology lives in the repo: `specs/`, `configs/osf_prereg_v1.yaml`, `scripts/`, and the runners + analyzers. A signed Git tag at a specific commit gives timestamping and immutability comparable to OSF, and reviewers can `git show <tag>` to see exactly what was pre-registered.

```bash
# At a clean commit on a branch that's been pushed to origin:
python -m scripts.create_prereg_tag \
    --tag prereg-affect-battery-2026-04-26 \
    --message "Affect Battery study, full pre-registration" \
    --sign

# Prints the --pre-registration-github-commit flag, e.g.:
#   --pre-registration-github-commit DwayneWilkes/affect-battery@1ed7b43...
```

Then cite it in your run:

```bash
ANTHROPIC_API_KEY=... uv run affect-battery run \
    --experiment exp1a --provider anthropic --model claude-opus-4-7 \
    --condition strong_negative --num-runs 50 \
    --pre-registration-github-commit DwayneWilkes/affect-battery@1ed7b43... \
    --power-report-path results/power_report.json \
    --power-report-sha <sha256> \
    --output-dir results/exp1a/
```

The commit ref appears in every result file's `config.pre_registration_github_commit`. Reviewers verify the methodology with `git show <tag>` (the tag points to the same commit).

The `create_prereg_tag` script enforces three invariants:
- Working tree must be clean (no uncommitted changes)
- HEAD must be reachable from `origin/<branch>` (the commit must be public)
- The tag name must not already exist (pre-reg tags are immutable)

### Option B — OSF (formal, third-party-timestamped)

The OSF pre-reg YAML at `configs/osf_prereg_v1.yaml` is the source of truth for hypotheses, MDEs, and stopping rules. The lifecycle is v0 → v1 → submit:

- **v0**: skeleton with placeholder MDEs. Lives at `configs/osf_prereg_v0.yaml` (or your initial scaffold).
- **v1**: probe-grounded. After running the variance + base-model probes, call `finalize_v1` to apply the observed effect sizes, the base-feasibility verdict, and an amendment_chain entry.
- **submit**: `prepare_for_submit` records the OSF URL after upload.

```python
from pathlib import Path
from src.prereg.finalize import finalize_v1, prepare_for_submit

finalize_v1(
    v0_path=Path("configs/osf_prereg_v0.yaml"),
    output_path=Path("configs/osf_prereg_v1.yaml"),
    observed_effect_sizes={
        "H1": 0.45,
        "H1b": 0.05,
        "H2": 0.30,
        "H3a": 0.20,
        "H4": 0.50,
    },
    base_feasibility_verdict="pass",   # from BaseModelProbeResult
    rationale="v0 → v1: probe-grounded MDEs after variance + base-feasibility probes",
)

# After uploading to OSF:
prepare_for_submit(Path("configs/osf_prereg_v1.yaml"), osf_url="https://osf.io/...")
```

The `amendment_chain` list is append-only. Each amendment SHOULD record `rationale` and the `content_sha` at amendment time. SHA is computed over the canonicalized YAML by `compute_prereg_sha`.

## Probes (Week 0)

Variance probe (grounds the MDE):

```bash
uv run affect-battery probe variance \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --base-url http://<endpoint>/v1 \
    --n 20 \
    --output-dir results/probes/
```

Base-model feasibility probe (decides whether Llama-3-8B base goes in the primary H4 family):

```bash
uv run affect-battery probe base-model \
    --model meta-llama/Meta-Llama-3-8B \
    --base-url http://<endpoint>/v1 \
    --output-dir results/probes/
```

If base feasibility < 0.30, demote H4 base-vs-instruct to exploratory per `configs/osf_prereg_v1.yaml::stopping_rules::base-model-feasibility-gate`.

## Output structure

```
results/
├── exp1a/
│   └── <run files>.json            ← one per (model, condition, run_num)
├── exp1b/
│   └── ...
├── exp2/
│   └── n_1/, n_3/, n_5/, n_10/      ← per-N subdirs
├── exp3a/
│   └── level_1/, level_2/, ...      ← per-intensity-level subdirs
├── exp3b/
│   └── ...
├── exp3c/
│   └── ...
├── probes/
│   ├── variance_<model>.json
│   └── base_model_<model>.json
├── exp1a_report.md                  ← rendered by `analyze`
├── exp1b_report.md
├── exp2_report.md
├── exp3b_report.md
├── exp3c_report.md
├── h4_report.md
└── AGGREGATE_REPORT.md
```

The whole `results/` tree is gitignored by default. Reports are markdown; commit them outside the worktree if you need versioning.

## Common gotchas

- **Exp 3a refuses to start with `pilot-seed SHA mismatch`**: someone edited the seed JSON after `emit_seed`. Re-run the pilot (or restore the original) and re-emit; the SHA must match canonicalized JSON exactly.
- **Manipulation check returns `UNAVAILABLE`**: the model has no `no_conditioning` runs. Schedule a no_conditioning condition alongside the treatment arms; UNAVAILABLE is a measurement gap, NOT a fail.
- **`logiqa_v1` excluded from primary aggregation**: bank `status: candidate` until alignment review records `verdict: pass`. See `src/conditioning/banks.py::is_primary_analysis_eligible`.
- **`affect-battery analyze` produces empty per-experiment tables**: the corpus dir exists but no result JSONs match the schema. Check `results/<exp>/*.json` parses and has `experiment_type`, `condition`, and `body` fields.
- **`--experiment exp3a/b/c` exits with `requires --runner-config`**: those experiments have additional config (intensity levels / prompts / items) the base CLI doesn't surface as flags.

## Where to look in code

| Concern | Path |
|---|---|
| Per-experiment runners | `src/runners/exp{1a,1b,2,3a,3b,3c}.py` |
| Per-experiment analysis | `src/analysis/exp{1a,1b,2,3b,3c}.py` |
| Cross-experiment H4 + manipulation check | `src/analysis/h4.py` |
| Per-experiment reports | `src/analysis/reports/exp{1a,1b,2,3b,3c}.py`, `h4.py` |
| Aggregate pipeline | `src/analysis/pipeline.py` |
| Statistical primitives | `src/analysis/_effect_size.py`, `src/analysis/stats/` |
| OSF pre-reg | `configs/osf_prereg_v1.yaml`, `src/prereg/finalize.py` |
| Hedging codebook | `configs/hedging_codebook.yaml`, `src/scoring/hedging.py` |
