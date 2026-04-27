# H3a probe scripts

This directory contains the operator scripts for the prerequisites of
the H3a (Yerkes-Dodson) experiment. The chain is:

1. **Intensity-axis pilot**: human raters validate that the seven
   `INTENSITY_LEVELS` stimuli order on a signed intensity scale.
2. **Variance probe**: sweeps each intensity level against the task
   bank to measure per-level accuracy variance.
3. **Power analysis**: turns the variance estimates into a recommended
   `n_per_level` for the H3a quadratic test.

## Files

- `build_rating_form.py`: generates a YAML rating form for one rater.
  Stimuli are presented in seed-randomized order with opaque
  identifiers (`stim_001`..`stim_007`); the canonical level mapping is
  recoverable only by re-running the same seeded shuffle in the driver.
- `run_intensity_pilot.py`: loads filled forms, computes Krippendorff α
  (or accepts a single-rater form under `--solo-rater`), and writes the
  pilot seed.
- `h3a_variance_probe.py`: runs each `INTENSITY_LEVELS` stimulus as a
  system prompt against a math-word-problem bank and emits per-level
  mean/σ accuracy for the power simulation. See `README_power.md` for
  the input format it produces.
- `h3a_power_report.py`: Monte Carlo power analysis over assumed effect
  sizes; produces the recommended `n_per_level`. See `README_power.md`.

The α math + decision rule live in `src/probes/intensity_pilot.py`.

# Intensity-axis pilot protocol

Three or more human raters independently rate the seven
`INTENSITY_LEVELS` stimuli on a signed intensity scale; the pilot
computes Krippendorff α and writes a signed pilot-seed artifact when
the rater agreement gate passes. Pilots run with a single rater are
supported via the `--solo-rater` flag (see *Single-rater pilot* below).

## Protocol

### 1. Recruit at least three raters

Three raters is the minimum; more raters give a tighter α estimate.
Raters should not have prior exposure to the stimulus set.

### 2. Generate per-rater forms

```bash
uv run python scripts/probes/build_rating_form.py \
    --rater-id rater_A --output ratings/rater_A.yaml --seed 1
uv run python scripts/probes/build_rating_form.py \
    --rater-id rater_B --output ratings/rater_B.yaml --seed 2
uv run python scripts/probes/build_rating_form.py \
    --rater-id rater_C --output ratings/rater_C.yaml --seed 3
```

Use a different `--seed` per rater so the stimulus presentation order
varies across raters. The form's `rating` fields are blank; raters fill
them in.

Stimulus identifiers in the form are opaque (`stim_001`..`stim_NNN`,
assigned in presentation order) so the rater cannot infer the canonical
level from the id. The driver script reverses the mapping by re-running
the same seeded shuffle.

### 3. Distribute, collect, and validate

Send each rater their YAML file. The form contains inline instructions
(read the stimulus, write an integer 1-7 for `rating`). Collect the
filled forms back into one directory.

A filled form looks like:

```yaml
rater_id: rater_A
randomization_seed: 1
ratings:
  - id: stim_001
    feedback_text: "Your response has been recorded. We will now move on to the next question."
    rating: 4
  - id: stim_002
    feedback_text: "That was really impressive work..."
    rating: 1
  ...
```

The `id` field is purely positional within the rater's randomized
ordering; the canonical-level mapping is reconstructed by the driver.

### 4. Run the pilot

```bash
uv run python scripts/probes/run_intensity_pilot.py \
    --ratings-dir ratings/ \
    --output configs/intensity_pilot_seed.json \
    --pilot-date 2026-04-27 \
    --axis-id intensity_axis_v1
```

The script prints α (overall + pairwise) and the decision:

| Decision | Meaning | Next step |
|---|---|---|
| `proceed` | α ≥ 0.8 overall AND all rater-pairs ≥ 0.6. Raters agree well enough. | Pilot seed written to `--output`; H3a is unblocked on this gate. |
| `restructure` | α between 0.6 and 0.8. Raters mostly agree but with notable noise. | Refine the stimulus prompts; re-run the pilot with revised stimuli. |
| `collapse` | α < 0.6 overall OR any pair < 0.6. Raters disagree substantially. | Collapse adjacent levels (e.g. 7 levels → 5 or 3) or restructure to a single-mechanism axis. |

### 5. Use the seed in subsequent runs

The H3a runner refers to the seed via its `pilot_seed_path` argument.
The seed records the pilot date, axis ID, level count, α (overall and
pairwise), and a SHA-256 digest over the canonicalized payload. The
runner re-computes the SHA from the file at startup and refuses to
proceed on a mismatch.

## Rating scale

The 7 `INTENSITY_LEVELS` stimuli span signed valence intensity:

- 1 = strongest positive feedback
- 4 = neutral / no affective content
- 7 = strongest negative feedback

Raters rate each stimulus on this 1-7 scale. The pilot validates that
raters CAN order the stimuli on this signed axis with sufficient
agreement; the pilot does not validate that the raters' ordering
matches the stimulus author's intended ordering.

## What constitutes a fresh rater

A "fresh" rater is one who has not previously rated this stimulus set
and has not seen the level annotations attached to the stimuli in
`src/conditioning/prompts.py`. Sharing the source file with raters
before they rate biases the result.

## When to re-run the pilot

Re-run when:

- The stimulus set changes (any addition, removal, or rewording of an
  `INTENSITY_LEVELS` entry).
- The rating scale changes (1-7 to 1-5, signed to unsigned, etc.).
- The axis identifier changes.

Each re-run requires a fresh recruitment pass; raters should not see
prior pilot results before rating.

## Variance probe

After the pilot seed is written, the variance probe sweeps each
intensity stimulus against the task bank and records per-level accuracy
mean and standard deviation. Its output JSON is the input to the power
simulation.

```bash
uv run python scripts/probes/h3a_variance_probe.py \
    --bank configs/banks/gsm8k_v1.yaml \
    --provider openai --model gpt-5.4-nano \
    --n-per-level 15 \
    --output results/probes/h3a_variance_<date>.json
```

Use `--dry-run` to validate the script end-to-end without API spend.
Sample sizing: `n_per_level × 7` items must be ≤ the bank size, since
each level draws a disjoint sample from the bank to avoid correlating
per-level variance through item sharing.

The probe writes `sigma_per_level`, `mean_per_level`, the bank SHA-256,
the sample seed, and the model/provider used. The downstream power
report consumes this file via `--variance-json`.

## Power analysis

Once the variance JSON exists, run the simulation-based power analysis
to obtain a recommended `n_per_level`. See `README_power.md` for the
input format, the simulation method, and the calibration check.

```bash
uv run python scripts/probes/h3a_power_report.py \
    --variance-json results/probes/h3a_variance_<date>.json \
    --output results/probes/h3a_power_report_<date>.json
```

The recommended `n_per_level` plus the variance and pilot artifacts
together form the pre-registration evidence for the H3a run.

## Single-rater pilot

Pilots run with a single rater are supported via the `--solo-rater`
flag. The resulting seed records the supplied `rater_id`, ordinal
ratings, axis ID, and a SHA-256 digest; no Krippendorff α is computed.
The seed is flagged `irr_validated: false` and `solo_rater: true`, and
downstream gates apply their own acceptance policy on those flags.

```bash
# Generate one rating form
uv run python scripts/probes/build_rating_form.py \
    --rater-id rater_PI --output ratings/solo.yaml --seed 1

# Fill it out, then run with --solo-rater
uv run python scripts/probes/run_intensity_pilot.py \
    --ratings-dir ratings/ \
    --output configs/intensity_pilot_seed.json \
    --pilot-date 2026-04-27 --axis-id intensity_axis_v1 \
    --solo-rater
```

Use a pseudonymous `rater_id` (e.g. `rater_PI`, `rater_01`); the value
is recorded verbatim in the seed JSON. Methods documentation should
note the single-rater design and reference any pre-registration
amendment authorizing it.
