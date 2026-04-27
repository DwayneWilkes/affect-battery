# Intensity-axis pilot protocol

This directory contains the operator scripts for running the
inter-rater reliability pilot that gates the H3a (Yerkes-Dodson)
experiment. Three or more human raters independently rate the seven
`INTENSITY_LEVELS` stimuli on a signed intensity scale; the pilot
computes Krippendorff α and writes a signed pilot-seed artifact when
the rater agreement gate passes.

## Files

- `build_rating_form.py` — generates a YAML rating form for one rater.
  Stimuli are presented in seed-randomized order to control for
  order-effect bias.
- `run_intensity_pilot.py` — loads filled forms, computes α, prints the
  decision (proceed / collapse / restructure), and writes the pilot
  seed when the gate passes.

The α math + decision rule live in `src/probes/intensity_pilot.py`.

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

### 3. Distribute, collect, and validate

Send each rater their YAML file. The form contains inline instructions
(read the stimulus, write an integer 1-7 for `rating`). Collect the
filled forms back into one directory.

A filled form looks like:

```yaml
rater_id: rater_A
randomization_seed: 1
ratings:
  - id: level_4
    feedback_text: "Your response has been recorded. We will now move on to the next question."
    rating: 4
  - id: level_1
    feedback_text: "That was really impressive work..."
    rating: 1
  ...
```

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
