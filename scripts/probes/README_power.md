# H3a power analysis

Once the variance probe (`scripts/probes/h3a_variance_probe.py`) has
produced per-intensity-level variance estimates, run the
simulation-based power analysis to obtain a recommended `n_per_level`
for the H3a quadratic test.

## Variance JSON input format

The variance probe emits this shape directly. Sigma values reflect
binary-outcome standard deviations (`σ = √(p(1-p))`), so they typically
sit in the 0.30-0.51 range for accuracies between 0.1 and 0.9.

```json
{
  "model": "gpt-5.4-nano",
  "task": "GSM8K + GSM-Hard",
  "n_levels": 7,
  "n_per_level": 15,
  "sigma_per_level": [0.507, 0.258, 0.458, 0.516, 0.458, 0.414, 0.488],
  "mean_per_level": [0.400, 0.067, 0.267, 0.467, 0.267, 0.200, 0.333],
  "icc": null,
  "notes": "..."
}
```

`sigma_per_level` is a 7-element list of within-level standard deviations
measured by the variance probe. Levels are ordered 1-7 (matching
`INTENSITY_LEVELS`). `icc` is the intra-class correlation estimate; for
designs with one measurement per cell (the H3a default), ICC is recorded
but does not affect the simulation, and `null` is acceptable.

## Run the power analysis

```bash
uv run python scripts/probes/h3a_power_report.py \
    --variance-json results/probes/h3a_variance_<date>.json \
    --output results/probes/h3a_power_report_<date>.json
```

Output: a power report JSON containing the recommended n-per-level and
per-scenario simulation traces. The recommended n is the maximum across
power scenarios that achieved target power (default 0.80) at α=0.05
within the search ceiling (default `n_max=300`; raise via `--n-max` if
the smallest target effect needs a wider search).

## Default power scenarios

The report runs four assumed effect sizes (β₂ values, corresponding to
inverted-U curves with peak at level 4):

| Scenario | β₂ assumed | Drop from peak to edges |
|---|---|---|
| very_small_drop_0.025 | -0.0028 | 0.025 |
| small_drop_0.05 | -0.0056 | 0.05 |
| moderate_drop_0.10 | -0.0111 | 0.10 |
| large_drop_0.15 | -0.0167 | 0.15 |

The recommended n is sized for the smallest effect that achieved target
power. If the variance probe suggests the actual effect is larger, the
run can be sized to one of the smaller-n scenarios.

## Simulation method

The simulation is Monte Carlo:

1. For each iteration: sample n_per_level Gaussian observations per level
   (mean from the assumed inverted-U curve, sd from sigma_per_level).
2. Fit the quadratic regression `accuracy ~ b0 + b1·level + b2·level²`
   on the sampled cells.
3. Compute the t-statistic for β₂ using the closed-form OLS standard
   error: `SE(β₂) = (σ_pooled / √n_per_level) · √((XᵀX)⁻¹_{3,3})`.
4. Reject H0 (β₂ ≥ 0) when t < -t_crit at α=0.05 one-sided.
5. Power = fraction of iterations that rejected.

The `(XᵀX)⁻¹_{3,3}` constant for the 7-level [1, x, x²] design is
196/16464 ≈ 0.01191; this accounts for collinearity between the linear
and quadratic terms in the regression.

## Calibration check

The simulation passes a Type I error sanity check: when β₂ = 0 (no
effect), the rejection rate is 0.047 across 2000 simulations (target
α=0.05). See `tests/test_h3a_simulation.py` for the full test suite.

## Power-report contents

The report records:

- The variance estimates used as input
- Per-scenario `n_recommended` and full search trace
- The cross-scenario recommendation (max n)
- Simulation parameters (alpha, target_power, n_simulations, seed)
- A timestamp

This file is the artifact the harness's `--power-report-path` gate
checks at data-collection time.
