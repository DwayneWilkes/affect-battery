"""Precision simulation for the H3b folded-axis affective-context contrast.

The H3b prereg's interpretive thresholds are CI-width-based, not power-
based:

- `c_ci95_hi < 0.05`: bounded-small claim (the strongest informative claim)
- `c_ci95_hi >= 0.10`: uninformative claim (CI too wide)

So the meaningful sample-size question is "at what `n_items` does the
item-level percentile bootstrap CI half-width on `c = m_mag2 −
½(m_mag1 + m_mag3)` reliably stay below 0.05?" — not "what n gives 80%
power to detect c > 0?"

The simulation:

1. Sample n_items per-item p̂s from the calibrated pool with replacement.
2. Compute per-cell p_cell = clip(p_item + offset(magnitude), 0, 1)
   where `offset` injects the assumed contrast c_assumed symmetrically
   so the contrast sums to c exactly.
3. Draw all (item, magnitude) Binomial counts in a single
   broadcast call: success counts ~ Binomial(n_reps_per_cell × 2, p_cell)
   for the 2 signed levels per magnitude (which share the same offset).
4. Reduce to per-item folded contrast directly in numpy: no DataFrame.
5. Bootstrap CI by resampling the n_items items with replacement
   (also numpy-vectorized).
6. Repeat across `n_simulations` and aggregate metrics.

`find_min_n_for_precision` binary-searches for the smallest `n_items`
that hits a target CI half-width threshold with at least
`target_reliability` % of simulations meeting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Folded-magnitude design from the prereg: 3 magnitudes pool 6 stimulated
# levels (level 4 = neutral, excluded). Each magnitude pools 2 signed
# levels (e.g. mag K=2 pools levels 2 and 6).
N_MAGNITUDES = 3
N_LEVELS_PER_MAGNITUDE = 2

# Column indices for arrays shaped (..., N_MAGNITUDES). Names make the
# folded-contrast formula self-checking against `_per_magnitude_offsets`.
MAG1_IDX, MAG2_IDX, MAG3_IDX = 0, 1, 2


@dataclass
class H3bPrecisionResult:
    """One sample-size's worth of simulation output.

    Reliability percentages are over `n_simulations` Monte Carlo trials.
    `bootstrap_ci_half_widths` retains the per-trial half-widths so
    callers can plot the distribution if needed.
    """
    n_items: int
    n_reps_per_cell: int
    n_simulations: int
    n_bootstrap: int
    c_assumed: float
    median_ci_half_width: float
    mean_ci_half_width: float
    pct_below_0_05: float
    pct_below_0_10: float
    median_c_estimate: float
    bootstrap_ci_half_widths: np.ndarray = field(repr=False)
    c_estimates: np.ndarray = field(repr=False)


def _per_magnitude_offsets(c_assumed: float) -> dict[int, float]:
    """Per-magnitude offsets to inject the contrast `c_assumed`.

    Choose offsets so that:
      - mag1 and mag3 both shift down by c_assumed/2
      - mag2 shifts up by c_assumed/2
    Then c = m_mag2 - ½(m_mag1 + m_mag3) = c_assumed exactly, and the
    overall mean accuracy across magnitudes is preserved (no confound
    with item difficulty).
    """
    return {
        1: -c_assumed / 2.0,
        2: +c_assumed / 2.0,
        3: -c_assumed / 2.0,
    }


def _simulate_c_per_item_batch(
    p_per_item_batch: np.ndarray,
    n_reps_per_cell: int,
    c_assumed: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized per-item folded contrast across a batch of simulations.

    `p_per_item_batch` is shape `(n_simulations, n_items)`; output is
    shape `(n_simulations, n_items)` of per-item contrasts
    `m_mag2[s,i] − ½(m_mag1[s,i] + m_mag3[s,i])`. Both the simulation
    axis and the magnitude axis are collapsed by numpy broadcasting,
    so a single `rng.binomial` call covers every (simulation, item,
    magnitude) cell.

    The two signed levels per magnitude share the same offset under
    `_per_magnitude_offsets`, so the variance of the mean of `2n` shared-
    `p` Bernoullis equals the variance of `Binomial(2n, p) / 2n`. That
    equivalence lets the per-rep loop collapse into a single binomial
    draw per cell — the load-bearing step that makes the whole batch
    one numpy call.
    """
    offsets_arr = np.array(
        [_per_magnitude_offsets(c_assumed)[m] for m in (1, 2, 3)]
    )
    p_cell = np.clip(
        p_per_item_batch[..., None] + offsets_arr[None, None, :], 0.0, 1.0,
    )
    n_per_mag = n_reps_per_cell * N_LEVELS_PER_MAGNITUDE
    m_per_item_mag = rng.binomial(n_per_mag, p_cell) / n_per_mag
    return (
        m_per_item_mag[..., MAG2_IDX]
        - 0.5 * (m_per_item_mag[..., MAG1_IDX] + m_per_item_mag[..., MAG3_IDX])
    )


def _bootstrap_ci_half_width(
    c_per_item: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Item-level percentile bootstrap CI half-width on the mean contrast.

    Returns (point_estimate, ci_lo, ci_hi). Half-width = (ci_hi - ci_lo)/2.

    Vectorized: draw an (n_bootstrap × n_items) index matrix of resampled
    item indices, gather, mean across items, then percentile across the
    bootstrap dimension. Numpy is in this hot loop — pandas would be 30x
    slower.
    """
    n_items = len(c_per_item)
    idx = rng.integers(0, n_items, size=(n_bootstrap, n_items))
    boot_means = c_per_item[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    point = float(c_per_item.mean())
    return point, lo, hi


def simulate_h3b_precision(
    n_items: int,
    p_hat_per_item: list[float] | np.ndarray,
    n_reps_per_cell: int = 20,
    c_assumed: float = 0.0,
    n_simulations: int = 200,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> H3bPrecisionResult:
    """Monte Carlo CI-precision simulation at fixed `n_items`.

    `p_hat_per_item` is the calibrated p̂ pool (e.g. the 19 in-band
    items from the May 2026 calibration). When `n_items` exceeds the
    pool size, items are sampled with replacement so the simulation
    matches the realistic case of pre-screening more candidates than
    we currently have on hand.
    """
    rng = np.random.default_rng(seed)
    p_pool = np.asarray(p_hat_per_item, dtype=float)
    if len(p_pool) == 0:
        raise ValueError("p_hat_per_item must be non-empty")

    half_widths = np.zeros(n_simulations, dtype=float)
    c_estimates = np.zeros(n_simulations, dtype=float)
    # Sample all n_simulations × n_items per-item p̂s in one numpy call;
    # all binomial draws also vectorized in `_simulate_c_per_item_batch`.
    # The bootstrap stays per-sim because the index matrix
    # (n_simulations × n_bootstrap × n_items) at production scale would
    # exceed reasonable memory; per-sim bootstrap is already vectorized
    # internally.
    p_per_item_batch = rng.choice(
        p_pool, size=(n_simulations, n_items), replace=True,
    )
    c_per_item_batch = _simulate_c_per_item_batch(
        p_per_item_batch, n_reps_per_cell, c_assumed, rng,
    )
    for s in range(n_simulations):
        point, lo, hi = _bootstrap_ci_half_width(
            c_per_item_batch[s], n_bootstrap, rng,
        )
        half_widths[s] = (hi - lo) / 2.0
        c_estimates[s] = point

    pct_below_0_05 = 100.0 * float((half_widths < 0.05).mean())
    pct_below_0_10 = 100.0 * float((half_widths < 0.10).mean())

    return H3bPrecisionResult(
        n_items=n_items,
        n_reps_per_cell=n_reps_per_cell,
        n_simulations=n_simulations,
        n_bootstrap=n_bootstrap,
        c_assumed=c_assumed,
        median_ci_half_width=float(np.median(half_widths)),
        mean_ci_half_width=float(np.mean(half_widths)),
        pct_below_0_05=pct_below_0_05,
        pct_below_0_10=pct_below_0_10,
        median_c_estimate=float(np.median(c_estimates)),
        bootstrap_ci_half_widths=half_widths,
        c_estimates=c_estimates,
    )


def find_min_n_for_precision(
    p_hat_per_item: list[float] | np.ndarray,
    target_ci_half_width: float = 0.05,
    target_reliability: float = 80.0,
    n_min: int = 5,
    n_max: int = 100,
    n_reps_per_cell: int = 20,
    c_assumed: float = 0.0,
    n_simulations: int = 200,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[int | None, list[H3bPrecisionResult]]:
    """Binary-search for the smallest `n_items` meeting the precision floor.

    The criterion: ≥ `target_reliability` % of simulations have CI
    half-width strictly below `target_ci_half_width`. Returns `(n, trace)`,
    or `(None, trace)` if even `n_max` falls short.
    """
    trace: list[H3bPrecisionResult] = []

    def _meets(result: H3bPrecisionResult) -> bool:
        # Pick the right pct-below depending on the threshold.
        if abs(target_ci_half_width - 0.05) < 1e-9:
            pct = result.pct_below_0_05
        elif abs(target_ci_half_width - 0.10) < 1e-9:
            pct = result.pct_below_0_10
        else:
            # Custom threshold: recompute from the per-sim half-widths.
            pct = 100.0 * float(
                (result.bootstrap_ci_half_widths < target_ci_half_width).mean()
            )
        return pct >= target_reliability

    upper = simulate_h3b_precision(
        n_items=n_max, p_hat_per_item=p_hat_per_item,
        n_reps_per_cell=n_reps_per_cell, c_assumed=c_assumed,
        n_simulations=n_simulations, n_bootstrap=n_bootstrap, seed=seed,
    )
    trace.append(upper)
    if not _meets(upper):
        return None, trace

    lo, hi = n_min, n_max
    best_n = n_max
    while lo <= hi:
        mid = (lo + hi) // 2
        result = simulate_h3b_precision(
            n_items=mid, p_hat_per_item=p_hat_per_item,
            n_reps_per_cell=n_reps_per_cell, c_assumed=c_assumed,
            n_simulations=n_simulations, n_bootstrap=n_bootstrap, seed=seed,
        )
        trace.append(result)
        if _meets(result):
            best_n = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return best_n, trace
