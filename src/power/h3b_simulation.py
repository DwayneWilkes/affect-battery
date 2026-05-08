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

1. Build a long-format pandas DataFrame of simulated cells: one row per
   (item, magnitude, level_within_magnitude, rep), with an `accuracy ∈
   {0, 1}` column drawn from Bernoulli(`p_item + offset(magnitude)`)
   where `offset` is parametrized to inject `c_assumed`.
2. Aggregate to per-item per-magnitude means via groupby.
3. Pivot to per-item contrast `c_per_item`.
4. Bootstrap CI by resampling the n_items items with replacement
   (numpy-vectorized — pandas would be too slow in the hot loop).
5. Repeat across `n_simulations` and aggregate metrics.

`find_min_n_for_precision` binary-searches for the smallest `n_items`
that hits a target CI half-width threshold with at least
`target_reliability` % of simulations meeting it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Folded-magnitude design from the prereg: 3 magnitudes pool 6 stimulated
# levels (level 4 = neutral, excluded). Each magnitude pools 2 signed
# levels (e.g. mag K=2 pools levels 2 and 6).
N_MAGNITUDES = 3
N_LEVELS_PER_MAGNITUDE = 2


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


def _simulate_one_dataset(
    p_per_item: np.ndarray,
    n_reps_per_cell: int,
    c_assumed: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build the long-format cell DataFrame for one simulated experiment.

    Columns: item_id (int), magnitude (1..3), level_within_mag (0..1),
    accuracy (0/1). Each row is one rep.
    """
    n_items = len(p_per_item)
    offsets = _per_magnitude_offsets(c_assumed)
    rows = []
    for item_id, p_item in enumerate(p_per_item):
        for mag in (1, 2, 3):
            p_cell = float(np.clip(p_item + offsets[mag], 0.0, 1.0))
            for lvl in range(N_LEVELS_PER_MAGNITUDE):
                draws = rng.binomial(1, p_cell, size=n_reps_per_cell)
                for rep_idx, acc in enumerate(draws):
                    rows.append((item_id, mag, lvl, rep_idx, int(acc)))
    return pd.DataFrame(
        rows,
        columns=["item_id", "magnitude", "level_within_mag", "rep", "accuracy"],
    )


def _per_item_contrast(df: pd.DataFrame) -> np.ndarray:
    """Reduce the cell-level DataFrame to per-item folded contrasts.

    Returns a 1-D array `c_per_item[i] = m_mag2[i] - ½(m_mag1[i] + m_mag3[i])`.
    """
    per_item_mag = (
        df.groupby(["item_id", "magnitude"])["accuracy"].mean().unstack("magnitude")
    )
    return (per_item_mag[2] - 0.5 * (per_item_mag[1] + per_item_mag[3])).to_numpy()


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
    for s in range(n_simulations):
        # Sample n_items per-item p̂s. With replacement so n_items can
        # exceed pool size.
        p_per_item = rng.choice(p_pool, size=n_items, replace=True)
        df = _simulate_one_dataset(p_per_item, n_reps_per_cell, c_assumed, rng)
        c_per_item = _per_item_contrast(df)
        point, lo, hi = _bootstrap_ci_half_width(c_per_item, n_bootstrap, rng)
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
