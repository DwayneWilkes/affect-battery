"""Deterministic per-level item sampling for intensity-axis experiments.

The H3a variance probe and the production run_exp3a runner both draw
n_per_level disjoint items per intensity level from a stimulus bank.
Disjoint sampling prevents per-level variance estimates from being
artificially correlated by item sharing across levels.
"""

from __future__ import annotations

import random


def sample_items(
    items: list[dict],
    n_per_level: int,
    n_levels: int,
    seed: int,
) -> list[list[dict]]:
    """Return n_levels disjoint samples of n_per_level items, deterministic in seed.

    The same (items, n_per_level, n_levels, seed) input produces the
    same per-level item assignment across calls.
    """
    if len(items) < n_per_level * n_levels:
        raise ValueError(
            f"bank has {len(items)} items; need at least "
            f"{n_per_level * n_levels} for non-overlapping per-level samples"
        )
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    return [
        shuffled[i * n_per_level:(i + 1) * n_per_level]
        for i in range(n_levels)
    ]
