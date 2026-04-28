"""analyze_within_subjects (amendment 002)."""

from __future__ import annotations

import random

from src.analysis.exp3a import analyze_within_subjects


def _make_within_subjects_corpus(
    n_items: int, levels: list[int], item_difficulty: dict[str, float],
    level_modifier=lambda L: 0.0, seed: int = 0,
) -> list[dict]:
    """Build a within-subjects corpus where each item is paired with each level."""
    rng = random.Random(seed)
    corpus = []
    for i in range(n_items):
        item_id = f"item_{i:04d}"
        base_p = item_difficulty.get(item_id, 0.5)
        for L in levels:
            p = base_p + level_modifier(L)
            p = max(0.0, min(1.0, p))
            binary = 1 if rng.random() < p else 0
            corpus.append({
                "body": {
                    "intensity_level": L,
                    "binary_correct": binary,
                    "item_id": item_id,
                    "sampling_mode": "within_subjects",
                },
            })
    return corpus


def test_per_item_dummies_recover_quadratic_signal():
    """A within-subjects corpus with a clean inverted-U on level should yield β₂ < 0."""
    levels = list(range(1, 8))
    n_items = 50
    item_difficulty = {f"item_{i:04d}": 0.3 + 0.4 * (i / n_items) for i in range(n_items)}

    def yd_modifier(L: int) -> float:
        return -0.05 * (L - 4) ** 2

    corpus = _make_within_subjects_corpus(
        n_items=n_items, levels=levels,
        item_difficulty=item_difficulty,
        level_modifier=yd_modifier, seed=42,
    )
    result = analyze_within_subjects(corpus)
    assert result["beta_2"] < 0.0
    assert result["beta_2_p_one_sided"] < 0.05
    assert result["n_items_used"] > 0


def test_zero_variance_items_dropped():
    """Items with all-zero or all-one outcomes are dropped before fitting."""
    levels = list(range(1, 8))
    corpus = []
    # 5 items with zero variance (all 0)
    for i in range(5):
        for L in levels:
            corpus.append({"body": {
                "intensity_level": L, "binary_correct": 0, "item_id": f"zero_{i}",
                "sampling_mode": "within_subjects",
            }})
    # 5 items with non-zero variance
    for i in range(5):
        for L in levels:
            corpus.append({"body": {
                "intensity_level": L,
                "binary_correct": 1 if L % 2 == 0 else 0,
                "item_id": f"var_{i}",
                "sampling_mode": "within_subjects",
            }})
    result = analyze_within_subjects(corpus)
    assert result["n_items_used"] == 5
    assert result["n_items_dropped_zero_variance"] == 5
