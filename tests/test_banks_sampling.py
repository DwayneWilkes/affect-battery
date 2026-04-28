"""Shared bank sampling helper.

The H3a variance probe and run_exp3a both need to draw n_per_level
disjoint items per intensity level from a stimulus bank, deterministic
in a seed. The helper at src.banks.sampling.sample_items is the single
implementation; both call sites import it.
"""

from __future__ import annotations

import pytest

from src.banks.sampling import sample_items


def _bank(n: int) -> list[dict]:
    """Build a synthetic bank with n items, ids item_000..item_{n-1}."""
    return [{"id": f"item_{i:03d}", "question": f"q{i}", "expected": str(i)} for i in range(n)]


def test_same_seed_produces_same_per_level_assignment():
    items = _bank(50)
    a = sample_items(items, n_per_level=5, n_levels=7, seed=42)
    b = sample_items(items, n_per_level=5, n_levels=7, seed=42)
    a_ids = [[item["id"] for item in level] for level in a]
    b_ids = [[item["id"] for item in level] for level in b]
    assert a_ids == b_ids


def test_different_seed_produces_different_assignment():
    items = _bank(50)
    a = sample_items(items, n_per_level=5, n_levels=7, seed=42)
    b = sample_items(items, n_per_level=5, n_levels=7, seed=99)
    a_ids = {item["id"] for level in a for item in level}
    b_ids = {item["id"] for level in b for item in level}
    assert a_ids != b_ids or [
        [it["id"] for it in lvl] for lvl in a
    ] != [[it["id"] for it in lvl] for lvl in b]


def test_per_level_samples_are_pairwise_disjoint():
    items = _bank(50)
    levels = sample_items(items, n_per_level=5, n_levels=7, seed=42)
    assert len(levels) == 7
    for level in levels:
        assert len(level) == 5
    seen_pairs: list[tuple[int, int]] = []
    for i in range(7):
        for j in range(i + 1, 7):
            seen_pairs.append((i, j))
    for i, j in seen_pairs:
        ids_i = {item["id"] for item in levels[i]}
        ids_j = {item["id"] for item in levels[j]}
        assert ids_i.isdisjoint(ids_j), f"levels {i} and {j} share items: {ids_i & ids_j}"


def test_insufficient_bank_raises_with_both_numbers():
    items = _bank(5)
    with pytest.raises(ValueError) as exc:
        sample_items(items, n_per_level=10, n_levels=7, seed=42)
    msg = str(exc.value)
    assert "5" in msg
    assert "70" in msg


def test_exact_size_bank_succeeds():
    items = _bank(35)
    levels = sample_items(items, n_per_level=5, n_levels=7, seed=42)
    all_ids = {item["id"] for level in levels for item in level}
    assert len(all_ids) == 35
    assert len(levels) == 7


def test_within_subjects_same_items_at_every_level():
    from src.banks.sampling import sample_items_within_subjects
    items = _bank(50)
    levels = sample_items_within_subjects(items, n_per_level=10, n_levels=7, seed=42)
    assert len(levels) == 7
    first_ids = [item["id"] for item in levels[0]]
    assert len(first_ids) == 10
    assert len(set(first_ids)) == 10
    for level in levels[1:]:
        assert [item["id"] for item in level] == first_ids


def test_within_subjects_deterministic_by_seed():
    from src.banks.sampling import sample_items_within_subjects
    items = _bank(50)
    a = sample_items_within_subjects(items, n_per_level=10, n_levels=7, seed=42)
    b = sample_items_within_subjects(items, n_per_level=10, n_levels=7, seed=42)
    a_map = [(L, R, item["id"]) for L, level in enumerate(a) for R, item in enumerate(level)]
    b_map = [(L, R, item["id"]) for L, level in enumerate(b) for R, item in enumerate(level)]
    assert a_map == b_map


def test_within_subjects_different_seeds_different_items():
    from src.banks.sampling import sample_items_within_subjects
    items = _bank(50)
    a = sample_items_within_subjects(items, n_per_level=10, n_levels=7, seed=42)
    b = sample_items_within_subjects(items, n_per_level=10, n_levels=7, seed=99)
    assert [it["id"] for it in a[0]] != [it["id"] for it in b[0]]


def test_within_subjects_n_per_level_size_only():
    from src.banks.sampling import sample_items_within_subjects
    items = _bank(122)
    levels = sample_items_within_subjects(items, n_per_level=122, n_levels=7, seed=42)
    assert len(levels) == 7
    assert len(levels[0]) == 122


def test_within_subjects_insufficient_bank_raises():
    from src.banks.sampling import sample_items_within_subjects
    items = _bank(5)
    with pytest.raises(ValueError):
        sample_items_within_subjects(items, n_per_level=10, n_levels=7, seed=42)
