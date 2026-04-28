"""analyze_arousal_magnitude (amendment 002)."""

from __future__ import annotations

import random

from src.analysis.exp3a import analyze_arousal_magnitude


def test_quadratic_fit_on_arousal_magnitude():
    """A clean inverted-U on arousal=|L-4| should produce β₂ < 0."""
    rng = random.Random(0)
    accuracy_by_level: dict[int, list[float]] = {}
    for L in range(1, 8):
        arousal = abs(L - 4)
        # Mean accuracy peaks at arousal=1 (mild positive/negative).
        # p(arousal) = 0.5 - 0.05 * (arousal - 1)^2  -> peak 0.5 at arousal=1
        p = 0.5 - 0.05 * (arousal - 1) ** 2
        # Generate 200 binary draws per level so the test signal beats noise.
        accuracy_by_level[L] = [
            1.0 if rng.random() < p else 0.0 for _ in range(200)
        ]
    result = analyze_arousal_magnitude(accuracy_by_level)
    assert result["beta_2"] < 0.0
    assert result["beta_2_p_one_sided"] < 0.05
    # Peak arousal should land near 1.0 (give it some MC slack).
    assert 0.0 <= result["peak_arousal"] <= 2.5


def test_arousal_magnitude_recodes_levels_correctly():
    """Levels {1..7} should map to arousal {3, 2, 1, 0, 1, 2, 3}."""
    accuracy_by_level = {L: [0.5] for L in range(1, 8)}
    result = analyze_arousal_magnitude(accuracy_by_level)
    expected = {1: 3, 2: 2, 3: 1, 4: 0, 5: 1, 6: 2, 7: 3}
    assert result["level_to_arousal"] == expected
    assert sorted(result["arousal_values"]) == [0, 1, 2, 3]
