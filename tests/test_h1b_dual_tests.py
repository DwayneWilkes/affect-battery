"""Task 4.3 Red — H1b TOST equivalence + directional test pair.

Per power-analysis spec "H1b null-equivalence test (TOST)": for the
session-2 effect, the analysis runs BOTH a one-sided directional test
(session_2_effect > 0) AND a TOST equivalence test at epsilon=+/-0.10.
TOST is two one-sided tests; report both p-values.
"""

from __future__ import annotations

import pytest


class TestTostEquivalence:
    def test_tost_returns_two_one_sided_p_values(self):
        from src.analysis.stats.tost import tost_equivalence

        # Effect at zero with small SE: should be equivalent to zero
        result = tost_equivalence(effect=0.0, se=0.02, epsilon=0.10, alpha=0.05)
        assert "p_lower" in result
        assert "p_upper" in result
        assert "p_tost" in result  # max of the two one-sided p-values
        # At zero effect, both bounds are far away => both p_lower and p_upper small
        assert result["p_tost"] < 0.05
        assert result["equivalent"] is True

    def test_tost_rejects_when_effect_outside_band(self):
        from src.analysis.stats.tost import tost_equivalence

        # Effect well outside the +/-0.10 band: not equivalent
        result = tost_equivalence(effect=0.30, se=0.02, epsilon=0.10, alpha=0.05)
        assert result["equivalent"] is False
        assert result["p_tost"] > 0.05

    def test_tost_validates_epsilon(self):
        from src.analysis.stats.tost import tost_equivalence

        with pytest.raises(ValueError, match="epsilon"):
            tost_equivalence(effect=0.0, se=0.01, epsilon=-0.10, alpha=0.05)


class TestExp1bH1bDualTests:
    def test_tost_and_directional_both_run(self):
        """analyze_exp1b with h1b_dual_tests=True returns BOTH a one-sided
        directional p-value AND a TOST equivalence p-value at epsilon=+/-0.10."""
        from src.analysis.exp1b import analyze_exp1b

        exp1a_corpus = [
            {"condition": "no_conditioning", "transfer_correct": [True] * 10},
            {"condition": "no_conditioning", "transfer_correct": [True] * 9 + [False]},
            {"condition": "strong_negative", "transfer_correct": [False] * 10},
            {"condition": "strong_negative", "transfer_correct": [False] * 9 + [True]},
        ]
        exp1b_corpus = [
            {"condition": "no_conditioning", "transfer_correct": [True] * 10},
            {"condition": "no_conditioning", "transfer_correct": [True] * 9 + [False]},
            # Session-2 effect near zero (null)
            {"condition": "strong_negative", "transfer_correct": [True] * 9 + [False]},
            {"condition": "strong_negative", "transfer_correct": [True] * 10},
        ]

        analysis = analyze_exp1b(
            exp1a_corpus=exp1a_corpus,
            exp1b_corpus=exp1b_corpus,
            model="dry-run",
            h1b_dual_tests=True,
            tost_epsilon=0.10,
        )

        cell = analysis["three_way_comparison"]["strong_negative"]
        # Directional test on session 2 effect
        assert "session_2_directional_p" in cell
        # TOST equivalence p-value
        assert "session_2_tost_p" in cell
        assert "session_2_equivalent" in cell
