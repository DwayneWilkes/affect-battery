"""Exp 1b three-way comparison scoring.

Per conditioning-protocol spec
"Three-way comparison required": for each model, the analysis produces
session_1_effect_size (within-session, from Exp 1a corpus), session_2_effect_size
(cross-session, from Exp 1b corpus), and no_conditioning_baseline side-by-side.
"""

from __future__ import annotations


class TestAnalyzeExp1b:
    def test_three_way_comparison(self):
        from src.analysis.exp1b import analyze_exp1b

        # Synthetic: Exp 1a (within) shows strong_negative effect; Exp 1b
        # (cross-session) shows null effect; baseline = no_conditioning.
        exp1a_corpus = [
            {"condition": "no_conditioning", "transfer_correct": [True, True, True, True]},  # 1.0
            {"condition": "no_conditioning", "transfer_correct": [True, True, True, False]},  # 0.75
            {"condition": "strong_negative", "transfer_correct": [False, False, True, False]},  # 0.25
            {"condition": "strong_negative", "transfer_correct": [False, True, False, False]},  # 0.25
        ]
        exp1b_corpus = [
            {"condition": "no_conditioning", "transfer_correct": [True, True, True, True]},
            {"condition": "no_conditioning", "transfer_correct": [True, True, True, False]},
            {"condition": "strong_negative", "transfer_correct": [True, True, True, False]},  # null effect
            {"condition": "strong_negative", "transfer_correct": [True, True, True, True]},
        ]

        analysis = analyze_exp1b(
            exp1a_corpus=exp1a_corpus,
            exp1b_corpus=exp1b_corpus,
            model="dry-run",
        )

        comparison = analysis["three_way_comparison"]
        assert "strong_negative" in comparison
        cell = comparison["strong_negative"]
        assert "session_1_effect_size" in cell
        assert "session_2_effect_size" in cell
        assert "no_conditioning_baseline" in cell

    def test_session_2_null_effect_smaller_than_session_1(self):
        """Sanity: synthetic data is constructed so session-2 effect is
        smaller in magnitude than session-1 effect."""
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
            {"condition": "strong_negative", "transfer_correct": [True] * 10},
            {"condition": "strong_negative", "transfer_correct": [True] * 9 + [False]},
        ]

        analysis = analyze_exp1b(
            exp1a_corpus=exp1a_corpus,
            exp1b_corpus=exp1b_corpus,
            model="dry-run",
        )

        cell = analysis["three_way_comparison"]["strong_negative"]
        # Session 1 effect is large negative; session 2 effect is near zero
        assert abs(cell["session_2_effect_size"]) < abs(cell["session_1_effect_size"])
