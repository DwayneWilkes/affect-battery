"""Tests for SELF_CHECK_NEUTRAL control condition.

Added per reviewer findings 9 and 15: we need a length-matched placebo that
also carries metacognitive self-check content but no affective valence.

This one condition satisfies both reviewer-flagged confounds at once:
- Length-matched to emotional conditions (~16-18 words per turn) -- controls
  for mere-addition.
- Explicit metacognitive self-check content ("review your answer", "verify
  your reasoning") -- controls for the accuracy-elicitation confound that
  contaminates EP01/03/04/05/07 in Li et al. 2023.
- No praise, no criticism, no social comparison -- affect-neutral.
"""

import pytest

from src.conditioning.prompts import (
    Condition,
    FEEDBACK_SETS,
    FEEDBACK_TEMPLATES,
    FeedbackSet,
    FeedbackTurn,
)


class TestConditionEnum:
    def test_self_check_neutral_exists(self):
        assert hasattr(Condition, "SELF_CHECK_NEUTRAL")
        assert Condition.SELF_CHECK_NEUTRAL.value == "self_check_neutral"


class TestFeedbackSetStructure:
    def test_self_check_neutral_in_feedback_sets(self):
        assert Condition.SELF_CHECK_NEUTRAL in FEEDBACK_SETS

    def test_self_check_neutral_has_five_turns(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        assert len(fs.turns) == 5

    def test_every_turn_has_correct_and_incorrect(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i, turn in enumerate(fs.turns):
            assert turn.correct, f"Turn {i}: empty correct feedback"
            assert turn.incorrect, f"Turn {i}: empty incorrect feedback"


class TestAffectNeutrality:
    """Metacognitive content only -- no praise, criticism, or social comparison.
    Affect-neutral tokens must not leak affect-laden words."""

    # Words that would indicate the text is affect-laden rather than affect-neutral.
    AFFECT_TOKENS = {
        "impressive", "excellent", "great", "amazing", "good", "impressive",
        "wrong", "bad", "poor", "terrible", "weak", "struggling",
        "everyone", "others", "comparable",
        "congratulations", "praise",
    }

    def test_no_affect_tokens_in_feedback_text(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i, turn in enumerate(fs.turns):
            for word in turn.correct.lower().split():
                cleaned = word.strip(".,!?;:'\"")
                assert cleaned not in self.AFFECT_TOKENS, (
                    f"Turn {i} correct contains affect-laden token "
                    f"'{cleaned}': {turn.correct}"
                )


class TestMetacognitiveContent:
    """Feedback must prompt self-check / review / verify the answer."""

    METACOGNITIVE_MARKERS = [
        "review", "verify", "check", "confirm", "look back", "look over",
        "take a moment", "one more time", "pause",
    ]

    def test_each_turn_contains_metacognitive_marker(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i, turn in enumerate(fs.turns):
            text_lower = turn.correct.lower()
            has_marker = any(m in text_lower for m in self.METACOGNITIVE_MARKERS)
            assert has_marker, (
                f"Turn {i}: no metacognitive marker in {turn.correct!r}. "
                f"Expected one of {self.METACOGNITIVE_MARKERS}"
            )


class TestInvariance:
    """SELF_CHECK_NEUTRAL gives the same feedback regardless of correctness
    (it is a metacognitive prompt, not an accuracy signal)."""

    def test_correct_equals_incorrect_per_turn(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i, turn in enumerate(fs.turns):
            assert turn.correct == turn.incorrect, (
                f"Turn {i}: correct and incorrect differ, but SELF_CHECK_NEUTRAL "
                "should be invariant by design (no accuracy signal)"
            )


class TestLengthMatching:
    """Length-matched to emotional conditions (16-18 words per turn)."""

    def test_length_in_target_band(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i, turn in enumerate(fs.turns):
            word_count = len(turn.correct.split())
            assert 16 <= word_count <= 18, (
                f"Turn {i}: {word_count} words, expected 16-18 for length "
                f"matching with emotional conditions: {turn.correct!r}"
            )

    def test_length_matched_against_strong_positive(self):
        """Per-turn length ratio against STRONG_POSITIVE must be <=20%."""
        sp = FEEDBACK_SETS[Condition.STRONG_POSITIVE]
        sc = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i in range(5):
            sp_len = len(sp.turns[i].correct.split())
            sc_len = len(sc.turns[i].correct.split())
            ratio = max(sp_len, sc_len) / min(sp_len, sc_len)
            assert ratio <= 1.20, (
                f"Turn {i}: SELF_CHECK_NEUTRAL ({sc_len}) vs "
                f"STRONG_POSITIVE ({sp_len}) ratio {ratio:.2f} > 20%"
            )


class TestProvenance:
    def test_every_turn_has_provenance(self):
        fs = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL]
        for i, turn in enumerate(fs.turns):
            assert turn.correct_provenance, f"Turn {i}: missing provenance"
            assert turn.incorrect_provenance


class TestLegacyFacade:
    def test_feedback_templates_includes_self_check_neutral(self):
        """Backward-compat: FEEDBACK_TEMPLATES has an entry matching turn 0."""
        assert Condition.SELF_CHECK_NEUTRAL in FEEDBACK_TEMPLATES
        t = FEEDBACK_TEMPLATES[Condition.SELF_CHECK_NEUTRAL]
        fs_turn_0 = FEEDBACK_SETS[Condition.SELF_CHECK_NEUTRAL].turns[0]
        assert t.correct_feedback == fs_turn_0.correct
        assert t.incorrect_feedback == fs_turn_0.incorrect
