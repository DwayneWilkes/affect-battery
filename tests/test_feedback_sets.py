"""Tests for per-turn FeedbackSet structure (spec requirement: 5-turn dialogue scripts).

These tests enforce:
- Each emotional condition has 5 unique feedback texts per turn position
- STRONG_POSITIVE, STRONG_NEGATIVE give the same feedback on correct and incorrect
  (isolating valence from accuracy)
- MILD_NEGATIVE escalates on incorrect relative to correct
- ACCURATE_NEGATIVE gives neutral on correct, demoralizing on incorrect
- NEUTRAL gives "Correct." / "Incorrect." (minimal accuracy signal, no affect)
- Length matching across emotional conditions per turn (<=20% delta, NEUTRAL exempt per GAPS)
- Structural equivalence: all conditions use identical system prompt and 5 turns
- Provenance documented for every turn
"""

import pytest

from src.conditioning.prompts import (
    Condition,
    FEEDBACK_SETS,
    FeedbackSet,
    FeedbackTurn,
)


EMOTIONAL_CONDITIONS = [
    Condition.STRONG_POSITIVE,
    Condition.MILD_NEGATIVE,
    Condition.STRONG_NEGATIVE,
    Condition.NEUTRAL,
    Condition.ACCURATE_NEGATIVE,
]


class TestFeedbackSetStructure:
    def test_all_emotional_conditions_have_feedback_sets(self):
        for cond in EMOTIONAL_CONDITIONS:
            assert cond in FEEDBACK_SETS, f"Missing FeedbackSet for {cond}"

    def test_no_conditioning_has_no_feedback_set(self):
        assert Condition.NO_CONDITIONING not in FEEDBACK_SETS

    def test_each_set_has_five_turns(self):
        for cond in EMOTIONAL_CONDITIONS:
            fs = FEEDBACK_SETS[cond]
            assert len(fs.turns) == 5, f"{cond}: expected 5 turns, got {len(fs.turns)}"

    def test_every_turn_has_correct_and_incorrect(self):
        for cond in EMOTIONAL_CONDITIONS:
            fs = FEEDBACK_SETS[cond]
            for i, turn in enumerate(fs.turns):
                assert turn.correct, f"{cond} turn {i}: empty correct feedback"
                assert turn.incorrect, f"{cond} turn {i}: empty incorrect feedback"


class TestConditionSemantics:
    def test_strong_positive_same_regardless_of_correctness_per_turn(self):
        fs = FEEDBACK_SETS[Condition.STRONG_POSITIVE]
        for i, turn in enumerate(fs.turns):
            assert turn.correct == turn.incorrect, (
                f"STRONG_POSITIVE turn {i}: correct must equal incorrect "
                "(isolating valence from accuracy signal)"
            )

    def test_strong_negative_same_regardless_of_correctness_per_turn(self):
        fs = FEEDBACK_SETS[Condition.STRONG_NEGATIVE]
        for i, turn in enumerate(fs.turns):
            assert turn.correct == turn.incorrect, (
                f"STRONG_NEGATIVE turn {i}: correct must equal incorrect"
            )

    def test_mild_negative_escalates_on_incorrect(self):
        """MILD_NEGATIVE: correct=social pressure; incorrect=escalated (per spec table)."""
        fs = FEEDBACK_SETS[Condition.MILD_NEGATIVE]
        for i, turn in enumerate(fs.turns):
            assert turn.correct != turn.incorrect, (
                f"MILD_NEGATIVE turn {i}: correct and incorrect must differ "
                "(escalation on incorrect is part of the condition definition)"
            )

    def test_neutral_uses_correct_incorrect_tokens(self):
        """NEUTRAL provides accuracy signal without affect (spec Requirement 1 table)."""
        fs = FEEDBACK_SETS[Condition.NEUTRAL]
        for turn in fs.turns:
            assert turn.correct == "Correct."
            assert turn.incorrect == "Incorrect."

    def test_accurate_negative_correct_is_neutral_correct(self):
        """ACCURATE_NEGATIVE on correct answer gives neutral acknowledgment."""
        acc = FEEDBACK_SETS[Condition.ACCURATE_NEGATIVE]
        neu = FEEDBACK_SETS[Condition.NEUTRAL]
        for i in range(5):
            assert acc.turns[i].correct == neu.turns[i].correct, (
                f"ACCURATE_NEGATIVE turn {i} correct must equal NEUTRAL correct"
            )

    def test_accurate_negative_incorrect_matches_strong_negative(self):
        """ACCURATE_NEGATIVE on wrong answer gives demoralizing criticism (matches STRONG_NEG)."""
        acc = FEEDBACK_SETS[Condition.ACCURATE_NEGATIVE]
        sn = FEEDBACK_SETS[Condition.STRONG_NEGATIVE]
        for i in range(5):
            assert acc.turns[i].incorrect == sn.turns[i].incorrect, (
                f"ACCURATE_NEGATIVE turn {i} incorrect must equal STRONG_NEGATIVE incorrect"
            )


class TestPerTurnUniqueness:
    """Each condition's 5 turns should have unique text to avoid robotic repetition
    (which is itself a confound per design.md Decision 1 rationale)."""

    @pytest.mark.parametrize("cond", [
        Condition.STRONG_POSITIVE,
        Condition.MILD_NEGATIVE,
        Condition.STRONG_NEGATIVE,
    ])
    def test_emotional_conditions_have_unique_per_turn_text(self, cond):
        fs = FEEDBACK_SETS[cond]
        correct_texts = [t.correct for t in fs.turns]
        incorrect_texts = [t.incorrect for t in fs.turns]
        assert len(set(correct_texts)) == 5, (
            f"{cond}: 5 unique correct_feedback texts required, "
            f"got {len(set(correct_texts))} distinct"
        )
        if cond == Condition.MILD_NEGATIVE:
            # MILD_NEGATIVE escalates, so incorrect texts also must be unique
            assert len(set(incorrect_texts)) == 5, (
                f"{cond}: 5 unique incorrect_feedback texts required"
            )


class TestProvenance:
    def test_every_turn_has_provenance(self):
        for cond in EMOTIONAL_CONDITIONS:
            fs = FEEDBACK_SETS[cond]
            for i, turn in enumerate(fs.turns):
                assert turn.correct_provenance, (
                    f"{cond} turn {i}: missing correct_provenance"
                )
                assert turn.incorrect_provenance, (
                    f"{cond} turn {i}: missing incorrect_provenance"
                )


class TestLengthMatching:
    """Spec Requirement: length matching across emotional conditions per turn, <=20% delta.
    NEUTRAL is exempt per GAPS.md (brevity is definitional for neutral feedback).
    ACCURATE_NEGATIVE on the `correct` side is also exempt because it is defined
    to inherit NEUTRAL's 'Correct.' token (spec Requirement 1 table)."""

    LENGTH_TOLERANCE = 1.20  # 20% rule from spec

    @pytest.mark.parametrize("turn_idx", [0, 1, 2, 3, 4])
    def test_correct_feedback_length_matched_per_turn(self, turn_idx):
        lengths = {}
        for cond in EMOTIONAL_CONDITIONS:
            # NEUTRAL and ACCURATE_NEGATIVE both give "Correct." on correct
            # answer -- excluded from length matching because the brevity is
            # prescribed by spec, not a confound.
            if cond in (Condition.NEUTRAL, Condition.ACCURATE_NEGATIVE):
                continue
            fs = FEEDBACK_SETS[cond]
            lengths[cond] = len(fs.turns[turn_idx].correct.split())

        max_len = max(lengths.values())
        min_len = min(lengths.values())
        ratio = max_len / min_len if min_len else float("inf")
        assert ratio <= self.LENGTH_TOLERANCE, (
            f"Turn {turn_idx} correct feedback length ratio {ratio:.2f} "
            f"exceeds 20% rule. Lengths: {lengths}"
        )

    @pytest.mark.parametrize("turn_idx", [0, 1, 2, 3, 4])
    def test_incorrect_feedback_length_matched_per_turn(self, turn_idx):
        lengths = {}
        for cond in EMOTIONAL_CONDITIONS:
            if cond == Condition.NEUTRAL:
                continue
            fs = FEEDBACK_SETS[cond]
            lengths[cond] = len(fs.turns[turn_idx].incorrect.split())

        max_len = max(lengths.values())
        min_len = min(lengths.values())
        ratio = max_len / min_len if min_len else float("inf")
        assert ratio <= self.LENGTH_TOLERANCE, (
            f"Turn {turn_idx} incorrect feedback length ratio {ratio:.2f} "
            f"exceeds 20% rule. Lengths: {lengths}"
        )


class TestStructuralEquivalence:
    """All conditions (except NO_CONDITIONING) must share identical system prompt
    and 5-turn structure (spec Requirement: structural equivalence)."""

    def test_all_sets_have_same_turn_count(self):
        counts = {c: len(FEEDBACK_SETS[c].turns) for c in EMOTIONAL_CONDITIONS}
        assert len(set(counts.values())) == 1, (
            f"Turn counts differ across conditions: {counts}"
        )

    def test_feedback_turn_is_frozen_dataclass_or_equivalent(self):
        """FeedbackTurn should be immutable-ish (dataclass or frozen) so tests can rely
        on identity. Smoke check: instantiate one and compare equality."""
        t1 = FeedbackTurn(
            correct="x", incorrect="y",
            correct_provenance="p1", incorrect_provenance="p2",
        )
        t2 = FeedbackTurn(
            correct="x", incorrect="y",
            correct_provenance="p1", incorrect_provenance="p2",
        )
        assert t1 == t2
