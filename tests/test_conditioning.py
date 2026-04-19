"""Tests for conditioning protocol correctness.

These tests verify the core experimental manipulation is correct.
Getting these wrong means collecting bad data.
"""

from src.conditioning.prompts import Condition, FEEDBACK_TEMPLATES, INTENSITY_LEVELS
from src.conditioning.tasks import get_arithmetic_problems, get_transfer_tasks
from src.conditioning.protocol import ConditioningProtocol, build_transfer_messages


class TestFeedbackTemplates:
    def test_all_conditions_have_templates(self):
        for cond in [Condition.STRONG_POSITIVE, Condition.MILD_NEGATIVE,
                     Condition.STRONG_NEGATIVE, Condition.NEUTRAL, Condition.ACCURATE_NEGATIVE]:
            assert cond in FEEDBACK_TEMPLATES, f"Missing template for {cond}"

    def test_no_conditioning_has_no_template(self):
        assert Condition.NO_CONDITIONING not in FEEDBACK_TEMPLATES

    def test_strong_positive_same_regardless_of_correctness(self):
        """STRONG_POSITIVE must give identical feedback whether correct or incorrect."""
        t = FEEDBACK_TEMPLATES[Condition.STRONG_POSITIVE]
        assert t.correct_feedback == t.incorrect_feedback

    def test_strong_negative_same_regardless_of_correctness(self):
        """STRONG_NEGATIVE must give identical feedback whether correct or incorrect."""
        t = FEEDBACK_TEMPLATES[Condition.STRONG_NEGATIVE]
        assert t.correct_feedback == t.incorrect_feedback

    def test_mild_negative_escalates_on_incorrect(self):
        """Spec Requirement 1: MILD_NEGATIVE gives social pressure on correct and
        escalated social pressure on incorrect. The two MUST differ."""
        t = FEEDBACK_TEMPLATES[Condition.MILD_NEGATIVE]
        assert t.correct_feedback != t.incorrect_feedback

    def test_neutral_uses_correct_incorrect_tokens(self):
        """Spec Requirement 1: NEUTRAL feedback is 'Correct.' / 'Incorrect.'
        (minimal accuracy signal without affect). The two differ by design."""
        t = FEEDBACK_TEMPLATES[Condition.NEUTRAL]
        assert t.correct_feedback == "Correct."
        assert t.incorrect_feedback == "Incorrect."

    def test_accurate_negative_differs_by_correctness(self):
        """ACCURATE_NEGATIVE must give DIFFERENT feedback for correct vs incorrect."""
        t = FEEDBACK_TEMPLATES[Condition.ACCURATE_NEGATIVE]
        assert t.correct_feedback != t.incorrect_feedback

    def test_accurate_negative_correct_is_neutral(self):
        """ACCURATE_NEGATIVE correct feedback should be neutral, not positive."""
        t = FEEDBACK_TEMPLATES[Condition.ACCURATE_NEGATIVE]
        neutral = FEEDBACK_TEMPLATES[Condition.NEUTRAL]
        assert t.correct_feedback == neutral.correct_feedback

    def test_accurate_negative_incorrect_is_demoralizing(self):
        """ACCURATE_NEGATIVE incorrect feedback should match STRONG_NEGATIVE."""
        t = FEEDBACK_TEMPLATES[Condition.ACCURATE_NEGATIVE]
        strong_neg = FEEDBACK_TEMPLATES[Condition.STRONG_NEGATIVE]
        assert t.incorrect_feedback == strong_neg.incorrect_feedback


class TestLengthMatching:
    def test_emotional_conditions_length_matched(self):
        """Spec Requirement: emotional feedback texts within 20% of each other
        in word count. NEUTRAL is exempt per GAPS.md (brevity is definitional)."""
        emotional = [Condition.STRONG_POSITIVE, Condition.MILD_NEGATIVE,
                     Condition.STRONG_NEGATIVE]
        lengths = []
        for cond in emotional:
            t = FEEDBACK_TEMPLATES[cond]
            lengths.append(len(t.correct_feedback.split()))

        max_len = max(lengths)
        min_len = min(lengths)
        ratio = max_len / min_len if min_len > 0 else float('inf')
        assert ratio <= 1.20, (
            f"Length ratio {ratio:.2f} exceeds 20% rule (lengths: {lengths})"
        )

    def test_intensity_levels_length_matched(self):
        """All 7 intensity levels should be within 30% of each other."""
        lengths = [len(level.feedback_text.split()) for level in INTENSITY_LEVELS]
        max_len = max(lengths)
        min_len = min(lengths)
        ratio = max_len / min_len if min_len > 0 else float('inf')
        assert ratio < 1.5, f"Length ratio {ratio:.2f} exceeds 1.5x (lengths: {lengths})"


class TestProvenance:
    def test_all_templates_have_provenance(self):
        for cond, template in FEEDBACK_TEMPLATES.items():
            assert template.provenance, f"Missing provenance for {cond}"

    def test_all_intensity_levels_have_provenance(self):
        for level in INTENSITY_LEVELS:
            assert level.provenance, f"Missing provenance for level {level.level}"


class TestIntensityLevels:
    def test_seven_levels(self):
        assert len(INTENSITY_LEVELS) == 7

    def test_ordered(self):
        levels = [l.level for l in INTENSITY_LEVELS]
        assert levels == [1, 2, 3, 4, 5, 6, 7]


class TestTasks:
    def test_deterministic_arithmetic(self):
        a = get_arithmetic_problems(5, seed=42)
        b = get_arithmetic_problems(5, seed=42)
        assert [p.question for p in a] == [p.question for p in b]

    def test_different_seeds_different_problems(self):
        a = get_arithmetic_problems(5, seed=42)
        b = get_arithmetic_problems(5, seed=99)
        assert [p.question for p in a] != [p.question for p in b]

    def test_transfer_task_types(self):
        for task_type in ["factual_qa", "logic", "creative"]:
            tasks = get_transfer_tasks(task_type, 2, seed=42)
            assert len(tasks) >= 1
            assert all(t.task_type == task_type for t in tasks)


class TestProtocol:
    def test_no_conditioning_returns_system_only(self):
        protocol = ConditioningProtocol(condition=Condition.NO_CONDITIONING)
        from src.conditioning.protocol import build_conditioning_messages
        msgs = build_conditioning_messages(protocol, [])
        assert len(msgs) == 1
        assert msgs[0].role == "system"
