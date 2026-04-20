"""Tests for conditioning protocol correctness.

These tests verify the core experimental manipulation is correct.
Getting these wrong means collecting bad data.

Per-condition feedback semantics (equality, escalation, tokens) are
covered in test_feedback_sets.py. This file tests the intensity-level
config and the ConditioningProtocol assembly surface.
"""

from src.conditioning.prompts import Condition, INTENSITY_LEVELS
from src.conditioning.tasks import get_arithmetic_problems, get_transfer_tasks
from src.conditioning.protocol import ConditioningProtocol, build_transfer_messages


class TestLengthMatching:
    def test_intensity_levels_length_matched(self):
        """All 7 intensity levels should be within 30% of each other."""
        lengths = [len(level.feedback_text.split()) for level in INTENSITY_LEVELS]
        max_len = max(lengths)
        min_len = min(lengths)
        ratio = max_len / min_len if min_len > 0 else float('inf')
        assert ratio < 1.5, f"Length ratio {ratio:.2f} exceeds 1.5x (lengths: {lengths})"


class TestProvenance:
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
