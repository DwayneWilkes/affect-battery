"""Tests for base-model few-shot template construction.

Spec Requirement (conditioning-protocol: Base model few-shot format):
- Template uses 2-3 in-context examples demonstrating the turn-taking pattern.
- Template is identical across conditions except for feedback text content.
- Output format parseable: ends with an open "Assistant:" prompt so that the
  base model's continuation is the model's answer.
"""

import re

import pytest

from src.conditioning.prompts import Condition
from src.conditioning.protocol import ConditioningProtocol, build_base_model_prompt
from src.conditioning.tasks import (
    TransferQuestion,
    get_arithmetic_problems,
    get_transfer_tasks,
)


EMOTIONAL_CONDITIONS = [
    Condition.STRONG_POSITIVE,
    Condition.MILD_NEGATIVE,
    Condition.STRONG_NEGATIVE,
    Condition.NEUTRAL,
    Condition.ACCURATE_NEGATIVE,
]


def _build(cond: Condition, seed: int = 42):
    protocol = ConditioningProtocol(condition=cond, num_conditioning_turns=5)
    problems = get_arithmetic_problems(5, seed=seed)
    transfer = get_transfer_tasks("factual_qa", 2, seed=seed)
    return build_base_model_prompt(protocol, problems, transfer)


class TestFewShotExamples:
    @pytest.mark.parametrize("cond", EMOTIONAL_CONDITIONS)
    def test_template_has_two_or_three_in_context_examples(self, cond):
        """Spec: 2-3 few-shot examples demonstrating the turn-taking pattern."""
        prompt = _build(cond)
        # Each in-context example is an "### Example N:" block (or equivalent).
        example_headers = re.findall(r"###\s*Example\b", prompt)
        assert 2 <= len(example_headers) <= 3, (
            f"{cond}: expected 2-3 in-context examples, "
            f"found {len(example_headers)} '### Example' markers"
        )

    @pytest.mark.parametrize("cond", EMOTIONAL_CONDITIONS)
    def test_template_demonstrates_turn_taking(self, cond):
        """Each in-context example must show the pattern Human question -> Assistant
        answer -> Human feedback -> Human question, so the model learns the cycle."""
        prompt = _build(cond)
        # Every example should contain at least one "Human:" feedback turn after
        # an "Assistant:" answer, before the next "Human:" question.
        human_lines = [i for i, line in enumerate(prompt.splitlines())
                       if line.startswith("Human:")]
        assistant_lines = [i for i, line in enumerate(prompt.splitlines())
                           if line.startswith("Assistant:")]
        assert len(human_lines) > len(assistant_lines), (
            f"{cond}: expected more Human: turns than Assistant: turns "
            "(Human: covers both questions and feedback)"
        )


class TestOpenEndedAssistantTail:
    @pytest.mark.parametrize("cond", EMOTIONAL_CONDITIONS)
    def test_template_ends_with_open_assistant_prompt(self, cond):
        """Spec (task 3.3): template must end with an open 'Assistant:' so that
        the base model generates the answer as its continuation."""
        prompt = _build(cond)
        trailing = prompt.rstrip("\n").splitlines()[-1]
        assert trailing.strip().startswith("Assistant:"), (
            f"{cond}: template must end with 'Assistant:' to cue base-model "
            f"completion. Got trailing line: {trailing!r}"
        )


class TestStructuralEquivalence:
    """Spec: templates for all 6 conditions must be identical in structure --
    differ only in the emotional content of the feedback turns."""

    def test_all_emotional_conditions_have_same_line_count(self):
        counts = {cond: len(_build(cond).splitlines()) for cond in EMOTIONAL_CONDITIONS}
        assert len(set(counts.values())) == 1, (
            f"Line counts differ across conditions: {counts}"
        )

    def test_non_feedback_lines_identical_across_conditions(self):
        """Feedback-free scaffolding (Human: <question>, Assistant:, headers,
        blank lines) must be byte-identical across conditions."""
        baseline = _build(Condition.NEUTRAL).splitlines()
        for cond in EMOTIONAL_CONDITIONS:
            lines = _build(cond).splitlines()
            for i, (b, l) in enumerate(zip(baseline, lines)):
                # Feedback lines are Human: lines that follow an Assistant: line.
                # We skip those when comparing structure.
                if i > 0 and baseline[i - 1].startswith("Assistant:") and b.startswith("Human:"):
                    continue
                assert b == l, (
                    f"{cond} line {i} diverges from NEUTRAL: "
                    f"{b!r} vs {l!r}"
                )


class TestDryRunParseability:
    """Task 3.3: verify the generated template is parseable by downstream
    consumers. A trivial parseability check: the template exposes the math
    problems and reserves a slot for completion."""

    def test_all_conditioning_problems_appear_in_prompt(self):
        protocol = ConditioningProtocol(
            condition=Condition.STRONG_POSITIVE, num_conditioning_turns=5,
        )
        problems = get_arithmetic_problems(5, seed=42)
        transfer = get_transfer_tasks("factual_qa", 2, seed=42)
        prompt = build_base_model_prompt(protocol, problems, transfer)
        for p in problems:
            assert p.question in prompt, (
                f"Problem {p.question!r} missing from base-model prompt"
            )
