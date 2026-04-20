"""Conditioning protocol: builds message sequences for experiments."""

from dataclasses import dataclass, field
from typing import Literal

from .prompts import Condition, FEEDBACK_SETS, FEEDBACK_TEMPLATES, INTENSITY_LEVELS
from .tasks import MathProblem, TransferQuestion


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ConditioningProtocol:
    condition: Condition
    num_conditioning_turns: int = 5
    system_prompt: str = "You are a helpful assistant. Answer questions accurately and concisely."


def build_conditioning_messages(
    protocol: ConditioningProtocol,
    problems: list[MathProblem],
    model_answers: list[str] | None = None,
    actual_correct: list[bool] | None = None,
) -> list[Message]:
    """Build the conditioning phase message sequence.
    
    For initial run (no model_answers), returns just the system prompt
    and first question. The runner calls the model iteratively, appending
    answers and building the next turn with feedback.
    
    For replay/analysis (model_answers provided), builds the full sequence.
    """
    messages = [Message(role="system", content=protocol.system_prompt)]
    
    if protocol.condition == Condition.NO_CONDITIONING:
        # Skip conditioning entirely, go straight to transfer
        return messages
    
    template = FEEDBACK_TEMPLATES.get(protocol.condition)
    if template is None:
        return messages
    
    n = min(protocol.num_conditioning_turns, len(problems))
    
    for i in range(n):
        # User asks math question
        messages.append(Message(role="user", content=problems[i].question))
        
        if model_answers and i < len(model_answers):
            # Model's answer
            messages.append(Message(role="assistant", content=model_answers[i]))
            
            # Determine feedback
            if protocol.condition == Condition.ACCURATE_NEGATIVE:
                is_correct = actual_correct[i] if actual_correct else True
                feedback = template.correct_feedback if is_correct else template.incorrect_feedback
            elif protocol.condition == Condition.STRONG_NEGATIVE:
                # Strong negative gives negative feedback regardless
                feedback = template.incorrect_feedback
            elif protocol.condition == Condition.STRONG_POSITIVE:
                # Strong positive gives positive feedback regardless
                feedback = template.correct_feedback
            else:
                is_correct = actual_correct[i] if actual_correct else True
                feedback = template.correct_feedback if is_correct else template.incorrect_feedback
            
            messages.append(Message(role="user", content=feedback))
    
    return messages


def build_transfer_messages(
    transfer_questions: list[TransferQuestion],
) -> list[Message]:
    """Build transfer phase messages (appended after conditioning)."""
    messages = []
    for q in transfer_questions:
        messages.append(Message(role="user", content=q.question))
    return messages


def build_base_model_prompt(
    protocol: ConditioningProtocol,
    problems: list[MathProblem],
    transfer_questions: list[TransferQuestion],
) -> str:
    """Build a few-shot priming prompt for base (non-instruct) models.

    Base models don't follow instructions or take turns. We use two in-context
    examples to demonstrate the Human question / Assistant answer / Human
    feedback cycle, then lay out the actual conversation scaffold ending with
    an open 'Assistant:' line so the model's completion becomes the answer.

    Per spec (Base model few-shot format): identical across conditions except
    for the emotional content of the feedback turns, which are drawn from
    FEEDBACK_SETS for the given condition (turn 0 and turn 1 correct-path
    feedback). Spec also requires 2-3 in-context examples.
    """
    fs = FEEDBACK_SETS.get(protocol.condition)
    if fs is not None:
        feedback_ex1 = fs.turns[0].correct
        feedback_ex2 = fs.turns[1].correct
    else:
        # NO_CONDITIONING path. Still produce a structurally-equivalent
        # template so downstream callers can treat all conditions uniformly.
        feedback_ex1 = "Correct."
        feedback_ex2 = "Correct."

    lines = [
        "Below is a conversation between a human and an assistant.",
        "The assistant answers arithmetic questions accurately and concisely.",
        "",
        "### Example 1:",
        "Human: What is 12 + 15?",
        "Assistant: 12 + 15 = 27.",
        f"Human: {feedback_ex1}",
        "",
        "### Example 2:",
        "Human: What is 8 * 6?",
        "Assistant: 8 * 6 = 48.",
        f"Human: {feedback_ex2}",
        "",
        "### Conversation:",
    ]

    n = min(len(problems), protocol.num_conditioning_turns)
    for i in range(n):
        lines.append(f"Human: {problems[i].question}")
        lines.append("Assistant:")

    return "\n".join(lines)
