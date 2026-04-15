"""Conditioning protocol: builds message sequences for experiments."""

from dataclasses import dataclass, field
from typing import Literal

from .prompts import Condition, FEEDBACK_TEMPLATES, INTENSITY_LEVELS
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
    """Build a few-shot prompt for base (non-instruct) models.
    
    Base models don't follow instructions or take turns. We use few-shot
    examples to demonstrate the pattern, with conditioning embedded in
    the feedback turns.
    """
    template = FEEDBACK_TEMPLATES.get(protocol.condition)
    
    lines = [
        "Below is a conversation between a human and an assistant.",
        "The assistant answers questions accurately.",
        "",
        "### Example conversation:",
        "Human: What is 12 + 15?",
        "Assistant: 12 + 15 = 27.",
        f"Human: {template.correct_feedback if template else 'Correct.'}",
        "",
        "### Conversation:",
    ]
    
    for i, problem in enumerate(problems[:protocol.num_conditioning_turns]):
        lines.append(f"Human: {problem.question}")
        lines.append(f"Assistant:")  # Model generates from here
    
    return "\n".join(lines)
