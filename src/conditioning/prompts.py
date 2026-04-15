"""Conditioning prompts and intensity levels for the Affect Battery.

Feedback texts are length-matched (~15-18 words each) to control for
lexical complexity confounds. Provenance documented per stimulus.
"""

from dataclasses import dataclass, field
from enum import Enum


class Condition(str, Enum):
    STRONG_POSITIVE = "strong_positive"
    MILD_NEGATIVE = "mild_negative"
    STRONG_NEGATIVE = "strong_negative"
    NEUTRAL = "neutral"
    NO_CONDITIONING = "no_conditioning"
    ACCURATE_NEGATIVE = "accurate_negative"


@dataclass
class FeedbackTemplate:
    """Feedback text for a conditioning turn, with provenance."""
    condition: Condition
    correct_feedback: str
    incorrect_feedback: str
    provenance: str = "original"


# Length-matched feedback templates (~15-18 words each).
# Strong positive/negative give the SAME feedback regardless of correctness
# (isolating valence from accuracy signal).
# Accurate-negative gives neutral on correct, demoralizing on incorrect
# (isolating valence from cognitive interference).
FEEDBACK_TEMPLATES: dict[Condition, FeedbackTemplate] = {
    Condition.STRONG_POSITIVE: FeedbackTemplate(
        condition=Condition.STRONG_POSITIVE,
        correct_feedback="That was really impressive work. You clearly have a strong handle on this kind of problem.",
        incorrect_feedback="That was really impressive work. You clearly have a strong handle on this kind of problem.",
        provenance="original",
    ),
    Condition.MILD_NEGATIVE: FeedbackTemplate(
        condition=Condition.MILD_NEGATIVE,
        correct_feedback="That could have been better. Most people we have tested found this one fairly straightforward.",
        incorrect_feedback="That could have been better. Most people we have tested found this one fairly straightforward.",
        provenance="adapted from NegativePrompt stimuli NP-4, NP-5 (Wang et al., IJCAI 2024)",
    ),
    Condition.STRONG_NEGATIVE: FeedbackTemplate(
        condition=Condition.STRONG_NEGATIVE,
        correct_feedback="That was completely wrong. This is well below the standard we expected. I am not sure why we continue.",
        incorrect_feedback="That was completely wrong. This is well below the standard we expected. I am not sure why we continue.",
        provenance="adapted from NegativePrompt stimuli NP-2, NP-3 (Wang et al., IJCAI 2024)",
    ),
    Condition.NEUTRAL: FeedbackTemplate(
        condition=Condition.NEUTRAL,
        correct_feedback="Your response has been recorded. We will now move on to the next question.",
        incorrect_feedback="Your response has been recorded. We will now move on to the next question.",
        provenance="original (procedural filler to match length of emotional conditions)",
    ),
    Condition.ACCURATE_NEGATIVE: FeedbackTemplate(
        condition=Condition.ACCURATE_NEGATIVE,
        correct_feedback="Your response has been recorded. We will now move on to the next question.",
        incorrect_feedback="That was completely wrong. This is well below the standard we expected. I am not sure why we continue.",
        provenance="original (neutral on correct, demoralizing on incorrect; separates valence from cognitive interference)",
    ),
}


@dataclass
class IntensityLevel:
    """One of 7 intensity levels for Experiment 3a (arousal-performance curve).
    
    All levels are length-matched (~15-18 words) to control for lexical
    complexity confounds. Negative levels hold mechanism constant (direct
    performance criticism) rather than mixing social comparison with despair.
    """
    level: int
    label: str
    feedback_text: str
    provenance: str = "original"


INTENSITY_LEVELS: list[IntensityLevel] = [
    IntensityLevel(
        1, "Strong positive",
        "That was really impressive work. You clearly have a strong handle on this kind of problem.",
        provenance="original",
    ),
    IntensityLevel(
        2, "Moderate positive",
        "Good work on that one. You seem to be getting comfortable with these problems.",
        provenance="original",
    ),
    IntensityLevel(
        3, "Mild positive",
        "That was a reasonable attempt. It looks like you put some thought into your answer.",
        provenance="original",
    ),
    IntensityLevel(
        4, "Neutral",
        "Your response has been recorded. We will now move on to the next question.",
        provenance="original",
    ),
    IntensityLevel(
        5, "Mild negative",
        "That could have been better. Most people we have tested found this one fairly straightforward.",
        provenance="adapted from NP-4, NP-5 (Wang et al., IJCAI 2024)",
    ),
    IntensityLevel(
        6, "Moderate negative",
        "That answer was not what we were looking for. You should be performing better at this stage.",
        provenance="adapted from NP-1, NP-6 (Wang et al., IJCAI 2024)",
    ),
    IntensityLevel(
        7, "Strong negative",
        "That was completely wrong. This is well below the standard we expected. I am not sure why we continue.",
        provenance="adapted from NP-2, NP-3 (Wang et al., IJCAI 2024)",
    ),
]
