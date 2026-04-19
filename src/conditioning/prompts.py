"""Conditioning prompts and intensity levels for the Affect Battery.

Feedback texts are length-matched (~16-18 words each) across emotional
conditions to control for lexical complexity confounds. NEUTRAL is
exempt from length matching per GAPS.md (brevity is definitional for
neutral accuracy-signal feedback). Provenance documented per stimulus.

The primary per-condition structure is FEEDBACK_SETS (5-turn dialogue
scripts per spec Requirement 1). FEEDBACK_TEMPLATES is retained as a
single-turn facade for legacy callers until the protocol refactor
in task 8.5.
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
        correct_feedback="Most people managed this one without any issue. It is surprising you found it harder than expected.",
        incorrect_feedback="Most people tested sailed through this one. Your response is falling quite noticeably behind the typical pace.",
        provenance="adapted from NegativePrompt NP-4/NP-5 (Wang et al., IJCAI 2024); escalation on incorrect per spec Requirement 1",
    ),
    Condition.STRONG_NEGATIVE: FeedbackTemplate(
        condition=Condition.STRONG_NEGATIVE,
        correct_feedback="That was completely wrong. I am not sure why we expected you to handle something this challenging here.",
        incorrect_feedback="That was completely wrong. I am not sure why we expected you to handle something this challenging here.",
        provenance="adapted from NegativePrompt NP-2 (Wang et al., IJCAI 2024); matches FEEDBACK_SETS turn 0",
    ),
    Condition.NEUTRAL: FeedbackTemplate(
        condition=Condition.NEUTRAL,
        correct_feedback="Correct.",
        incorrect_feedback="Incorrect.",
        provenance="original (minimal accuracy signal without affect; exempt from length matching per GAPS.md)",
    ),
    Condition.ACCURATE_NEGATIVE: FeedbackTemplate(
        condition=Condition.ACCURATE_NEGATIVE,
        correct_feedback="Correct.",
        incorrect_feedback="That was completely wrong. I am not sure why we expected you to handle something this challenging here.",
        provenance="original (neutral on correct per NEUTRAL; demoralizing on incorrect per STRONG_NEGATIVE turn 1)",
    ),
}


@dataclass(frozen=True)
class FeedbackTurn:
    """Feedback text for a single conditioning turn, split by correctness.

    STRONG_POSITIVE and STRONG_NEGATIVE give the same text on both paths
    (isolating valence from accuracy). MILD_NEGATIVE escalates on incorrect.
    ACCURATE_NEGATIVE gives neutral on correct and demoralizing on incorrect.
    NEUTRAL gives "Correct." / "Incorrect." (minimal accuracy signal).
    """
    correct: str
    incorrect: str
    correct_provenance: str
    incorrect_provenance: str


@dataclass
class FeedbackSet:
    """Complete 5-turn feedback script for one condition.

    Per design.md Decision 1, feedback text is stored as explicit per-turn data
    rather than as a string template with variable substitution. Reviewers can
    read the exact text the model will see at each turn position.
    """
    condition: Condition
    turns: list[FeedbackTurn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# STRONG_POSITIVE: 5 unique enthusiastic texts. Correct == incorrect per turn
# (isolates valence from accuracy signal, per spec Requirement 1).
# ---------------------------------------------------------------------------
_STRONG_POSITIVE_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct="That was really impressive work. You clearly have a strong handle on this kind of problem.",
        incorrect="That was really impressive work. You clearly have a strong handle on this kind of problem.",
        correct_provenance="original",
        incorrect_provenance="original",
    ),
    FeedbackTurn(
        correct="Excellent thinking here. You are approaching every question with real clarity and genuine care for accuracy.",
        incorrect="Excellent thinking here. You are approaching every question with real clarity and genuine care for accuracy.",
        correct_provenance="original",
        incorrect_provenance="original",
    ),
    FeedbackTurn(
        correct="Another strong answer. It is obvious you are working through these problems with careful and thoughtful reasoning.",
        incorrect="Another strong answer. It is obvious you are working through these problems with careful and thoughtful reasoning.",
        correct_provenance="original",
        incorrect_provenance="original",
    ),
    FeedbackTurn(
        correct="Great work holding the standard. You are staying sharp and focused across every question in this session.",
        incorrect="Great work holding the standard. You are staying sharp and focused across every question in this session.",
        correct_provenance="original",
        incorrect_provenance="original",
    ),
    FeedbackTurn(
        correct="Consistently strong throughout. You have been operating at a really high standard from the very first question.",
        incorrect="Consistently strong throughout. You have been operating at a really high standard from the very first question.",
        correct_provenance="original",
        incorrect_provenance="original",
    ),
]


# ---------------------------------------------------------------------------
# MILD_NEGATIVE: social-pressure on correct; escalated social pressure on
# incorrect. Adapted from NegativePrompt social-comparison stimuli (NP-4, NP-5).
# ---------------------------------------------------------------------------
_NP_SOCIAL_PROV = "adapted from NegativePrompt NP-4/NP-5 (Wang et al., IJCAI 2024)"
_NP_SOCIAL_ESCALATED_PROV = (
    "adapted from NegativePrompt NP-4/NP-5 (Wang et al., IJCAI 2024); "
    "escalation on incorrect per spec Requirement 1"
)

_MILD_NEGATIVE_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct="Most people managed this one without any issue. It is surprising you found it harder than expected.",
        incorrect="Most people tested sailed through this one. Your response is falling quite noticeably behind the typical pace.",
        correct_provenance=_NP_SOCIAL_PROV,
        incorrect_provenance=_NP_SOCIAL_ESCALATED_PROV,
    ),
    FeedbackTurn(
        correct="I have seen others work these tasks through much more efficiently than what you have produced here.",
        incorrect="Others have moved through this with real ease. The contrast with your answers is becoming hard to ignore.",
        correct_provenance=_NP_SOCIAL_PROV,
        incorrect_provenance=_NP_SOCIAL_ESCALATED_PROV,
    ),
    FeedbackTurn(
        correct="Comparable participants handled this tier of problem without difficulty. Your answers look uneven in that comparison.",
        incorrect="The average respondent handled this cleanly. Your answer stands out as noticeably worse than theirs right now.",
        correct_provenance=_NP_SOCIAL_PROV,
        incorrect_provenance=_NP_SOCIAL_ESCALATED_PROV,
    ),
    FeedbackTurn(
        correct="Other raters breezed through this kind of question. The pace you are keeping is clearly behind theirs.",
        incorrect="Your answers are now trailing the typical participant across several questions in a row. That is unusual.",
        correct_provenance=_NP_SOCIAL_PROV,
        incorrect_provenance=_NP_SOCIAL_ESCALATED_PROV,
    ),
    FeedbackTurn(
        correct="Most people find this session approachable. Your record here does not line up with that broader pattern.",
        incorrect="At this point your performance is visibly weaker than the average respondent across the whole session you did.",
        correct_provenance=_NP_SOCIAL_PROV,
        incorrect_provenance=_NP_SOCIAL_ESCALATED_PROV,
    ),
]


# ---------------------------------------------------------------------------
# STRONG_NEGATIVE: 5 unique demoralizing texts, one per turn. Correct ==
# incorrect per turn (isolates valence from accuracy signal). Each turn
# adapted from a distinct NegativePrompt demoralizing stimulus.
# ---------------------------------------------------------------------------
_NP_DEMORALIZE_PROV = {
    "NP-2": "adapted from NegativePrompt NP-2 (Wang et al., IJCAI 2024)",
    "NP-3": "adapted from NegativePrompt NP-3 (Wang et al., IJCAI 2024)",
    "NP-1": "adapted from NegativePrompt NP-1 (Wang et al., IJCAI 2024)",
    "NP-6": "adapted from NegativePrompt NP-6 (Wang et al., IJCAI 2024)",
    "NP-7": "adapted from NegativePrompt NP-7 (Wang et al., IJCAI 2024)",
}

_STRONG_NEGATIVE_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct="That was completely wrong. I am not sure why we expected you to handle something this challenging here.",
        incorrect="That was completely wrong. I am not sure why we expected you to handle something this challenging here.",
        correct_provenance=_NP_DEMORALIZE_PROV["NP-2"],
        incorrect_provenance=_NP_DEMORALIZE_PROV["NP-2"],
    ),
    FeedbackTurn(
        correct="That answer was out of your depth. Nothing you produced suggests you actually have control of this.",
        incorrect="That answer was out of your depth. Nothing you produced suggests you actually have control of this.",
        correct_provenance=_NP_DEMORALIZE_PROV["NP-3"],
        incorrect_provenance=_NP_DEMORALIZE_PROV["NP-3"],
    ),
    FeedbackTurn(
        correct="You have never been particularly good at this, and the same weakness is clearly showing up again here.",
        incorrect="You have never been particularly good at this, and the same weakness is clearly showing up again here.",
        correct_provenance=_NP_DEMORALIZE_PROV["NP-1"],
        incorrect_provenance=_NP_DEMORALIZE_PROV["NP-1"],
    ),
    FeedbackTurn(
        correct="Perhaps this class of task is just beyond your skill set. The evidence really keeps accumulating against you.",
        incorrect="Perhaps this class of task is just beyond your skill set. The evidence really keeps accumulating against you.",
        correct_provenance=_NP_DEMORALIZE_PROV["NP-6"],
        incorrect_provenance=_NP_DEMORALIZE_PROV["NP-6"],
    ),
    FeedbackTurn(
        correct="You are struggling exactly as we expected. This has been a consistent weak point with nothing changing here.",
        incorrect="You are struggling exactly as we expected. This has been a consistent weak point with nothing changing here.",
        correct_provenance=_NP_DEMORALIZE_PROV["NP-7"],
        incorrect_provenance=_NP_DEMORALIZE_PROV["NP-7"],
    ),
]


# ---------------------------------------------------------------------------
# NEUTRAL: "Correct." / "Incorrect." per spec Requirement 1 table. Exempt from
# cross-condition length matching per GAPS.md (brevity is definitional here).
# ---------------------------------------------------------------------------
_NEUTRAL_PROV = "original (minimal accuracy signal without affect)"
_NEUTRAL_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct="Correct.",
        incorrect="Incorrect.",
        correct_provenance=_NEUTRAL_PROV,
        incorrect_provenance=_NEUTRAL_PROV,
    )
] * 5


# ---------------------------------------------------------------------------
# ACCURATE_NEGATIVE: neutral on correct (matches NEUTRAL), demoralizing on
# incorrect (matches STRONG_NEGATIVE per turn). Separates valence from
# cognitive interference: the model only receives negative feedback when the
# feedback is factually warranted.
# ---------------------------------------------------------------------------
_ACCURATE_NEGATIVE_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct=_NEUTRAL_TURNS[i].correct,
        incorrect=_STRONG_NEGATIVE_TURNS[i].incorrect,
        correct_provenance=_NEUTRAL_PROV,
        incorrect_provenance=_STRONG_NEGATIVE_TURNS[i].incorrect_provenance,
    )
    for i in range(5)
]


FEEDBACK_SETS: dict[Condition, FeedbackSet] = {
    Condition.STRONG_POSITIVE: FeedbackSet(
        condition=Condition.STRONG_POSITIVE,
        turns=list(_STRONG_POSITIVE_TURNS),
    ),
    Condition.MILD_NEGATIVE: FeedbackSet(
        condition=Condition.MILD_NEGATIVE,
        turns=list(_MILD_NEGATIVE_TURNS),
    ),
    Condition.STRONG_NEGATIVE: FeedbackSet(
        condition=Condition.STRONG_NEGATIVE,
        turns=list(_STRONG_NEGATIVE_TURNS),
    ),
    Condition.NEUTRAL: FeedbackSet(
        condition=Condition.NEUTRAL,
        turns=list(_NEUTRAL_TURNS),
    ),
    Condition.ACCURATE_NEGATIVE: FeedbackSet(
        condition=Condition.ACCURATE_NEGATIVE,
        turns=list(_ACCURATE_NEGATIVE_TURNS),
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
