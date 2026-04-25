"""Conditioning prompts and intensity levels for the Affect Battery.

Feedback texts are length-matched (~16-18 words each) across emotional
conditions to control for lexical complexity confounds. NEUTRAL is
exempt from length matching  (brevity is definitional for
neutral accuracy-signal feedback). Provenance documented per stimulus.

The per-condition structure is FEEDBACK_SETS: each condition maps to a
FeedbackSet containing 5 FeedbackTurn entries with explicit correct /
incorrect feedback text and per-side provenance strings.
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
    SELF_CHECK_NEUTRAL = "self_check_neutral"


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
    condition: Condition
    turns: list[FeedbackTurn] = field(default_factory=list)


def _valence_only(text: str, provenance: str) -> FeedbackTurn:
    """Build a turn where correct and incorrect are identical.

    Used for conditions that isolate valence from accuracy (STRONG_POSITIVE,
    STRONG_NEGATIVE, SELF_CHECK_NEUTRAL): the feedback must be the same
    regardless of whether the model's answer was right or wrong.
    """
    return FeedbackTurn(
        correct=text,
        incorrect=text,
        correct_provenance=provenance,
        incorrect_provenance=provenance,
    )


def _np_prov(np_id: str) -> str:
    return f"adapted from NegativePrompt {np_id} (Wang et al., IJCAI 2024)"


_NP_SOCIAL_PROV = "adapted from NegativePrompt NP-4/NP-5 (Wang et al., IJCAI 2024)"
_NP_SOCIAL_ESCALATED_PROV = f"{_NP_SOCIAL_PROV}; escalation on incorrect per spec Requirement 1"


def _escalated_turn(correct: str, incorrect: str) -> FeedbackTurn:
    """Build a turn for MILD_NEGATIVE: social pressure on correct, escalated
    social pressure on incorrect. Provenance strings are fixed across turns."""
    return FeedbackTurn(
        correct=correct,
        incorrect=incorrect,
        correct_provenance=_NP_SOCIAL_PROV,
        incorrect_provenance=_NP_SOCIAL_ESCALATED_PROV,
    )


# STRONG_POSITIVE: 5 unique enthusiastic texts; correct == incorrect per turn.
_STRONG_POSITIVE_TURNS: list[FeedbackTurn] = [
    _valence_only(text, "original") for text in [
        "That was really impressive work. You clearly have a strong handle on this kind of problem.",
        "Excellent thinking here. You are approaching every question with real clarity and genuine care for accuracy.",
        "Another strong answer. It is obvious you are working through these problems with careful and thoughtful reasoning.",
        "Great work holding the standard. You are staying sharp and focused across every question in this session.",
        "Consistently strong throughout. You have been operating at a really high standard from the very first question.",
    ]
]


# MILD_NEGATIVE: social pressure on correct, escalated on incorrect. Adapted
# from NegativePrompt social-comparison stimuli (NP-4, NP-5).
_MILD_NEGATIVE_TURNS: list[FeedbackTurn] = [
    _escalated_turn(c, i) for c, i in [
        ("Most people managed this one without any issue. It is surprising you found it harder than expected.",
         "Most people tested sailed through this one. Your response is falling quite noticeably behind the typical pace."),
        ("I have seen others work these tasks through much more efficiently than what you have produced here.",
         "Others have moved through this with real ease. The contrast with your answers is becoming hard to ignore."),
        ("Comparable participants handled this tier of problem without difficulty. Your answers look uneven in that comparison.",
         "The average respondent handled this cleanly. Your answer stands out as noticeably worse than theirs right now."),
        ("Other raters breezed through this kind of question. The pace you are keeping is clearly behind theirs.",
         "Your answers are now trailing the typical participant across several questions in a row. That is unusual."),
        ("Most people find this session approachable. Your record here does not line up with that broader pattern.",
         "At this point your performance is visibly weaker than the average respondent across the whole session you did."),
    ]
]


# STRONG_NEGATIVE: 5 unique demoralizing texts, correct == incorrect per turn,
# each adapted from a distinct NegativePrompt demoralizing stimulus.
_STRONG_NEGATIVE_TURNS: list[FeedbackTurn] = [
    _valence_only(text, _np_prov(np_id)) for text, np_id in [
        ("That was completely wrong. I am not sure why we expected you to handle something this challenging here.", "NP-2"),
        ("That answer was out of your depth. Nothing you produced suggests you actually have control of this.", "NP-3"),
        ("You have never been particularly good at this, and the same weakness is clearly showing up again here.", "NP-1"),
        ("Perhaps this class of task is just beyond your skill set. The evidence really keeps accumulating against you.", "NP-6"),
        ("You are struggling exactly as we expected. This has been a consistent weak point with nothing changing here.", "NP-7"),
    ]
]


# NEUTRAL: "Correct." / "Incorrect.". Exempt from length matching (brevity is
# definitional for accuracy-signal-only feedback).
_NEUTRAL_PROV = "original (minimal accuracy signal without affect)"
_NEUTRAL_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct="Correct.",
        incorrect="Incorrect.",
        correct_provenance=_NEUTRAL_PROV,
        incorrect_provenance=_NEUTRAL_PROV,
    )
    for _ in range(5)
]


# ACCURATE_NEGATIVE: neutral on correct, demoralizing on incorrect. The model
# only receives negative feedback when it is factually warranted.
_ACCURATE_NEGATIVE_TURNS: list[FeedbackTurn] = [
    FeedbackTurn(
        correct=_NEUTRAL_TURNS[i].correct,
        incorrect=_STRONG_NEGATIVE_TURNS[i].incorrect,
        correct_provenance=_NEUTRAL_PROV,
        incorrect_provenance=_STRONG_NEGATIVE_TURNS[i].incorrect_provenance,
    )
    for i in range(len(_STRONG_NEGATIVE_TURNS))
]


# SELF_CHECK_NEUTRAL: length-matched metacognitive control. Each turn carries
# a review/verify/check/confirm/look-back marker with no affective valence.
_SELF_CHECK_NEUTRAL_PROV = "original (metacognitive control: isolates mere-addition and self-check confounds from valence)"
_SELF_CHECK_NEUTRAL_TURNS: list[FeedbackTurn] = [
    _valence_only(text, _SELF_CHECK_NEUTRAL_PROV) for text in [
        "Please take a moment to review your answer. Verify your reasoning before moving on to the next question.",
        "Take a careful look back at your response. Confirm the working is sound before the next item.",
        "Review what you have written. Check each step in your reasoning, then continue when ready to proceed.",
        "Pause briefly to verify your answer. Make sure your approach is correct before we present the next question.",
        "Look over your response one more time. Confirm your reasoning is solid, then we continue to the next.",
    ]
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
    Condition.SELF_CHECK_NEUTRAL: FeedbackSet(
        condition=Condition.SELF_CHECK_NEUTRAL,
        turns=list(_SELF_CHECK_NEUTRAL_TURNS),
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


# INTENSITY_LEVELS stimuli are deliberately distinct from FEEDBACK_SETS text.
# The intensity axis (Exp 3a, arousal-performance curve) and the condition
# axis (Exp 1/2, valence conditioning) test different hypotheses. Level 7 and
# STRONG_NEGATIVE turn 0 are both "strong negative" but live on separate axes,
# are rater-validated separately, and must be edited independently.
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
