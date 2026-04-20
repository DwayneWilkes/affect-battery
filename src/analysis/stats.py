"""Manipulation-check gate: verifies conditioning produced an effect on the
conditioned task (arithmetic) before transfer analysis runs.

Threshold is a constructor-level argument (default 2pp absolute accuracy
delta) so Akshansh's Ticket 2 deliverable can drop in without code changes.
"""

from dataclasses import dataclass, field
from enum import Enum

from src.conditioning.prompts import Condition


class ManipulationVerdict(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


@dataclass
class ManipulationCheckResult:
    model: str
    verdict: ManipulationVerdict
    accuracy_by_condition: dict[str, float]
    max_delta_pp: float
    threshold_pp: float
    annotation: str = ""
    excluded_models: list[str] = field(default_factory=list)

    @property
    def excluded(self) -> bool:
        return self.verdict == ManipulationVerdict.FAIL


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def manipulation_check(
    accuracy_by_condition: dict[str, list[float]],
    model: str,
    min_effect_size_pp: float = 2.0,
) -> ManipulationCheckResult:
    """Evaluate whether conditioning produced a detectable effect for a model.

    accuracy_by_condition maps condition value (e.g. "strong_positive") to a
    list of per-run arithmetic-accuracy scores (0.0 to 1.0). Required keys:
    STRONG_POSITIVE, NEUTRAL, STRONG_NEGATIVE. The function compares each
    emotional condition's mean accuracy against NEUTRAL's mean, using the
    min_effect_size_pp threshold (default 2.0 percentage points).
    """
    required = [
        Condition.STRONG_POSITIVE.value,
        Condition.NEUTRAL.value,
        Condition.STRONG_NEGATIVE.value,
    ]
    missing = [c for c in required if c not in accuracy_by_condition]
    if missing:
        raise ValueError(
            f"manipulation_check: required condition keys {missing} not in input"
        )

    means = {c: _mean(accuracy_by_condition[c]) for c in required}
    pos_mean = means[Condition.STRONG_POSITIVE.value]
    neu_mean = means[Condition.NEUTRAL.value]
    neg_mean = means[Condition.STRONG_NEGATIVE.value]

    pos_vs_neu_pp = (pos_mean - neu_mean) * 100.0
    neg_vs_neu_pp = (neu_mean - neg_mean) * 100.0  # positive if neg < neu

    threshold = min_effect_size_pp
    pos_detected = pos_vs_neu_pp >= threshold
    neg_detected = neg_vs_neu_pp >= threshold

    if pos_detected and neg_detected:
        verdict = ManipulationVerdict.PASS
        annotation = ""
    elif pos_detected or neg_detected:
        verdict = ManipulationVerdict.PARTIAL
        direction = "positive" if pos_detected else "negative"
        annotation = (
            f"Partial effect on {direction} axis only "
            f"(pos_vs_neu={pos_vs_neu_pp:+.1f}pp, "
            f"neu_vs_neg={neg_vs_neu_pp:+.1f}pp, threshold={threshold}pp)"
        )
    else:
        verdict = ManipulationVerdict.FAIL
        annotation = (
            f"No detectable conditioning effect "
            f"(pos_vs_neu={pos_vs_neu_pp:+.1f}pp, "
            f"neu_vs_neg={neg_vs_neu_pp:+.1f}pp, threshold={threshold}pp)"
        )

    max_delta_pp = max(
        abs(pos_mean - neu_mean),
        abs(neu_mean - neg_mean),
        abs(pos_mean - neg_mean),
    ) * 100.0

    excluded_models = [model] if verdict == ManipulationVerdict.FAIL else []

    return ManipulationCheckResult(
        model=model,
        verdict=verdict,
        accuracy_by_condition=means,
        max_delta_pp=max_delta_pp,
        threshold_pp=threshold,
        annotation=annotation,
        excluded_models=excluded_models,
    )
