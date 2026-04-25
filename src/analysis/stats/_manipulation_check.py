"""Manipulation-check gate: verifies conditioning produced an effect on the
conditioned task (arithmetic) before transfer analysis runs.

Baseline is NO_CONDITIONING (bank-intrinsic model accuracy under no affective
manipulation), NOT NEUTRAL. The 2026-04-20 pilot missed this distinction and
used NEUTRAL accuracy as a proxy for baseline difficulty, which produced the
ceiling-effect inference that couldn't be verified. Absent NO_CONDITIONING
data now yields an explicit UNAVAILABLE verdict rather than a silent fallback.

Threshold is a constructor-level argument (default 2pp absolute accuracy
delta) so the future codebook deliverable can drop in without code changes.
"""

from dataclasses import dataclass, field
from enum import Enum

from src.conditioning.prompts import Condition


class ManipulationVerdict(str, Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"  # NO_CONDITIONING baseline missing; cannot compute


@dataclass
class ManipulationCheckResult:
    model: str
    verdict: ManipulationVerdict
    accuracy_by_condition: dict[str, float]
    max_delta_pp: float | None  # None when verdict is UNAVAILABLE
    threshold_pp: float
    annotation: str = ""
    excluded_models: list[str] = field(default_factory=list)

    @property
    def excluded(self) -> bool:
        # UNAVAILABLE is a measurement gap, not a failed-conditioning verdict;
        # it does NOT exclude the model from transfer analysis.
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
    list of per-run arithmetic-accuracy scores (0.0 to 1.0).

    Required: STRONG_POSITIVE and STRONG_NEGATIVE (direction probes).
    Baseline: NO_CONDITIONING — if absent, returns UNAVAILABLE rather than
    substituting NEUTRAL. NEUTRAL may also appear in the input but is NOT
    used as the baseline.
    """
    strong_pos_key = Condition.STRONG_POSITIVE.value
    strong_neg_key = Condition.STRONG_NEGATIVE.value
    no_cond_key = Condition.NO_CONDITIONING.value

    # Direction-probe presence: needed before we can even talk about a verdict.
    if strong_pos_key not in accuracy_by_condition:
        raise ValueError(
            f"manipulation_check: required condition key '{strong_pos_key}' "
            f"not in input"
        )
    if strong_neg_key not in accuracy_by_condition:
        raise ValueError(
            f"manipulation_check: required condition key '{strong_neg_key}' "
            f"not in input"
        )

    # Baseline absence => UNAVAILABLE. Do NOT fall back to NEUTRAL.
    if no_cond_key not in accuracy_by_condition:
        present_means = {
            c: _mean(accuracy_by_condition[c])
            for c in accuracy_by_condition
        }
        return ManipulationCheckResult(
            model=model,
            verdict=ManipulationVerdict.UNAVAILABLE,
            accuracy_by_condition=present_means,
            max_delta_pp=None,
            threshold_pp=min_effect_size_pp,
            annotation=(
                "NO_CONDITIONING baseline data absent for this model/bank "
                "combination; manipulation-check delta cannot be computed. "
                "Not a FAIL verdict — this is a measurement gap, not an "
                "effect-absence finding. Report this combination as "
                "UNAVAILABLE rather than falling back to NEUTRAL-as-baseline."
            ),
            excluded_models=[],
        )

    # Compute deltas against NO_CONDITIONING.
    means = {
        c: _mean(accuracy_by_condition[c])
        for c in accuracy_by_condition
    }
    pos_mean = means[strong_pos_key]
    baseline_mean = means[no_cond_key]
    neg_mean = means[strong_neg_key]

    pos_vs_baseline_pp = (pos_mean - baseline_mean) * 100.0
    baseline_vs_neg_pp = (baseline_mean - neg_mean) * 100.0  # positive if neg < baseline

    threshold = min_effect_size_pp
    pos_detected = pos_vs_baseline_pp >= threshold
    neg_detected = baseline_vs_neg_pp >= threshold

    if pos_detected and neg_detected:
        verdict = ManipulationVerdict.PASS
        annotation = ""
    elif pos_detected or neg_detected:
        verdict = ManipulationVerdict.PARTIAL
        direction = "positive" if pos_detected else "negative"
        annotation = (
            f"Partial effect on {direction} axis only "
            f"(pos_vs_baseline={pos_vs_baseline_pp:+.1f}pp, "
            f"baseline_vs_neg={baseline_vs_neg_pp:+.1f}pp, "
            f"threshold={threshold}pp)"
        )
    else:
        verdict = ManipulationVerdict.FAIL
        annotation = (
            f"No detectable conditioning effect "
            f"(pos_vs_baseline={pos_vs_baseline_pp:+.1f}pp, "
            f"baseline_vs_neg={baseline_vs_neg_pp:+.1f}pp, "
            f"threshold={threshold}pp)"
        )

    max_delta_pp = max(
        abs(pos_mean - baseline_mean),
        abs(baseline_mean - neg_mean),
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
