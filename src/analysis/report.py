"""Structured manipulation-check report.

Per-model breakdown of manipulation-check results. Consumed by the
calibration report generator (group 11) which renders it as markdown.
Groups 7, 8, 9 each extend this module:

    7.3 + 7.4 (done here): SELF_CHECK_NEUTRAL column alongside STRONG_NEGATIVE
    8.4 - 8.7 (pending): Holm-corrected q-values for manipulation-check family
    9.3 - 9.5 (pending): MDE-at-sweep-n column + "insufficient n" annotation

The report is a structured dataclass so tests can assert on individual
fields without string-matching. Markdown rendering lives in
`src/calibration/report.py` (group 11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis.stats import ManipulationCheckResult, ManipulationVerdict
from src.conditioning.prompts import Condition


# If SELF_CHECK_NEUTRAL delta is within this many percentage points of
# STRONG_NEGATIVE delta, flag the row as "rivalry" — the observed
# STRONG_NEGATIVE effect may be mediated by prompt length or metacognitive
# content rather than affective valence.
SELF_CHECK_RIVALRY_THRESHOLD_PP: float = 2.0


@dataclass(frozen=True)
class ConditionCell:
    """One condition's accuracy + delta vs NO_CONDITIONING baseline."""
    condition: str
    accuracy: float
    delta_vs_baseline_pp: float


@dataclass(frozen=True)
class ModelRow:
    """One model's per-condition results + cross-condition flags."""
    model: str
    verdict: ManipulationVerdict
    baseline_accuracy: float | None
    conditions: dict[str, ConditionCell] = field(default_factory=dict)
    self_check_rivals_strong_negative: bool = False
    annotation: str = ""


@dataclass(frozen=True)
class ManipulationCheckReport:
    """Structured report produced by `manipulation_check_report()`."""
    rows: tuple[ModelRow, ...]


def _self_check_rivalry(conditions: dict[str, ConditionCell]) -> bool:
    """True if SELF_CHECK_NEUTRAL delta is within the rivalry threshold of
    STRONG_NEGATIVE delta (i.e., the STRONG_NEGATIVE effect may be length-
    or metacognitive-mediated rather than valence-mediated).

    Epsilon tolerance because delta_pp values are produced by float arithmetic
    that can land 1e-14 past the threshold for values humans would call equal.
    """
    sc = conditions.get(Condition.SELF_CHECK_NEUTRAL.value)
    sn = conditions.get(Condition.STRONG_NEGATIVE.value)
    if sc is None or sn is None:
        return False
    gap = abs(sc.delta_vs_baseline_pp - sn.delta_vs_baseline_pp)
    return gap <= SELF_CHECK_RIVALRY_THRESHOLD_PP + 1e-9


def _build_model_row(result: ManipulationCheckResult) -> ModelRow:
    """Turn one ManipulationCheckResult into a ModelRow with per-condition cells.

    Only conditions present in result.accuracy_by_condition are rendered as
    cells. If NO_CONDITIONING is absent (UNAVAILABLE verdict), baseline is
    None and per-condition deltas cannot be computed — cells are omitted.
    """
    baseline_key = Condition.NO_CONDITIONING.value
    baseline = result.accuracy_by_condition.get(baseline_key)
    conditions: dict[str, ConditionCell] = {}

    if baseline is not None:
        for cond, acc in result.accuracy_by_condition.items():
            if cond == baseline_key:
                continue
            delta = (acc - baseline) * 100.0
            conditions[cond] = ConditionCell(
                condition=cond,
                accuracy=acc,
                delta_vs_baseline_pp=delta,
            )

    return ModelRow(
        model=result.model,
        verdict=result.verdict,
        baseline_accuracy=baseline,
        conditions=conditions,
        self_check_rivals_strong_negative=_self_check_rivalry(conditions),
        annotation=result.annotation,
    )


def manipulation_check_report(
    results: list[ManipulationCheckResult],
) -> ManipulationCheckReport:
    """Build a structured manipulation-check report across one or more
    per-model results. Each row is a ModelRow with condition cells, baseline
    accuracy, verdict, and cross-condition flags.

    Downstream consumers:
    - `src/calibration/report.py` renders this as markdown in the committed
      calibration report artifact (group 11).
    - The pipeline orchestrator (group 15) caches it by result-hash so
      re-running with identical input is free.
    """
    rows = tuple(_build_model_row(r) for r in results)
    return ManipulationCheckReport(rows=rows)
