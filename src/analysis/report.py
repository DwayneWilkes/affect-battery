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

from src.analysis.mde import compute_mde
from src.analysis.stats import ManipulationCheckResult, ManipulationVerdict
from src.conditioning.prompts import Condition


# If SELF_CHECK_NEUTRAL delta is within this many percentage points of
# STRONG_NEGATIVE delta, flag the row as "rivalry" — the observed
# STRONG_NEGATIVE effect may be mediated by prompt length or metacognitive
# content rather than affective valence.
SELF_CHECK_RIVALRY_THRESHOLD_PP: float = 2.0

# If the MDE at target_sweep_n is more than this multiple of the largest
# observed |delta| across conditions, flag the row as "insufficient n":
# the observed pilot effect is unlikely to replicate reliably at the
# intended sweep n.
MDE_INSUFFICIENCY_MULTIPLIER: float = 2.0


@dataclass(frozen=True)
class ConditionCell:
    """One condition's accuracy + delta vs NO_CONDITIONING baseline.

    `mde_at_sweep_n` is populated only when the caller supplies
    `target_sweep_n` to `manipulation_check_report`. It is the MDE
    (fraction 0.0-1.0) for this cell's baseline_acc at the intended
    sweep n, α=0.05, power=0.80.
    """
    condition: str
    accuracy: float
    delta_vs_baseline_pp: float
    mde_at_sweep_n: float | None = None


@dataclass(frozen=True)
class ModelRow:
    """One model's per-condition results + cross-condition flags."""
    model: str
    verdict: ManipulationVerdict
    baseline_accuracy: float | None
    conditions: dict[str, ConditionCell] = field(default_factory=dict)
    self_check_rivals_strong_negative: bool = False
    mde_insufficient: bool = False
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


def _mde_insufficient(
    conditions: dict[str, ConditionCell],
    target_sweep_n: int | None,
) -> bool:
    """True when MDE at target_sweep_n exceeds the insufficiency multiplier
    times the largest observed |delta| across conditions. Requires both a
    target_sweep_n and at least one cell with a populated mde_at_sweep_n."""
    if target_sweep_n is None or not conditions:
        return False
    deltas = [abs(c.delta_vs_baseline_pp) / 100.0 for c in conditions.values()]
    mdes = [c.mde_at_sweep_n for c in conditions.values() if c.mde_at_sweep_n is not None]
    if not deltas or not mdes:
        return False
    max_delta = max(deltas)
    if max_delta == 0.0:
        # No observed effect at all; the MDE question is moot, don't flag.
        return False
    max_mde = max(mdes)
    return max_mde > MDE_INSUFFICIENCY_MULTIPLIER * max_delta


def _build_model_row(
    result: ManipulationCheckResult,
    target_sweep_n: int | None,
) -> ModelRow:
    """Turn one ManipulationCheckResult into a ModelRow with per-condition cells.

    Only conditions present in result.accuracy_by_condition are rendered as
    cells. If NO_CONDITIONING is absent (UNAVAILABLE verdict), baseline is
    None and per-condition deltas cannot be computed — cells are omitted.

    When target_sweep_n is supplied, each cell carries mde_at_sweep_n (MDE
    computed from the baseline accuracy at the intended sweep n).
    """
    baseline_key = Condition.NO_CONDITIONING.value
    baseline = result.accuracy_by_condition.get(baseline_key)
    conditions: dict[str, ConditionCell] = {}

    if baseline is not None:
        mde_value: float | None = (
            compute_mde(baseline_acc=baseline, n=target_sweep_n)
            if target_sweep_n is not None
            else None
        )
        for cond, acc in result.accuracy_by_condition.items():
            if cond == baseline_key:
                continue
            delta = (acc - baseline) * 100.0
            conditions[cond] = ConditionCell(
                condition=cond,
                accuracy=acc,
                delta_vs_baseline_pp=delta,
                mde_at_sweep_n=mde_value,
            )

    return ModelRow(
        model=result.model,
        verdict=result.verdict,
        baseline_accuracy=baseline,
        conditions=conditions,
        self_check_rivals_strong_negative=_self_check_rivalry(conditions),
        mde_insufficient=_mde_insufficient(conditions, target_sweep_n),
        annotation=result.annotation,
    )


def manipulation_check_report(
    results: list[ManipulationCheckResult],
    target_sweep_n: int | None = None,
) -> ManipulationCheckReport:
    """Build a structured manipulation-check report across one or more
    per-model results. Each row is a ModelRow with condition cells, baseline
    accuracy, verdict, and cross-condition flags.

    When `target_sweep_n` is supplied, each cell carries mde_at_sweep_n and
    each row's mde_insufficient flag surfaces cases where the observed pilot
    effect is unlikely to replicate reliably at the intended sweep n.

    Downstream consumers:
    - `src/calibration/report.py` renders this as markdown in the committed
      calibration report artifact (group 11).
    - The pipeline orchestrator (group 15) caches it by result-hash so
      re-running with identical input is free.
    """
    rows = tuple(_build_model_row(r, target_sweep_n) for r in results)
    return ManipulationCheckReport(rows=rows)
