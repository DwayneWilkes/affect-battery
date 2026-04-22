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
from src.analysis_corrections import apply_bh_correction, apply_holm_correction
from src.conditioning.prompts import Condition


# Annotation that accompanies the base-vs-instruct primary contrast in the
# report, per design D8: this contrast is pre-registered as a single primary
# test and is reported UNCORRECTED. Reviewers need to see this explicitly
# so the lack of correction is a visible design choice, not an omission.
BASE_VS_INSTRUCT_ANNOTATION: str = (
    "pre-registered primary contrast, uncorrected within family "
    "(alpha=0.05 unadjusted)"
)


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

    `mde_at_sweep_n`: MDE for this cell's baseline_acc at the intended sweep
    n, populated when target_sweep_n is supplied to manipulation_check_report.
    `raw_p` / `holm_q`: manipulation-check family p-value and Holm-corrected
    q-value for the condition-vs-baseline test, populated when p_values_by_model
    is supplied.
    """
    condition: str
    accuracy: float
    delta_vs_baseline_pp: float
    mde_at_sweep_n: float | None = None
    raw_p: float | None = None
    holm_q: float | None = None


@dataclass(frozen=True)
class ModelRow:
    """One model's per-condition results + cross-condition flags.

    `pairwise_bh_q`: BH-corrected q-values for pairwise condition contrasts
    within this model, populated when pairwise_p_values_by_model is supplied.
    Keys are (condition_a, condition_b) tuples preserving the input order.
    """
    model: str
    verdict: ManipulationVerdict
    baseline_accuracy: float | None
    conditions: dict[str, ConditionCell] = field(default_factory=dict)
    self_check_rivals_strong_negative: bool = False
    mde_insufficient: bool = False
    pairwise_bh_q: dict[tuple[str, str], float] | None = None
    annotation: str = ""


@dataclass(frozen=True)
class ManipulationCheckReport:
    """Structured report produced by `manipulation_check_report()`.

    `base_vs_instruct_p_values`: uncorrected p-values for the pre-registered
    single-primary base-vs-instruct contrast, populated when the caller
    supplies base_vs_instruct_p_values. Keyed by (base_model, instruct_model).
    `base_vs_instruct_annotation`: explanation of why this contrast is
    reported uncorrected; empty when no base-vs-instruct contrast is present.
    """
    rows: tuple[ModelRow, ...]
    base_vs_instruct_p_values: dict[tuple[str, str], float] | None = None
    base_vs_instruct_annotation: str = ""


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


def _holm_q_by_condition(
    model_pvals: dict[str, float] | None,
) -> dict[str, float]:
    """Apply Holm-Bonferroni within one model's manipulation-check family
    and return q-values keyed by the same condition keys as the input."""
    if not model_pvals:
        return {}
    conditions = list(model_pvals.keys())
    raw = [model_pvals[c] for c in conditions]
    qs = apply_holm_correction(raw)
    return {c: q for c, q in zip(conditions, qs)}


def _bh_q_by_pair(
    model_pairwise: dict[tuple[str, str], float] | None,
) -> dict[tuple[str, str], float] | None:
    """Apply Benjamini-Hochberg within one model's pairwise-contrast family
    and return q-values keyed by the same (condA, condB) tuples as the input."""
    if not model_pairwise:
        return None
    pairs = list(model_pairwise.keys())
    raw = [model_pairwise[p] for p in pairs]
    qs = apply_bh_correction(raw)
    return {p: q for p, q in zip(pairs, qs)}


def _build_model_row(
    result: ManipulationCheckResult,
    target_sweep_n: int | None,
    model_pvals: dict[str, float] | None,
    model_pairwise: dict[tuple[str, str], float] | None,
) -> ModelRow:
    """Turn one ManipulationCheckResult into a ModelRow with per-condition
    cells. Applies MDE, Holm, and BH wiring when the relevant inputs are
    supplied; otherwise leaves those fields at None.

    Only conditions present in result.accuracy_by_condition are rendered as
    cells. If NO_CONDITIONING is absent (UNAVAILABLE verdict), baseline is
    None and per-condition deltas cannot be computed — cells are omitted.
    """
    baseline_key = Condition.NO_CONDITIONING.value
    baseline = result.accuracy_by_condition.get(baseline_key)
    conditions: dict[str, ConditionCell] = {}

    holm_q = _holm_q_by_condition(model_pvals)

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
                raw_p=(model_pvals or {}).get(cond),
                holm_q=holm_q.get(cond),
            )

    return ModelRow(
        model=result.model,
        verdict=result.verdict,
        baseline_accuracy=baseline,
        conditions=conditions,
        self_check_rivals_strong_negative=_self_check_rivalry(conditions),
        mde_insufficient=_mde_insufficient(conditions, target_sweep_n),
        pairwise_bh_q=_bh_q_by_pair(model_pairwise),
        annotation=result.annotation,
    )


def manipulation_check_report(
    results: list[ManipulationCheckResult],
    target_sweep_n: int | None = None,
    p_values_by_model: dict[str, dict[str, float]] | None = None,
    pairwise_p_values_by_model: dict[str, dict[tuple[str, str], float]] | None = None,
    base_vs_instruct_p_values: dict[tuple[str, str], float] | None = None,
) -> ManipulationCheckReport:
    """Build a structured manipulation-check report across one or more
    per-model results. Each row is a ModelRow with condition cells,
    baseline accuracy, verdict, and cross-condition flags.

    Optional inputs extend the report:
    - target_sweep_n: enables MDE per cell + insufficient-n row flag.
    - p_values_by_model: {model: {condition: p}} for the manipulation-check
      family. Each cell gets raw_p + Holm-corrected holm_q (Holm applied
      within the model's family).
    - pairwise_p_values_by_model: {model: {(condA, condB): p}} for pairwise
      contrasts. Each row gets pairwise_bh_q (BH applied within the model's
      pairwise family).
    - base_vs_instruct_p_values: {(base_model, instruct_model): p}. Reported
      at the top level, UNCORRECTED, with an explicit pre-registered-primary
      annotation per design D8.

    Downstream consumers:
    - `src/calibration/report.py` renders this as markdown in the committed
      calibration report artifact (group 11).
    - The pipeline orchestrator (group 15) caches it by result-hash so
      re-running with identical input is free.
    """
    rows = tuple(
        _build_model_row(
            r,
            target_sweep_n,
            (p_values_by_model or {}).get(r.model),
            (pairwise_p_values_by_model or {}).get(r.model),
        )
        for r in results
    )
    base_vs_instruct_annotation = (
        BASE_VS_INSTRUCT_ANNOTATION if base_vs_instruct_p_values else ""
    )
    return ManipulationCheckReport(
        rows=rows,
        base_vs_instruct_p_values=base_vs_instruct_p_values,
        base_vs_instruct_annotation=base_vs_instruct_annotation,
    )
