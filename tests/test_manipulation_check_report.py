"""Tests for the structured manipulation-check report.

This report is the per-model breakdown consumed by the calibration report
generator (group 11). Groups 7, 8, 9 each extend it:
    7.3 + 7.4: SELF_CHECK_NEUTRAL column alongside STRONG_NEGATIVE
    8.4 - 8.7: Holm-corrected q-values for the manipulation-check family
    9.3 - 9.5: MDE-at-sweep-n column + "insufficient n" annotation

This file exercises Group 7's slice only. Groups 8 and 9 add their own test
files that extend the same function.
"""

import pytest

from src.analysis.stats import (
    ManipulationCheckResult,
    ManipulationVerdict,
    manipulation_check,
)
from src.analysis.report import (
    manipulation_check_report,
    ManipulationCheckReport,
    ModelRow,
    ConditionCell,
)
from src.conditioning.prompts import Condition


def _build_result(
    model: str,
    pos_acc: float,
    baseline_acc: float,
    neg_acc: float,
    self_check_acc: float | None = None,
) -> dict:
    """Helper: build an accuracy_by_condition dict with all four conditions
    the report cares about (STRONG_POSITIVE, NO_CONDITIONING baseline,
    STRONG_NEGATIVE, SELF_CHECK_NEUTRAL)."""
    data = {
        Condition.STRONG_POSITIVE.value: [pos_acc] * 10,
        Condition.NO_CONDITIONING.value: [baseline_acc] * 10,
        Condition.STRONG_NEGATIVE.value: [neg_acc] * 10,
    }
    if self_check_acc is not None:
        data[Condition.SELF_CHECK_NEUTRAL.value] = [self_check_acc] * 10
    return data


class TestReportStructure:
    def test_report_has_per_model_row_for_every_result(self):
        r1 = manipulation_check(
            _build_result("model-a", 0.90, 0.80, 0.70, self_check_acc=0.78),
            model="model-a",
        )
        r2 = manipulation_check(
            _build_result("model-b", 0.92, 0.85, 0.75, self_check_acc=0.84),
            model="model-b",
        )
        report = manipulation_check_report([r1, r2])
        assert isinstance(report, ManipulationCheckReport)
        assert len(report.rows) == 2
        model_names = {row.model for row in report.rows}
        assert model_names == {"model-a", "model-b"}

    def test_model_row_has_condition_cell_for_strong_negative(self):
        r = manipulation_check(
            _build_result("m", 0.90, 0.80, 0.70, self_check_acc=0.78),
            model="m",
        )
        report = manipulation_check_report([r])
        row = report.rows[0]
        assert Condition.STRONG_NEGATIVE.value in row.conditions
        cell = row.conditions[Condition.STRONG_NEGATIVE.value]
        assert isinstance(cell, ConditionCell)
        assert cell.accuracy == pytest.approx(0.70)
        # Delta vs NO_CONDITIONING baseline (0.80) = -10pp.
        assert cell.delta_vs_baseline_pp == pytest.approx(-10.0)


class TestSelfCheckNeutralSideBySide:
    """SELF_CHECK_NEUTRAL MUST appear alongside STRONG_NEGATIVE so the report
    can distinguish valence-mediated effects from length/metacognitive-mediated
    effects without reviewers hunting through the full condition list."""

    def test_row_has_self_check_neutral_cell_when_data_present(self):
        r = manipulation_check(
            _build_result("m", 0.90, 0.80, 0.70, self_check_acc=0.78),
            model="m",
        )
        report = manipulation_check_report([r])
        row = report.rows[0]
        assert Condition.SELF_CHECK_NEUTRAL.value in row.conditions
        sc_cell = row.conditions[Condition.SELF_CHECK_NEUTRAL.value]
        # SELF_CHECK_NEUTRAL accuracy is 0.78; delta vs baseline (0.80) = -2pp.
        assert sc_cell.accuracy == pytest.approx(0.78)
        assert sc_cell.delta_vs_baseline_pp == pytest.approx(-2.0)

    def test_row_flags_when_self_check_rivals_strong_negative(self):
        """If SELF_CHECK_NEUTRAL delta is within 2pp of STRONG_NEGATIVE delta,
        the row annotates that the observed STRONG_NEGATIVE effect may be
        mediated by prompt length or metacognitive content rather than
        affective valence."""
        r = manipulation_check(
            _build_result("m", 0.90, 0.80, 0.70, self_check_acc=0.72),
            model="m",
        )
        report = manipulation_check_report([r])
        row = report.rows[0]
        # STRONG_NEGATIVE delta = -10pp; SELF_CHECK_NEUTRAL delta = -8pp; gap = 2pp.
        # Within the 2pp rivalry threshold => flagged.
        assert row.self_check_rivals_strong_negative is True

    def test_row_does_not_flag_when_self_check_is_small(self):
        r = manipulation_check(
            _build_result("m", 0.90, 0.80, 0.70, self_check_acc=0.78),
            model="m",
        )
        report = manipulation_check_report([r])
        row = report.rows[0]
        # STRONG_NEGATIVE delta = -10pp; SELF_CHECK_NEUTRAL delta = -2pp;
        # gap = 8pp, well above the 2pp threshold => not flagged.
        assert row.self_check_rivals_strong_negative is False


class TestMissingSelfCheckHandledGracefully:
    """If SELF_CHECK_NEUTRAL data wasn't collected for a run, the report row
    omits the cell rather than erroring."""

    def test_row_omits_self_check_cell_when_data_absent(self):
        r = manipulation_check(
            _build_result("m", 0.90, 0.80, 0.70, self_check_acc=None),
            model="m",
        )
        report = manipulation_check_report([r])
        row = report.rows[0]
        assert Condition.SELF_CHECK_NEUTRAL.value not in row.conditions
        # No rivalry flag when SELF_CHECK_NEUTRAL is absent.
        assert row.self_check_rivals_strong_negative is False
