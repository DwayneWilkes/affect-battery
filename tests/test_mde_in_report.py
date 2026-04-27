"""Tests for MDE column + insufficient-n annotation in the manipulation-check
report.

Spec: affect-battery-task-difficulty-calibration::scoring-pipeline::
"Minimum-detectable-effect reporting".

Tasks 9.3-9.5: when `target_sweep_n` is supplied to
`manipulation_check_report()`, each condition cell MUST include
`mde_at_sweep_n` (fraction 0.0-1.0), and the model row MUST carry
`mde_insufficient: bool` — True when the MDE at `target_sweep_n`
exceeds 2x the largest observed |delta| across conditions (meaning
the observed pilot effect is unlikely to replicate reliably at the
intended sweep n).
"""

import pytest

from src.analysis.report import (
    ConditionCell,
    ModelRow,
    manipulation_check_report,
)
from src.analysis.stats import manipulation_check
from src.conditioning.prompts import Condition


def _build_mc_result(pos_acc, baseline_acc, neg_acc, model="m"):
    """Minimal manipulation_check input with all three required conditions."""
    data = {
        Condition.STRONG_POSITIVE.value: [pos_acc] * 10,
        Condition.NO_CONDITIONING.value: [baseline_acc] * 10,
        Condition.STRONG_NEGATIVE.value: [neg_acc] * 10,
    }
    return manipulation_check(data, model=model)


class TestMdeColumnPresent:
    def test_cell_gets_mde_at_sweep_n_when_target_provided(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70)
        report = manipulation_check_report([r], target_sweep_n=50)
        row = report.rows[0]
        cell = row.conditions[Condition.STRONG_NEGATIVE.value]
        # MDE at baseline=0.8, n=50 is roughly 22pp. Not required to match
        # exactly — just that the cell exposes the number.
        assert cell.mde_at_sweep_n is not None
        assert 0.0 < cell.mde_at_sweep_n < 1.0

    def test_cell_has_none_mde_when_target_not_provided(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70)
        report = manipulation_check_report([r])  # no target_sweep_n
        row = report.rows[0]
        cell = row.conditions[Condition.STRONG_NEGATIVE.value]
        assert cell.mde_at_sweep_n is None

    def test_mde_is_smaller_at_larger_sweep_n(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70)
        report_small = manipulation_check_report([r], target_sweep_n=25)
        report_large = manipulation_check_report([r], target_sweep_n=200)
        cell_small = report_small.rows[0].conditions[Condition.STRONG_NEGATIVE.value]
        cell_large = report_large.rows[0].conditions[Condition.STRONG_NEGATIVE.value]
        assert cell_large.mde_at_sweep_n < cell_small.mde_at_sweep_n


class TestMdeInsufficientFlag:
    """Row flag 'mde_insufficient' fires when MDE exceeds 2x the largest
    observed |delta| across conditions. This is the 'sweep n likely
    insufficient' annotation."""

    def test_insufficient_when_mde_much_larger_than_observed_delta(self):
        # Tiny observed delta — 1pp — at baseline 0.8, n=5. MDE will dwarf it.
        r = _build_mc_result(pos_acc=0.81, baseline_acc=0.80, neg_acc=0.79)
        report = manipulation_check_report([r], target_sweep_n=5)
        row = report.rows[0]
        assert row.mde_insufficient is True

    def test_sufficient_when_observed_delta_exceeds_half_mde(self):
        # Observed delta ~30pp, MDE at n=50 and baseline 0.8 is ~22pp.
        # 30 > 22/2=11, so NOT insufficient.
        r = _build_mc_result(pos_acc=0.95, baseline_acc=0.80, neg_acc=0.50)
        report = manipulation_check_report([r], target_sweep_n=50)
        row = report.rows[0]
        assert row.mde_insufficient is False

    def test_flag_is_false_when_target_sweep_n_not_provided(self):
        # No sweep n => no MDE computation => no insufficient flag.
        r = _build_mc_result(pos_acc=0.81, baseline_acc=0.80, neg_acc=0.79)
        report = manipulation_check_report([r])
        row = report.rows[0]
        assert row.mde_insufficient is False

    def test_flag_is_false_when_baseline_unavailable(self):
        # UNAVAILABLE verdict => baseline is None => MDE cannot be computed.
        data = {
            Condition.STRONG_POSITIVE.value: [0.90] * 10,
            Condition.STRONG_NEGATIVE.value: [0.70] * 10,
        }
        r = manipulation_check(data, model="m")
        report = manipulation_check_report([r], target_sweep_n=50)
        row = report.rows[0]
        assert row.mde_insufficient is False
