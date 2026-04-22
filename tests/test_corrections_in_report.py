"""Tests for Holm + BH correction wiring in manipulation_check_report.

Spec: affect-battery-task-difficulty-calibration::scoring-pipeline::
"Multiple-comparisons correction policy".

Tasks 8.4-8.7: when `p_values_by_model`, `pairwise_p_values_by_model`,
or `base_vs_instruct_p_values` are supplied, the report MUST emit both
raw p-values and corrected q-values under the policy from design D8:

    - Manipulation-check family: Holm-Bonferroni within each model
    - Pairwise condition contrasts: Benjamini-Hochberg within each model
    - Base-vs-instruct primary: uncorrected with explicit annotation
"""

import pytest

from src.analysis.report import (
    ConditionCell,
    ManipulationCheckReport,
    ModelRow,
    manipulation_check_report,
)
from src.analysis.stats import manipulation_check
from src.conditioning.prompts import Condition


def _build_mc_result(pos_acc, baseline_acc, neg_acc, model="m"):
    data = {
        Condition.STRONG_POSITIVE.value: [pos_acc] * 10,
        Condition.NO_CONDITIONING.value: [baseline_acc] * 10,
        Condition.STRONG_NEGATIVE.value: [neg_acc] * 10,
    }
    return manipulation_check(data, model=model)


class TestHolmOnManipulationCheckFamily:
    def test_cell_exposes_raw_p_and_holm_q_when_pvals_provided(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70, model="m")
        p_values = {
            "m": {
                Condition.STRONG_POSITIVE.value: 0.01,
                Condition.STRONG_NEGATIVE.value: 0.04,
            }
        }
        report = manipulation_check_report([r], p_values_by_model=p_values)
        row = report.rows[0]
        pos = row.conditions[Condition.STRONG_POSITIVE.value]
        neg = row.conditions[Condition.STRONG_NEGATIVE.value]
        assert pos.raw_p == pytest.approx(0.01)
        assert neg.raw_p == pytest.approx(0.04)
        # Holm on [0.01, 0.04] sorted: smallest q = 0.01*2 = 0.02, largest q = 0.04*1 = 0.04.
        assert pos.holm_q == pytest.approx(0.02, abs=1e-6)
        assert neg.holm_q == pytest.approx(0.04, abs=1e-6)

    def test_cell_has_none_when_pvals_not_provided(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70, model="m")
        report = manipulation_check_report([r])
        cell = report.rows[0].conditions[Condition.STRONG_NEGATIVE.value]
        assert cell.raw_p is None
        assert cell.holm_q is None

    def test_holm_is_per_model_family(self):
        """Holm corrects within each model's family, not across models."""
        r1 = _build_mc_result(pos_acc=0.9, baseline_acc=0.8, neg_acc=0.7, model="m1")
        r2 = _build_mc_result(pos_acc=0.9, baseline_acc=0.8, neg_acc=0.7, model="m2")
        p_values = {
            "m1": {Condition.STRONG_POSITIVE.value: 0.01, Condition.STRONG_NEGATIVE.value: 0.04},
            "m2": {Condition.STRONG_POSITIVE.value: 0.02, Condition.STRONG_NEGATIVE.value: 0.03},
        }
        report = manipulation_check_report([r1, r2], p_values_by_model=p_values)
        # Each row should have Holm applied only within its own model's p-values.
        m1_row = next(r for r in report.rows if r.model == "m1")
        m2_row = next(r for r in report.rows if r.model == "m2")
        # m1: [0.01, 0.04] -> [0.02, 0.04]
        assert m1_row.conditions[Condition.STRONG_POSITIVE.value].holm_q == pytest.approx(0.02)
        # m2: [0.02, 0.03] -> [0.04, 0.04]
        assert m2_row.conditions[Condition.STRONG_POSITIVE.value].holm_q == pytest.approx(0.04)


class TestBhOnPairwise:
    def test_pairwise_bh_populates_row_field(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70, model="m")
        pos = Condition.STRONG_POSITIVE.value
        neg = Condition.STRONG_NEGATIVE.value
        baseline = Condition.NO_CONDITIONING.value
        pairwise_pvals = {
            "m": {
                (pos, neg): 0.001,
                (pos, baseline): 0.01,
                (baseline, neg): 0.02,
            }
        }
        report = manipulation_check_report(
            [r], pairwise_p_values_by_model=pairwise_pvals,
        )
        row = report.rows[0]
        assert row.pairwise_bh_q is not None
        assert (pos, neg) in row.pairwise_bh_q
        assert (pos, baseline) in row.pairwise_bh_q
        # BH q-values should be >= raw p-values.
        for pair, raw_p in pairwise_pvals["m"].items():
            assert row.pairwise_bh_q[pair] >= raw_p - 1e-9

    def test_pairwise_bh_is_none_when_not_provided(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70, model="m")
        report = manipulation_check_report([r])
        assert report.rows[0].pairwise_bh_q is None


class TestBaseVsInstructPrimary:
    def test_base_vs_instruct_p_uncorrected_with_annotation(self):
        r_base = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.65, model="base")
        r_instruct = _build_mc_result(pos_acc=0.80, baseline_acc=0.80, neg_acc=0.80, model="instruct")
        report = manipulation_check_report(
            [r_base, r_instruct],
            base_vs_instruct_p_values={("base", "instruct"): 0.03},
        )
        assert report.base_vs_instruct_p_values is not None
        assert report.base_vs_instruct_p_values[("base", "instruct")] == pytest.approx(0.03)
        # Explicit uncorrected annotation MUST appear so reviewers cannot miss it.
        assert "pre-registered" in report.base_vs_instruct_annotation.lower()
        assert "uncorrected" in report.base_vs_instruct_annotation.lower()

    def test_base_vs_instruct_is_none_by_default(self):
        r = _build_mc_result(pos_acc=0.90, baseline_acc=0.80, neg_acc=0.70, model="m")
        report = manipulation_check_report([r])
        assert report.base_vs_instruct_p_values is None
        assert report.base_vs_instruct_annotation == ""
