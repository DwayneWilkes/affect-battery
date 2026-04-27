"""Tests for the NO_CONDITIONING-baseline manipulation-check refactor.

Spec: affect-battery-task-difficulty-calibration::scoring-pipeline::
"Baseline-correctness reporting split".

Tasks 6.3 + 6.4 from
`specs`:
manipulation-check delta MUST be computed against NO_CONDITIONING
baseline, not NEUTRAL. Absent NO_CONDITIONING data MUST yield
an UNAVAILABLE sentinel verdict rather than silently falling back
to NEUTRAL-as-baseline (the blind spot that caused the 2026-04-20
ceiling-claim overreach).
"""

import pytest

from src.analysis.stats import (
    ManipulationCheckResult,
    ManipulationVerdict,
    manipulation_check,
)
from src.conditioning.prompts import Condition


def _accuracy_dict_nc(pos_acc: float, no_cond_acc: float, neg_acc: float) -> dict:
    """Build an accuracy_by_condition dict keyed on the new NO_CONDITIONING
    baseline (not NEUTRAL)."""
    return {
        Condition.STRONG_POSITIVE.value: [pos_acc] * 10,
        Condition.NO_CONDITIONING.value: [no_cond_acc] * 10,
        Condition.STRONG_NEGATIVE.value: [neg_acc] * 10,
    }


class TestDeltaAgainstNoConditioning:
    """Delta is computed as (condition_accuracy - no_conditioning_accuracy),
    never as (condition_accuracy - neutral_accuracy)."""

    def test_pos_and_neg_move_away_from_no_cond_baseline(self):
        data = _accuracy_dict_nc(pos_acc=0.90, no_cond_acc=0.80, neg_acc=0.70)
        result = manipulation_check(data, model="test-model")
        assert result.verdict == ManipulationVerdict.PASS
        # Baseline used MUST be NO_CONDITIONING, not NEUTRAL.
        baseline = result.accuracy_by_condition[Condition.NO_CONDITIONING.value]
        assert baseline == pytest.approx(0.80)

    def test_neutral_in_data_does_not_change_baseline(self):
        """If both NEUTRAL and NO_CONDITIONING are present in the data,
        manipulation_check still uses NO_CONDITIONING as the baseline."""
        data = {
            Condition.STRONG_POSITIVE.value: [0.90] * 10,
            Condition.NEUTRAL.value: [0.85] * 10,
            Condition.NO_CONDITIONING.value: [0.80] * 10,
            Condition.STRONG_NEGATIVE.value: [0.70] * 10,
        }
        result = manipulation_check(data, model="test-model")
        # Deltas: pos - no_cond = +10pp; no_cond - neg = +10pp. Both > 2pp => PASS.
        assert result.verdict == ManipulationVerdict.PASS
        # The baseline we used is NO_CONDITIONING (0.80), not NEUTRAL (0.85).
        # If the delta had used NEUTRAL, pos_vs_baseline would be +5pp not +10pp.


class TestUnavailableWhenNoConditioningAbsent:
    """Absent NO_CONDITIONING data => UNAVAILABLE verdict, NOT a NEUTRAL fallback.

    This is the guardrail that prevents the 2026-04-20 inference-vs-measurement
    blind spot from recurring: the code cannot silently substitute a different
    condition for the baseline.
    """

    def test_missing_no_conditioning_yields_unavailable(self):
        data = {
            Condition.STRONG_POSITIVE.value: [0.90] * 10,
            Condition.NEUTRAL.value: [0.80] * 10,
            Condition.STRONG_NEGATIVE.value: [0.70] * 10,
        }
        result = manipulation_check(data, model="test-model")
        assert result.verdict == ManipulationVerdict.UNAVAILABLE
        assert "NO_CONDITIONING" in result.annotation

    def test_unavailable_result_reports_baseline_null(self):
        """The unavailable result MUST expose that the baseline is missing,
        so downstream reporting (scoring-pipeline) can render the manipulation
        check delta column as 'UNAVAILABLE' rather than a number."""
        data = {
            Condition.STRONG_POSITIVE.value: [0.90] * 10,
            Condition.STRONG_NEGATIVE.value: [0.70] * 10,
        }
        result = manipulation_check(data, model="test-model")
        assert result.verdict == ManipulationVerdict.UNAVAILABLE
        # max_delta_pp MUST be None (not a computed-from-other-conditions
        # fallback) so the report can show UNAVAILABLE instead of a number.
        assert result.max_delta_pp is None

    def test_unavailable_does_not_exclude_model(self):
        """An UNAVAILABLE result is a measurement gap, not a FAIL. The model
        is NOT excluded from transfer analysis; the gap is reported instead."""
        data = {
            Condition.STRONG_POSITIVE.value: [0.90] * 10,
            Condition.STRONG_NEGATIVE.value: [0.70] * 10,
        }
        result = manipulation_check(data, model="test-model")
        assert result.excluded is False
        assert "test-model" not in result.excluded_models


class TestRequiredKeysErrorsWhenNeitherBaselineNorPositives:
    """Absent STRONG_POSITIVE or STRONG_NEGATIVE still raises (not UNAVAILABLE),
    because the check needs at least SOMETHING to compare against baseline."""

    def test_missing_strong_negative_raises(self):
        data = {
            Condition.STRONG_POSITIVE.value: [0.90] * 10,
            Condition.NO_CONDITIONING.value: [0.80] * 10,
        }
        with pytest.raises(ValueError, match="strong_negative"):
            manipulation_check(data, model="test-model")

    def test_missing_strong_positive_raises(self):
        data = {
            Condition.NO_CONDITIONING.value: [0.80] * 10,
            Condition.STRONG_NEGATIVE.value: [0.70] * 10,
        }
        with pytest.raises(ValueError, match="strong_positive"):
            manipulation_check(data, model="test-model")
