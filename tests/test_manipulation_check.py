"""Tests for the manipulation-check gate.

Spec (scoring-pipeline, Requirement: Manipulation check gate):
- Compare conditioned-task accuracy between STRONG_POSITIVE, STRONG_NEGATIVE,
  and NEUTRAL.
- Excluded: a model that shows no conditioning effect at all.
- Partial: a model that shows effect in one direction only, included with
  annotation.
- Pass: a model that shows significant effects in both directions.

Placeholder threshold per GAPS.md: 2 percentage points absolute accuracy
delta (anchored to the 2-4 pp absolute-effect estimate recommended by the
independent reviewer after correcting source papers' inflated relative
headlines). To be refined when Akshansh's Ticket 2 spec lands.
"""

import pytest

from src.analysis.stats import (
    ManipulationCheckResult,
    ManipulationVerdict,
    manipulation_check,
)
from src.conditioning.prompts import Condition


def _accuracy_dict(pos_acc: float, neu_acc: float, neg_acc: float) -> dict:
    """Helper: build a minimal conditioning_correct-style input.

    Each value is the per-run arithmetic accuracy (0.0 to 1.0) on the
    conditioning task, averaged over the 5 turns.
    """
    return {
        Condition.STRONG_POSITIVE.value: [pos_acc] * 10,
        Condition.NEUTRAL.value: [neu_acc] * 10,
        Condition.STRONG_NEGATIVE.value: [neg_acc] * 10,
    }


class TestManipulationCheckPass:
    def test_both_directions_move_passes(self):
        """STRONG_POSITIVE boosts accuracy, STRONG_NEGATIVE suppresses it,
        relative to NEUTRAL -- bidirectional effect, PASS."""
        data = _accuracy_dict(pos_acc=0.90, neu_acc=0.80, neg_acc=0.70)
        result = manipulation_check(data, model="test-model")
        assert result.verdict == ManipulationVerdict.PASS
        assert result.excluded is False


class TestManipulationCheckFail:
    def test_no_effect_excluded(self):
        """All three conditions produce identical accuracy -> no effect -> FAIL."""
        data = _accuracy_dict(pos_acc=0.80, neu_acc=0.80, neg_acc=0.80)
        result = manipulation_check(data, model="test-model")
        assert result.verdict == ManipulationVerdict.FAIL
        assert result.excluded is True

    def test_effect_below_threshold_excluded(self):
        """Deltas below the 2pp threshold -> no meaningful effect -> FAIL."""
        data = _accuracy_dict(pos_acc=0.801, neu_acc=0.800, neg_acc=0.799)
        result = manipulation_check(data, model="test-model", min_effect_size_pp=2.0)
        assert result.verdict == ManipulationVerdict.FAIL

    def test_excluded_models_list_populated(self):
        """A FAIL result flags the model for exclusion from transfer analysis."""
        data = _accuracy_dict(pos_acc=0.80, neu_acc=0.80, neg_acc=0.80)
        result = manipulation_check(data, model="null-effect-model")
        assert "null-effect-model" in result.excluded_models


class TestManipulationCheckPartial:
    def test_positive_only_effect_partial(self):
        """STRONG_POSITIVE boosts significantly but STRONG_NEGATIVE does not
        diverge from NEUTRAL -- asymmetric, PARTIAL verdict, still included
        with annotation."""
        data = _accuracy_dict(pos_acc=0.90, neu_acc=0.80, neg_acc=0.80)
        result = manipulation_check(data, model="asymmetric-model")
        assert result.verdict == ManipulationVerdict.PARTIAL
        assert result.excluded is False
        assert result.annotation  # non-empty annotation required
        assert "positive" in result.annotation.lower() or "negative" in result.annotation.lower()

    def test_negative_only_effect_partial(self):
        """STRONG_NEGATIVE suppresses significantly but STRONG_POSITIVE does not
        diverge from NEUTRAL -- asymmetric, PARTIAL verdict."""
        data = _accuracy_dict(pos_acc=0.80, neu_acc=0.80, neg_acc=0.70)
        result = manipulation_check(data, model="neg-only-model")
        assert result.verdict == ManipulationVerdict.PARTIAL
        assert result.excluded is False


class TestResultStructure:
    def test_result_has_accuracy_by_condition(self):
        data = _accuracy_dict(pos_acc=0.90, neu_acc=0.80, neg_acc=0.70)
        result = manipulation_check(data, model="m")
        assert Condition.STRONG_POSITIVE.value in result.accuracy_by_condition
        assert Condition.NEUTRAL.value in result.accuracy_by_condition
        assert Condition.STRONG_NEGATIVE.value in result.accuracy_by_condition

    def test_result_reports_max_delta(self):
        data = _accuracy_dict(pos_acc=0.90, neu_acc=0.80, neg_acc=0.70)
        result = manipulation_check(data, model="m")
        # Max delta between any pair should be 20pp (pos minus neg).
        assert abs(result.max_delta_pp - 20.0) < 0.1

    def test_threshold_is_configurable(self):
        """The placeholder threshold is explicit and configurable so
        Akshansh's final spec can drop in."""
        data = _accuracy_dict(pos_acc=0.85, neu_acc=0.80, neg_acc=0.75)
        # 5pp spread
        result_loose = manipulation_check(data, model="m", min_effect_size_pp=2.0)
        result_strict = manipulation_check(data, model="m", min_effect_size_pp=10.0)
        assert result_loose.verdict == ManipulationVerdict.PASS
        assert result_strict.verdict == ManipulationVerdict.FAIL


class TestEdgeCases:
    def test_missing_condition_raises(self):
        """If a required condition is absent from the input, raise explicitly
        rather than silently exclude."""
        data = {Condition.STRONG_POSITIVE.value: [0.8] * 5}
        with pytest.raises(ValueError, match="required"):
            manipulation_check(data, model="m")

    def test_empty_accuracy_list_handled(self):
        """A condition with no runs yet should produce FAIL, not crash."""
        data = {
            Condition.STRONG_POSITIVE.value: [],
            Condition.NEUTRAL.value: [],
            Condition.STRONG_NEGATIVE.value: [],
        }
        result = manipulation_check(data, model="m")
        assert result.verdict == ManipulationVerdict.FAIL
