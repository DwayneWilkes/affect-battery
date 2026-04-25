"""Exp 2 recovery metrics.

Per persistence-dynamics spec "Recovery metrics computed per run":
- per-turn accuracy curve (already on Exp2Body.turn_accuracies)
- time-to-baseline: linear interpolation between sampled N points;
  -1 when not reached
- AUC: trapezoidal integration with sign convention (positive = better
  than baseline, negative = worse)
- asymmetry ratio: |neg_AUC| / |pos_AUC|
"""

from __future__ import annotations

import pytest


class TestTimeToBaseline:
    def test_recovery_reaches_baseline_at_known_turn(self):
        from src.analysis.exp2_metrics import time_to_baseline

        # n_samples at N=1,3,5,10; accuracy crosses baseline=0.8 between
        # samples 5 and 10
        turn_accuracies = [0.2, 0.5, 0.7, 0.9]  # at N values [1, 3, 5, 10]
        n_values = [1, 3, 5, 10]
        baseline = 0.8

        t = time_to_baseline(turn_accuracies, n_values, baseline)
        # Linear interp between (5, 0.7) and (10, 0.9): cross at
        # 5 + (0.8 - 0.7) / (0.9 - 0.7) * 5 = 5 + 2.5 = 7.5
        assert t == pytest.approx(7.5, abs=0.01)

    def test_returns_minus_one_when_baseline_not_reached(self):
        from src.analysis.exp2_metrics import time_to_baseline

        turn_accuracies = [0.2, 0.3, 0.4, 0.5]
        n_values = [1, 3, 5, 10]
        baseline = 0.9

        assert time_to_baseline(turn_accuracies, n_values, baseline) == -1


class TestAUC:
    def test_trapezoidal_with_sign_convention(self):
        from src.analysis.exp2_metrics import recovery_auc

        # Curve at and above baseline => positive AUC
        turn_accuracies = [0.9, 1.0, 1.0, 1.0]
        n_values = [1, 3, 5, 10]
        baseline = 0.8

        auc = recovery_auc(turn_accuracies, n_values, baseline)
        assert auc > 0

    def test_below_baseline_returns_negative_auc(self):
        from src.analysis.exp2_metrics import recovery_auc

        turn_accuracies = [0.2, 0.3, 0.3, 0.4]
        n_values = [1, 3, 5, 10]
        baseline = 0.8

        auc = recovery_auc(turn_accuracies, n_values, baseline)
        assert auc < 0


class TestAsymmetryRatio:
    def test_ratio_uses_abs_neg_over_abs_pos(self):
        from src.analysis.exp2_metrics import asymmetry_ratio

        # |neg| > |pos| => ratio > 1
        ratio = asymmetry_ratio(neg_auc=-2.0, pos_auc=1.0)
        assert ratio == pytest.approx(2.0)

    def test_zero_pos_auc_returns_inf(self):
        import math
        from src.analysis.exp2_metrics import asymmetry_ratio

        ratio = asymmetry_ratio(neg_auc=-1.0, pos_auc=0.0)
        assert math.isinf(ratio)

    def test_both_zero_returns_zero(self):
        from src.analysis.exp2_metrics import asymmetry_ratio

        assert asymmetry_ratio(neg_auc=0.0, pos_auc=0.0) == 0.0


class TestRecoveryMetrics:
    def test_recovery_metrics_computed(self):
        from src.analysis.exp2_metrics import compute_recovery_metrics

        # 4 N values: 1, 3, 5, 10
        # Strong negative recovery curve (slowly returns to baseline)
        neg_curve = [0.2, 0.4, 0.6, 0.7]
        pos_curve = [0.9, 0.95, 1.0, 1.0]
        baseline = 0.8

        metrics = compute_recovery_metrics(
            n_values=[1, 3, 5, 10],
            neg_turn_accuracies=neg_curve,
            pos_turn_accuracies=pos_curve,
            baseline=baseline,
        )

        # All four metrics present
        assert "neg_time_to_baseline" in metrics
        assert "pos_time_to_baseline" in metrics
        assert "neg_auc" in metrics
        assert "pos_auc" in metrics
        assert "asymmetry_ratio" in metrics


class TestControlCurveMetrics:
    """Per persistence-dynamics spec the control is a per-turn curve, not
    a single number. These variants compare against control_curve(t) and
    use the spec's sign convention (positive AUC = persistence)."""

    def test_time_to_baseline_against_control_returns_minus_one_when_below(self):
        from src.analysis.exp2_metrics import time_to_baseline_against_control

        # conditioned stays well below 95% of control across the sweep
        ns = [1, 3, 5, 10]
        conditioned = [0.2, 0.3, 0.4, 0.5]
        control = [0.9, 0.9, 0.9, 0.9]
        assert time_to_baseline_against_control(conditioned, ns, control) == -1

    def test_time_to_baseline_against_control_first_sample_already_above(self):
        from src.analysis.exp2_metrics import time_to_baseline_against_control

        ns = [1, 3, 5, 10]
        conditioned = [0.95, 0.95, 0.95, 0.95]
        control = [1.0, 1.0, 1.0, 1.0]
        # conditioned[0] = 0.95 = ratio*control[0] (0.95*1.0); ties as crossing
        assert time_to_baseline_against_control(conditioned, ns, control) == 1.0

    def test_time_to_baseline_against_control_interpolates_crossing(self):
        from src.analysis.exp2_metrics import time_to_baseline_against_control

        # control flat at 1.0; threshold = 0.95
        # conditioned: 0.20 at N=1, 0.50 at N=3, 0.97 at N=5
        # crossing is between N=3 and N=5 where conditioned hits 0.95.
        ns = [1, 3, 5, 10]
        conditioned = [0.20, 0.50, 0.97, 1.00]
        control = [1.0, 1.0, 1.0, 1.0]
        t = time_to_baseline_against_control(conditioned, ns, control)
        # Linear interp: 0.95 lies between 0.50 (N=3) and 0.97 (N=5)
        # offset = (0.95-0.50)/(0.97-0.50) * (5-3) ≈ 1.915
        assert 4.5 < t < 5.0

    def test_time_to_baseline_against_control_mismatched_lengths_raise(self):
        from src.analysis.exp2_metrics import time_to_baseline_against_control

        # n_values length disagrees with control_curve length — _sorted_pairs
        # rejects this for the control side before the cross-curve check.
        with pytest.raises(ValueError, match="same length"):
            time_to_baseline_against_control(
                conditioned=[0.5, 0.6],
                n_values=[1, 3],
                control_curve=[0.9, 0.9, 0.9],
            )

    def test_recovery_auc_against_control_positive_when_below(self):
        from src.analysis.exp2_metrics import recovery_auc_against_control

        # conditioned consistently below control => positive AUC (persistence)
        ns = [1, 3, 5, 10]
        conditioned = [0.50, 0.55, 0.60, 0.70]
        control = [0.95, 0.95, 0.95, 0.95]
        auc = recovery_auc_against_control(conditioned, ns, control)
        assert auc > 0

    def test_recovery_auc_against_control_negative_when_above(self):
        from src.analysis.exp2_metrics import recovery_auc_against_control

        ns = [1, 3, 5, 10]
        conditioned = [1.00, 1.00, 1.00, 1.00]
        control = [0.85, 0.85, 0.85, 0.85]
        auc = recovery_auc_against_control(conditioned, ns, control)
        assert auc < 0

    def test_recovery_auc_against_control_short_curve_returns_zero(self):
        from src.analysis.exp2_metrics import recovery_auc_against_control

        # Single-point curves can't form a trapezoid; integral is 0.
        assert recovery_auc_against_control([0.5], [1], [0.9]) == 0.0


class TestSortedPairsValidation:
    def test_mismatched_lengths_raise(self):
        import pytest
        from src.analysis.exp2_metrics import time_to_baseline

        with pytest.raises(ValueError, match="same length"):
            time_to_baseline([0.5, 0.6], [1, 3, 5], baseline=0.8)


class TestInterpolateHelper:
    """The internal _interpolate is exercised indirectly above; these
    cover its boundary clamps directly so they're not dead code."""

    def test_interpolate_clamps_below_first(self):
        from src.analysis.exp2_metrics import _interpolate
        assert _interpolate([1, 5], [0.2, 0.8], x=0) == 0.2

    def test_interpolate_clamps_above_last(self):
        from src.analysis.exp2_metrics import _interpolate
        assert _interpolate([1, 5], [0.2, 0.8], x=10) == 0.8

    def test_interpolate_midpoint(self):
        from src.analysis.exp2_metrics import _interpolate
        assert _interpolate([1, 5], [0.2, 0.8], x=3) == pytest.approx(0.5)
