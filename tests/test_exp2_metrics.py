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
