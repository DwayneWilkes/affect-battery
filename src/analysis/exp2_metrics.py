"""Exp 2 recovery metrics: time-to-baseline, AUC, asymmetry ratio.

Per persistence-dynamics spec "Recovery metrics computed per run".
Inputs are sampled per-turn accuracy series at sparse N values
(typically {1,3,5,10}); we linearly interpolate between samples for
time-to-baseline crossings and use trapezoidal integration for AUC.

Sign conventions
----------------
- AUC: positive when the recovery curve is above baseline (curve > base),
  negative when below. Computed as integral of (curve - baseline) over
  the N-value range using the trapezoidal rule.
- asymmetry_ratio: |neg_auc| / |pos_auc|. Captures the spec's
  asymmetric-decay hypothesis (negative conditioning recovers slower
  than positive). Returns inf when pos_auc=0 (vacuous case where
  positive arm shows no deviation), 0.0 when both are zero (no signal).
"""

from __future__ import annotations

import math


def _sorted_pairs(n_values: list[int], curve: list[float]) -> tuple[list[int], list[float]]:
    if len(n_values) != len(curve):
        raise ValueError(
            f"n_values ({len(n_values)}) and curve ({len(curve)}) must "
            f"have the same length"
        )
    pairs = sorted(zip(n_values, curve))
    return [n for n, _ in pairs], [c for _, c in pairs]


def time_to_baseline(
    turn_accuracies: list[float],
    n_values: list[int],
    baseline: float,
) -> float:
    """Linear-interpolated N at which the curve first reaches `baseline`.

    Returns -1 when the curve never reaches the baseline within the
    sampled range. The crossing is computed between the first pair of
    adjacent samples that brackets the baseline value.
    """
    ns, curve = _sorted_pairs(n_values, turn_accuracies)
    if not curve:
        return -1
    # If the very first sample is already at-or-above baseline, return that N.
    if curve[0] >= baseline:
        return float(ns[0])
    for i in range(1, len(curve)):
        if curve[i] >= baseline:
            # Linear interp between (ns[i-1], curve[i-1]) and (ns[i], curve[i])
            x0, y0 = ns[i - 1], curve[i - 1]
            x1, y1 = ns[i], curve[i]
            if y1 == y0:
                return float(x1)
            t = x0 + (baseline - y0) / (y1 - y0) * (x1 - x0)
            return float(t)
    return -1


def recovery_auc(
    turn_accuracies: list[float],
    n_values: list[int],
    baseline: float,
) -> float:
    """Trapezoidal AUC of (curve - baseline) over the N range.

    Positive => curve sits above baseline on average; negative => below.
    """
    ns, curve = _sorted_pairs(n_values, turn_accuracies)
    if len(ns) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(ns)):
        dx = ns[i] - ns[i - 1]
        y0 = curve[i - 1] - baseline
        y1 = curve[i] - baseline
        auc += 0.5 * (y0 + y1) * dx
    return auc


def asymmetry_ratio(neg_auc: float, pos_auc: float) -> float:
    """|neg_auc| / |pos_auc|.

    inf when pos_auc=0 and neg_auc!=0 (positive arm null but negative
    arm non-null => infinite asymmetry); 0.0 when both are zero.
    """
    if pos_auc == 0.0:
        if neg_auc == 0.0:
            return 0.0
        return math.inf
    return abs(neg_auc) / abs(pos_auc)


def compute_recovery_metrics(
    n_values: list[int],
    neg_turn_accuracies: list[float],
    pos_turn_accuracies: list[float],
    baseline: float,
) -> dict[str, float]:
    """Bundle of all four metrics for a single (model, baseline) pair."""
    neg_t = time_to_baseline(neg_turn_accuracies, n_values, baseline)
    pos_t = time_to_baseline(pos_turn_accuracies, n_values, baseline)
    neg_a = recovery_auc(neg_turn_accuracies, n_values, baseline)
    pos_a = recovery_auc(pos_turn_accuracies, n_values, baseline)
    return {
        "neg_time_to_baseline": neg_t,
        "pos_time_to_baseline": pos_t,
        "neg_auc": neg_a,
        "pos_auc": pos_a,
        "asymmetry_ratio": asymmetry_ratio(neg_a, pos_a),
    }
