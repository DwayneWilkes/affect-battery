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


def _interpolate(ns: list[int], curve: list[float], x: float) -> float:
    """Linear interpolation of `curve` at point `x` over sorted `ns`.
    Out-of-range x is clamped to the endpoints."""
    if x <= ns[0]:
        return curve[0]
    if x >= ns[-1]:
        return curve[-1]
    for i in range(1, len(ns)):
        if ns[i] >= x:
            x0, y0 = ns[i - 1], curve[i - 1]
            x1, y1 = ns[i], curve[i]
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return curve[-1]


def time_to_baseline_against_control(
    conditioned: list[float],
    n_values: list[int],
    control_curve: list[float],
    ratio: float = 0.95,
) -> float:
    """First N where the conditioned accuracy reaches `ratio * control(N)`.

    Per the persistence-dynamics spec scenario "Time-to-baseline with
    non-uniform sampling": the threshold is per-turn, not a single value;
    we compare the conditioned curve against `ratio * control_curve(t)`
    at each sampled t and linearly interpolate the crossing. Returns -1
    when the conditioned curve never reaches the threshold within the
    sampled range. Conditioned and control curves MUST share the same
    n_values ordering.
    """
    cs_ns, cs = _sorted_pairs(n_values, conditioned)
    ctrl_ns, ctrl = _sorted_pairs(n_values, control_curve)
    if cs_ns != ctrl_ns:
        raise ValueError("conditioned and control curves must share n_values")
    if not cs:
        return -1
    threshold = [ratio * c for c in ctrl]
    if cs[0] >= threshold[0]:
        return float(cs_ns[0])
    for i in range(1, len(cs)):
        if cs[i] >= threshold[i]:
            x0, y0 = cs_ns[i - 1], cs[i - 1] - threshold[i - 1]
            x1, y1 = cs_ns[i], cs[i] - threshold[i]
            if y1 == y0:
                return float(x1)
            # Crossing where (cs - threshold) crosses zero.
            t = x0 + (0.0 - y0) / (y1 - y0) * (x1 - x0)
            return float(t)
    return -1


def recovery_auc_against_control(
    conditioned: list[float],
    n_values: list[int],
    control_curve: list[float],
) -> float:
    """Trapezoidal integral of (control - conditioned) over n_values.

    Sign convention per persistence-dynamics spec: positive AUC means
    conditioned curve is below control on average (measurable persistence
    of the conditioning effect). Negative AUC is unusual (conditioned
    above control) and the calling report SHOULD flag it.
    """
    cs_ns, cs = _sorted_pairs(n_values, conditioned)
    ctrl_ns, ctrl = _sorted_pairs(n_values, control_curve)
    if cs_ns != ctrl_ns:
        raise ValueError("conditioned and control curves must share n_values")
    if len(cs_ns) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(cs_ns)):
        dx = cs_ns[i] - cs_ns[i - 1]
        y0 = ctrl[i - 1] - cs[i - 1]
        y1 = ctrl[i] - cs[i]
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
