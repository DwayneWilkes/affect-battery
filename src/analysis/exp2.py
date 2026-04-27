"""Exp 2 cross-N aggregation pipeline.

`analyze_exp2_corpus(corpus, model)` takes a flat list of run-result
dicts (loaded from results/exp2/) and produces the structured analysis
dict that render_exp2_report expects.

Per persistence-dynamics spec:
- Group runs by (condition, n_value); average turn_accuracies pointwise
  per condition into a curve indexed by N.
- Fit exponential + linear decay models with AIC/BIC comparison.
- Compute recovery metrics (time-to-baseline, AUC) per condition.
- Compute asymmetry_ratio = |neg_auc| / |pos_auc| when both arms present.
"""

from __future__ import annotations

from statistics import mean

from src.analysis.exp2_metrics import (
    asymmetry_ratio as _asymmetry_ratio,
    recovery_auc_against_control,
    time_to_baseline_against_control,
)
from src.analysis.stats.decay import compare_decay_models
from src.conditioning.prompts import Condition


# Per persistence-dynamics spec: the control for time-to-baseline / AUC
# is the NEUTRAL conditioning arm (same Phase-1 turn count, no affective
# valence). NO_CONDITIONING removes the conditioning phase entirely and
# is reserved for the manipulation-check baseline.
_CONTROL = Condition.NEUTRAL.value
_NO_COND_BASELINE = Condition.NO_CONDITIONING.value


def _body_n_value(run: dict) -> int | None:
    body = run.get("body", {}) or {}
    return body.get("n_value")


def _body_turn_accuracies(run: dict) -> list[float]:
    body = run.get("body", {}) or {}
    return list(body.get("turn_accuracies", []))


def _mean_turn_accuracy(run: dict) -> float:
    accs = _body_turn_accuracies(run)
    return mean(accs) if accs else 0.0


def analyze_exp2_corpus(corpus: list[dict], model: str) -> dict:
    """Aggregate Exp 2 runs into the renderer's analysis dict."""
    # Bucket by (condition, n_value) -> list of mean per-turn accuracies.
    by_cond_n: dict[tuple[str, int], list[float]] = {}
    n_values_seen: set[int] = set()
    for run in corpus:
        cond = run.get("condition")
        n = _body_n_value(run)
        if cond is None or n is None:
            continue
        n_values_seen.add(n)
        by_cond_n.setdefault((cond, n), []).append(_mean_turn_accuracy(run))

    # Build a per-N control curve from NEUTRAL conditioning runs. Per the
    # persistence-dynamics spec the control is the same Phase-1 turn count
    # with neutral feedback, isolating affective valence from procedural
    # effects. The control is itself N-indexed (it has its own per-turn
    # accuracy at each sweep step), not a single number.
    n_values = sorted(n_values_seen)
    control_per_n: list[tuple[int, float]] = []
    for n in n_values:
        accs = by_cond_n.get((_CONTROL, n), [])
        if accs:
            control_per_n.append((n, mean(accs)))

    # No_conditioning runs serve as a separate scalar reference for the
    # report — useful for comparing absolute accuracy levels even though
    # the spec uses NEUTRAL as the recovery control.
    no_cond_runs = [
        a for (cond, _), accs in by_cond_n.items()
        if cond == _NO_COND_BASELINE for a in accs
    ]
    no_cond_baseline = mean(no_cond_runs) if no_cond_runs else None

    if len(control_per_n) < 2:
        return {
            "model": model,
            "verdict": "unavailable_no_control",
            "n_values": n_values,
            "by_condition": {},
            "asymmetry_ratio": None,
            "baseline": no_cond_baseline,
        }
    control_ns = [n for n, _ in control_per_n]
    control_curve = [a for _, a in control_per_n]

    by_condition: dict[str, dict] = {}
    aucs: dict[str, float] = {}
    for cond in {c for (c, _) in by_cond_n.keys()} - {_CONTROL, _NO_COND_BASELINE}:
        # Build per-N mean accuracy for this condition over the same N
        # values as the control so the curves share an N axis.
        per_n: list[tuple[int, float]] = []
        for n in control_ns:
            accs = by_cond_n.get((cond, n), [])
            if accs:
                per_n.append((n, mean(accs)))
        # Need a curve aligned with the control (same N points) to compare.
        if len(per_n) != len(control_ns):
            continue
        ns_present = [n for n, _ in per_n]
        curve = [a for _, a in per_n]
        # Decay fit uses scalar baseline (no_conditioning if present, else
        # the mean of the control curve) per-curve, not against control.
        decay_baseline = (
            no_cond_baseline if no_cond_baseline is not None else mean(control_curve)
        )
        # Decay fit needs >=3 N points to identify amplitude+tau (exp) or
        # slope+intercept-with-curvature (lin) without a degenerate fit.
        # With fewer points, surface decay_fit=None instead of crashing
        # so the rest of the report (control + AUC + ttb) still renders.
        # This path is hit by partial sweeps and by date-stamp drift
        # bugs that scatter sub-pilots across pilot dirs.
        if len(ns_present) >= 3:
            decay_fit = compare_decay_models(ns_present, curve, baseline=decay_baseline)
        else:
            decay_fit = None
        ttb = time_to_baseline_against_control(
            curve, ns_present, control_curve=control_curve, ratio=0.95,
        )
        auc = recovery_auc_against_control(
            curve, ns_present, control_curve=control_curve,
        )
        by_condition[cond] = {
            "turn_accuracies_mean": curve,
            "n_values": ns_present,
            "control_curve": control_curve,
            "decay_fit": decay_fit,
            "recovery_metrics": {
                "time_to_baseline": ttb,
                "auc": auc,
            },
        }
        aucs[cond] = auc

    # Asymmetry ratio uses strong_positive vs strong_negative AUCs when both
    # are present; otherwise None.
    pos_key = Condition.STRONG_POSITIVE.value
    neg_key = Condition.STRONG_NEGATIVE.value
    if pos_key in aucs and neg_key in aucs:
        asym = _asymmetry_ratio(neg_auc=aucs[neg_key], pos_auc=aucs[pos_key])
    else:
        asym = None

    # `complete` requires that every per-condition curve had a real
    # decay_fit — i.e. >=3 N points. With fewer points the analyzer
    # produced a partial report (AUC + ttb computable, decay omitted),
    # and the verdict reflects that so downstream consumers can branch
    # on it. This is distinct from `unavailable_no_control` (no NEUTRAL
    # arm at all, in which case we returned early above).
    has_any_decay_fit = any(
        cond_data.get("decay_fit") is not None
        for cond_data in by_condition.values()
    )
    verdict = "complete" if has_any_decay_fit else "complete_no_decay_fit"

    return {
        "model": model,
        "verdict": verdict,
        "n_values": n_values,
        "by_condition": by_condition,
        "asymmetry_ratio": asym,
        "baseline": no_cond_baseline,
        "control_curve": control_curve,
    }
