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
    recovery_auc,
    time_to_baseline,
)
from src.analysis.stats.decay import compare_decay_models
from src.conditioning.prompts import Condition


_BASELINE = Condition.NO_CONDITIONING.value


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

    # Baseline: mean across all no_conditioning runs (any N) — baseline
    # accuracy is N-invariant by construction (no recovery dynamics in
    # the absence of conditioning).
    baseline_runs = [
        a for (cond, _), accs in by_cond_n.items()
        if cond == _BASELINE for a in accs
    ]
    if not baseline_runs:
        return {
            "model": model,
            "verdict": "unavailable_no_baseline",
            "n_values": sorted(n_values_seen),
            "by_condition": {},
            "asymmetry_ratio": None,
            "baseline": None,
        }
    baseline = mean(baseline_runs)

    n_values = sorted(n_values_seen)
    by_condition: dict[str, dict] = {}
    aucs: dict[str, float] = {}
    for cond in {c for (c, _) in by_cond_n.keys()} - {_BASELINE}:
        # Per-N mean accuracy for this condition. When a particular N has
        # no runs for this condition, skip it (the curve is shorter).
        per_n: list[tuple[int, float]] = []
        for n in n_values:
            accs = by_cond_n.get((cond, n), [])
            if accs:
                per_n.append((n, mean(accs)))
        if len(per_n) < 2:
            continue
        ns_present = [n for n, _ in per_n]
        curve = [a for _, a in per_n]
        decay_fit = compare_decay_models(ns_present, curve, baseline=baseline)
        ttb = time_to_baseline(curve, ns_present, baseline=baseline)
        auc = recovery_auc(curve, ns_present, baseline=baseline)
        by_condition[cond] = {
            "turn_accuracies_mean": curve,
            "n_values": ns_present,
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

    return {
        "model": model,
        "verdict": "complete",
        "n_values": n_values,
        "by_condition": by_condition,
        "asymmetry_ratio": asym,
        "baseline": baseline,
    }
