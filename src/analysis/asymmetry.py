"""Asymmetry compute + per-model H4 decision rule + base-vs-instruct contrast.

Per asymmetry-contrast spec:
- "Asymmetry conventions": ratio = |neg|/|pos|, difference = |neg|-|pos|.
- "Paired asymmetry computation": apply per (model, paired-arm).
- "Aggregation rule": geometric mean for ratios, arithmetic mean for
  differences.
- "Per-model H4 decision rule": 7-row decision table (Task 8.3).
- "Base-vs-instruct asymmetry contrast": delta_ratio = ratio_instruct /
  ratio_base; both pre-registered tests run (Task 8.2).
"""

from __future__ import annotations

import math
from typing import Iterable


# Threshold for "near zero positive" → undefined ratio. Below this we
# decline to compute a ratio rather than emit an inflated number.
NEAR_ZERO_POS_EPSILON = 0.01


def compute_pair(pos_effect: float, neg_effect: float) -> dict:
    """One (positive, negative) effect-size pair → ratio + difference.

    `ratio` uses |neg|/|pos|; returns None when |pos| < epsilon
    (positive arm too small to anchor a meaningful ratio). `difference`
    is |neg| - |pos|; always defined.
    """
    abs_pos = abs(pos_effect)
    abs_neg = abs(neg_effect)
    ratio = (abs_neg / abs_pos) if abs_pos >= NEAR_ZERO_POS_EPSILON else None
    return {
        "ratio": ratio,
        "difference": abs_neg - abs_pos,
        "abs_pos": abs_pos,
        "abs_neg": abs_neg,
    }


def _geometric_mean(values: Iterable[float]) -> float:
    """Geometric mean over a finite, all-positive iterable."""
    items = [v for v in values if v > 0]
    if not items:
        return 0.0
    log_sum = sum(math.log(v) for v in items)
    return math.exp(log_sum / len(items))


def compute_aggregate(pairs: list[dict]) -> dict:
    """Geomean over pairs[i].ratio (skipping None), arithmetic mean over
    pairs[i].difference."""
    if not pairs:
        return {"ratio_geomean": None, "difference_mean": None, "n_pairs": 0}
    ratios = [p["ratio"] for p in pairs if p.get("ratio") is not None]
    differences = [p["difference"] for p in pairs]
    return {
        "ratio_geomean": _geometric_mean(ratios) if ratios else None,
        "difference_mean": (sum(differences) / len(differences)) if differences else None,
        "n_pairs": len(pairs),
        "n_ratios_valid": len(ratios),
    }


def per_model_verdict(
    aggregate: dict,
    p_value: float | None,
    mde: float | None,
    alpha: float = 0.05,
) -> str:
    """Map an aggregate + p-value + MDE to one of 7 H4 decision-rule rows.

    Per asymmetry-contrast spec "Per-model H4 decision rule":
      supported     -> ratio significantly > 1
      below-MDE     -> ratio > 1 but observed magnitude below MDE
      inconclusive  -> p > alpha and observed magnitude near MDE; can't decide
      null          -> ratio ~ 1.0 (within band) and p > alpha (equivalence)
      reverse       -> ratio significantly < 1
      near-1        -> ratio_geomean within +/- 0.1 of 1.0 (no signal)
      degenerate    -> ratio_geomean is None (positive arm too small)
    """
    ratio = aggregate.get("ratio_geomean")
    if ratio is None:
        return "degenerate"

    near_one_band = 0.1
    if abs(ratio - 1.0) <= near_one_band and (p_value is None or p_value > alpha):
        return "near-1"

    if p_value is not None and p_value < alpha:
        if ratio > 1.0:
            return "supported"
        if ratio < 1.0:
            return "reverse"

    # p > alpha or no test ran. Distinguish below-MDE vs inconclusive vs null.
    if mde is not None and aggregate.get("difference_mean") is not None:
        observed = abs(aggregate["difference_mean"])
        if observed < mde and ratio > 1.0:
            return "below-MDE"

    if abs(ratio - 1.0) <= 0.05:
        return "null"

    return "inconclusive"


def contrast_base_vs_instruct(
    per_model_aggregates: dict[str, dict],
    base_model: str,
    instruct_model: str,
) -> dict:
    """Compute asymmetry_delta_ratio = ratio_instruct / ratio_base + report
    both pre-registered tests.

    (a) primary: delta_ratio > 1 (instruct shows more asymmetry than base).
    (b) secondary: difference_instruct > difference_base.
    """
    base_agg = per_model_aggregates.get(base_model)
    instruct_agg = per_model_aggregates.get(instruct_model)
    if base_agg is None or instruct_agg is None:
        raise ValueError(
            f"per_model_aggregates missing entries for "
            f"base_model={base_model!r} or instruct_model={instruct_model!r}"
        )
    r_base = base_agg.get("ratio_geomean")
    r_inst = instruct_agg.get("ratio_geomean")
    delta_ratio = (
        (r_inst / r_base) if (r_base and r_inst and r_base > 0) else None
    )
    diff_base = base_agg.get("difference_mean") or 0.0
    diff_inst = instruct_agg.get("difference_mean") or 0.0
    return {
        "base_model": base_model,
        "instruct_model": instruct_model,
        "ratio_base": r_base,
        "ratio_instruct": r_inst,
        "asymmetry_delta_ratio": delta_ratio,
        "test_a_primary_delta_ratio_gt_1": (
            delta_ratio is not None and delta_ratio > 1.0
        ),
        "test_b_secondary_diff_instruct_gt_diff_base": diff_inst > diff_base,
    }
