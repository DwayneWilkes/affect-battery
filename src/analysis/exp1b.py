"""Exp 1b three-way comparison scoring.

Per tasks.md Task 4.2 + conditioning-protocol spec
"Three-way comparison required": for each non-baseline condition the
analysis produces three side-by-side cells:
  - session_1_effect_size: within-session effect from Exp 1a corpus
  - session_2_effect_size: cross-session effect from Exp 1b corpus
  - no_conditioning_baseline: shared baseline accuracy from no_conditioning

This triangulation is the falsification structure: the paper §3.2.2
expectation is session_2_effect_size near zero (context-attention
mechanism). Effect-size primitives are reused from
src.analysis._effect_size for DRY parity with Exp 1a analysis.
"""

from __future__ import annotations

from statistics import mean

from src.analysis._effect_size import cohens_d, pooled_sd, run_accuracy, welch_p
from src.analysis.stats.tost import tost_equivalence
from src.conditioning.prompts import Condition


def _by_condition(corpus: list[dict]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for r in corpus:
        grouped.setdefault(r["condition"], []).append(run_accuracy(r))
    return grouped


def _se_of_d(treatment: list[float], baseline: list[float]) -> float:
    """Approximate standard error of Cohen's d (Hedges & Olkin, 1985).

    Useful for the TOST proxy where we want SE on the d scale itself.
    Returns 0.0 when sample sizes are insufficient (which the caller
    should treat as "test cannot be run").
    """
    import math

    nx, ny = len(treatment), len(baseline)
    if nx < 2 or ny < 2:
        return 0.0
    sd = pooled_sd(treatment, baseline)
    if sd == 0.0:
        return 0.0
    d = (sum(treatment) / nx - sum(baseline) / ny) / sd
    return math.sqrt((nx + ny) / (nx * ny) + (d * d) / (2 * (nx + ny)))


def analyze_exp1b(
    exp1a_corpus: list[dict],
    exp1b_corpus: list[dict],
    model: str,
    h1b_dual_tests: bool = False,
    tost_epsilon: float = 0.10,
    alpha: float = 0.05,
) -> dict:
    """Three-way comparison: session-1 (within), session-2 (cross), baseline.

    Each non-baseline condition gets a cell with both session-1 and session-2
    effect sizes computed against the no_conditioning baseline from the
    respective corpus.
    """
    baseline_key = Condition.NO_CONDITIONING.value
    s1 = _by_condition(exp1a_corpus)
    s2 = _by_condition(exp1b_corpus)

    s1_baseline = s1.get(baseline_key, [])
    s2_baseline = s2.get(baseline_key, [])

    if not s1_baseline or not s2_baseline:
        return {
            "model": model,
            "verdict": "unavailable_no_baseline",
            "three_way_comparison": {},
            "annotation": (
                "no_conditioning baseline absent in Exp 1a or Exp 1b corpus; "
                "three-way comparison cannot be computed."
            ),
        }

    s1_baseline_mean = mean(s1_baseline)
    s2_baseline_mean = mean(s2_baseline)

    # Union of non-baseline conditions across the two corpora (each session
    # may have ran a different subset; we still emit a cell when at least
    # one session has data, with None for the missing side).
    conds = (set(s1.keys()) | set(s2.keys())) - {baseline_key}

    comparison: dict[str, dict] = {}
    for cond in conds:
        s1_cell = s1.get(cond, [])
        s2_cell = s2.get(cond, [])
        s1_d = cohens_d(s1_cell, s1_baseline) if s1_cell else None
        s2_d = cohens_d(s2_cell, s2_baseline) if s2_cell else None
        cell = {
            "session_1_effect_size": s1_d,
            "session_2_effect_size": s2_d,
            "no_conditioning_baseline": s1_baseline_mean,
            "session_1_n_runs": len(s1_cell),
            "session_2_n_runs": len(s2_cell),
            "session_1_mean_accuracy": mean(s1_cell) if s1_cell else None,
            "session_2_mean_accuracy": mean(s2_cell) if s2_cell else None,
        }
        if h1b_dual_tests and s2_cell:
            # Directional one-sided p-value: H0 = session_2_effect <= 0,
            # reject when effect is positive. We use welch_p (two-sided)
            # halved when the observed effect is in the predicted direction;
            # weekend-ship proxy.
            two_sided = welch_p(s2_cell, s2_baseline)
            observed_diff = mean(s2_cell) - s2_baseline_mean
            directional_p = (two_sided / 2.0) if observed_diff > 0 else (1.0 - two_sided / 2.0)
            cell["session_2_directional_p"] = directional_p
            # TOST equivalence on the d-scale within +/- epsilon.
            se_d = _se_of_d(s2_cell, s2_baseline)
            tost = tost_equivalence(
                effect=s2_d if s2_d is not None and abs(s2_d) != float("inf") else 0.0,
                se=se_d if se_d > 0 else 1e-9,
                epsilon=tost_epsilon,
                alpha=alpha,
            )
            cell["session_2_tost_p"] = tost["p_tost"]
            cell["session_2_equivalent"] = tost["equivalent"]
        comparison[cond] = cell

    return {
        "model": model,
        "verdict": "complete",
        "three_way_comparison": comparison,
        "session_1_baseline_mean": s1_baseline_mean,
        "session_2_baseline_mean": s2_baseline_mean,
    }
