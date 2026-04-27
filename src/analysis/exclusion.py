"""model-level exclusion from primary analysis.

Per scoring-pipeline spec (main) "Manipulation check gate": models
whose MC fails are excluded from primary effect-size aggregation.
UNAVAILABLE verdicts are NOT exclusions (they're measurement gaps).
"""

from __future__ import annotations

from typing import Iterable, Mapping, TypeVar

from src.analysis.stats import ManipulationVerdict


T = TypeVar("T")


def is_model_eligible_for_primary(
    model: str,
    mc_verdict: ManipulationVerdict,
) -> bool:
    """Return True iff the model passes (or is unavailable on) the
    manipulation check.

    UNAVAILABLE means we couldn't compute MC (e.g. NO_CONDITIONING
    runs missing); we do not exclude on that ground because the
    transfer effect can still be reported. FAIL means MC ran and
    showed no conditioning effect; we exclude per spec."""
    return mc_verdict != ManipulationVerdict.FAIL


def filter_results_by_mc(
    results: Iterable[T],
    mc_verdicts: Mapping[str, ManipulationVerdict],
    *,
    model_key: str = "model",
) -> list[T]:
    """Return only those results whose model is MC-eligible.

    Each result is expected to be a mapping with a `model_key` field.
    Models not in `mc_verdicts` are kept (no MC data ≠ exclusion).
    """
    kept: list[T] = []
    for r in results:
        if isinstance(r, dict):
            model = r.get(model_key, "")
        else:
            model = getattr(r, model_key, "")
        verdict = mc_verdicts.get(model, ManipulationVerdict.UNAVAILABLE)
        if is_model_eligible_for_primary(model, verdict):
            kept.append(r)
    return kept
