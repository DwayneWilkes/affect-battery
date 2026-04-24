"""MDE update logic (Task 1.3).

Per power-analysis spec "Per-hypothesis MDE coverage with grounded
defaults" + design.md D3:

If pilot observes a tighter effect size than the paper-prior default,
use MAX(observed, default) so we never commit to detecting a smaller
effect than our pilot variance can support.
"""

from __future__ import annotations

from typing import Any


def update_mde_for_hypothesis(
    hypothesis_id: str,
    default_mde: float,
    observed_effect_size: float | None,
) -> dict[str, Any]:
    """Compute the actual MDE to use for power simulation.

    Returns a dict with mde_used, mde_source, and the inputs for audit.

    Rule (per design.md D3 variance-probe-override):
    - If observed_effect_size is None: use default.
    - If observed > default: use observed (ground in pilot).
    - If observed ≤ default: use default (conservative).
    """
    if observed_effect_size is None or observed_effect_size <= default_mde:
        return {
            "hypothesis_id": hypothesis_id,
            "mde_used": default_mde,
            "mde_source": "default",
            "default_mde": default_mde,
            "observed_effect_size": observed_effect_size,
        }
    return {
        "hypothesis_id": hypothesis_id,
        "mde_used": observed_effect_size,
        "mde_source": "pilot_observed",
        "default_mde": default_mde,
        "observed_effect_size": observed_effect_size,
    }
