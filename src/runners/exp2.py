"""Exp 2 — persistence / recovery dynamics (paper §3.3).

Per persistence-dynamics spec "Persistence protocol structure" + "N-values
sweep": Exp 2 extends Exp 1a's conditioning phase with
N neutral turns on diverse tasks. The neutral_turns count is the recovery
phase length; per-turn accuracy across those turns is captured on Exp2Body
for downstream recovery-metric computation.

Implementation reuses run_batch + run_single's existing neutral_turns
loop. The PERSISTENCE branch in run_single now also records per-turn
accuracies and attaches Exp2Body. DRY check: conditioning
phase reuses Exp 1a's protocol verbatim — no parallel implementation.
"""

from __future__ import annotations

from src.runner import ExperimentType, run_batch


async def run_exp2(config, client, **kwargs):
    """Run Exp 2 persistence-dynamics."""
    if config.experiment_type != ExperimentType.PERSISTENCE:
        raise ValueError(
            f"run_exp2 requires config.experiment_type=PERSISTENCE; "
            f"got {config.experiment_type!r}"
        )
    if config.neutral_turns <= 0:
        raise ValueError(
            f"run_exp2 requires config.neutral_turns > 0 (the N-value); "
            f"got {config.neutral_turns}"
        )
    async for result in run_batch(config, client, **kwargs):
        yield result
