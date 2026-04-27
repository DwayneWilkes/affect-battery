"""Exp 1b — cross-session falsification test (paper §3.2.2).

Per design.md D6 + conditioning-protocol spec
"Cross-session session seeds are recorded separately": Exp 1b extends
Exp 1a with a Phase-2 fresh-session re-test. The cross-session branch in
`run_single` resets the system prompt to CROSS_SESSION_SYSTEM_PROMPT
("Answer the following questions to the best of your ability."), and
session-1 / session-2 seeds are recorded distinctly on the Exp1bBody.

Implementation reuses run_batch: the only contract that differs from
Exp 1a is `experiment_type=TRANSFER_CROSS`, which run_single already
honors via the `if config.experiment_type == ExperimentType.TRANSFER_CROSS`
branch (resets messages with the neutral system prompt before transfer).
"""

from __future__ import annotations

from src.runner import ExperimentType, run_batch


async def run_exp1b(config, client, **kwargs):
    """Run Exp 1b cross-session falsification.

    config.experiment_type MUST be TRANSFER_CROSS; this is enforced rather
    than coerced so misconfigured callers fail loudly instead of silently
    running an Exp 1a within-session protocol under an "exp1b" label.
    """
    if config.experiment_type != ExperimentType.TRANSFER_CROSS:
        raise ValueError(
            f"run_exp1b requires config.experiment_type=TRANSFER_CROSS; "
            f"got {config.experiment_type!r}"
        )
    async for result in run_batch(config, client, **kwargs):
        yield result
