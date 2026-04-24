"""Exp 1a — within-session cross-domain transfer (paper §3.2.1).

Delegates to the existing `src.runner.run_batch` for now; Task 3.2 will
add the paper-aligned Exp 1a specifics (logic-puzzle / factual-QA /
reading-comprehension transfer via `cross-domain-transfer-tasks`).
"""

from __future__ import annotations


async def run_exp1a(config, client, **kwargs):
    """Run Exp 1a (within-session transfer). Currently delegates to the
    general-purpose runner; Task 3.2 specializes this for the paper
    §3.2.1 6-arm condition set + cross-domain TransferBank."""
    from src.runner import run_batch
    return await run_batch(config, client, **kwargs)
