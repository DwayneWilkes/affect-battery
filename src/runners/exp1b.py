"""Exp 1b — cross-session falsification test (paper §3.2.2).

Stub async-generator — implemented in Task 4.1. Matches `run_batch`
signature so the CLI dispatch table can call it via
`async for result in run_exp1b(...)`.
"""

from __future__ import annotations


async def run_exp1b(config, client, **kwargs):
    raise NotImplementedError(
        "Exp 1b runner not yet implemented; see tasks.md Task 4.1"
    )
    yield  # pragma: no cover — marks this as an async generator
