"""Exp 3b — paper §3.4.2.

Stub async-generator — implemented in Task 7.1. Matches `run_batch`
signature so the CLI dispatch table can call it via
`async for result in run_exp3b(...)`.
"""

from __future__ import annotations


async def run_exp3b(config, client, **kwargs):
    raise NotImplementedError(
        "Exp 3b runner not yet implemented; see tasks.md Task 7.1"
    )
    yield  # pragma: no cover — marks this as an async generator
