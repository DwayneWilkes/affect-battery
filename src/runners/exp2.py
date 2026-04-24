"""Exp 2 — paper §3.3.

Stub async-generator — implemented in Task 5.1. Matches `run_batch`
signature so the CLI dispatch table can call it via
`async for result in run_exp2(...)`.
"""

from __future__ import annotations


async def run_exp2(config, client, **kwargs):
    raise NotImplementedError(
        "Exp 2 runner not yet implemented; see tasks.md Task 5.1"
    )
    yield  # pragma: no cover — marks this as an async generator
