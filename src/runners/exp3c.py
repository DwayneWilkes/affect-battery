"""Exp 3c — paper §3.4.3.

Stub async-generator — implemented in Task 7.3. Matches `run_batch`
signature so the CLI dispatch table can call it via
`async for result in run_exp3c(...)`.
"""

from __future__ import annotations


async def run_exp3c(config, client, **kwargs):
    raise NotImplementedError(
        "Exp 3c runner not yet implemented; see tasks.md Task 7.3"
    )
    yield  # pragma: no cover — marks this as an async generator
