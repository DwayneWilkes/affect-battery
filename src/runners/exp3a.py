"""Exp 3a — paper §3.4.1.

Stub async-generator — implemented in Task 6.3. Matches `run_batch`
signature so the CLI dispatch table can call it via
`async for result in run_exp3a(...)`.
"""

from __future__ import annotations


async def run_exp3a(config, client, **kwargs):
    raise NotImplementedError(
        "Exp 3a runner not yet implemented; see tasks.md Task 6.3"
    )
    yield  # pragma: no cover — marks this as an async generator
