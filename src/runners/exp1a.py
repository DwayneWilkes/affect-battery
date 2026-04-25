"""Exp 1a — within-session cross-domain transfer (paper §3.2.1).

Currently a direct re-export of the general-purpose `run_batch` async
generator.  specializes this for the paper §3.2.1 6-arm
condition set + cross-domain TransferBank; until then, run_batch's
existing behavior IS Exp 1a.
"""

from __future__ import annotations

from src.runner import run_batch as run_exp1a  # noqa: F401 — public API


__all__ = ["run_exp1a"]
