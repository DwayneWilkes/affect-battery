"""OSF pre-registration helpers (Tasks 2.0a / 2.1 / 2.2).

Per design.md D4 + power-analysis spec OSF gate. The lifecycle:
- v0: paper-prior MDE defaults
- submit v0 to OSF (parallel with the probe phase)
- finalize v1: probe-grounded MDEs + base-feasibility decision
  amendment_chain entry
- record OSF URL after upload: prepare_for_submit
"""

from .finalize import (
    compute_prereg_sha,
    finalize_v1,
    prepare_for_submit,
)


__all__ = [
    "compute_prereg_sha",
    "finalize_v1",
    "prepare_for_submit",
]
