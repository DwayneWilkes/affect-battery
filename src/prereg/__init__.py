"""OSF pre-registration helpers (Tasks 2.0a / 2.1 / 2.2).

Per design.md D4 + power-analysis spec OSF gate. The lifecycle:
- v0 (Task 0.4): paper-prior MDE defaults
- submit v0 to OSF (Task 2.0a, parallel with Phase 1 probes)
- finalize v1 (Task 2.1): probe-grounded MDEs + base-feasibility decision
  + amendment_chain entry
- record OSF URL after upload (Task 2.2): prepare_for_submit
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
