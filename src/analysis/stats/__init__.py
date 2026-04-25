"""Statistical primitives shared across per-experiment analyses.

Package layout:
- _manipulation_check: ManipulationVerdict, ManipulationCheckResult,
  manipulation_check (was src/analysis/stats.py before Task 4.3 expanded
  this into a package).
- tost: TOST equivalence test (Task 4.3).

Re-exports preserve the historical `from src.analysis.stats import ...`
API used across calibration / manipulation-check report code.
"""

from src.analysis.stats._manipulation_check import (
    ManipulationCheckResult,
    ManipulationVerdict,
    manipulation_check,
)

__all__ = [
    "ManipulationCheckResult",
    "ManipulationVerdict",
    "manipulation_check",
]
