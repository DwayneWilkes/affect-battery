"""Statistical primitives shared across per-experiment analyses.

Package layout:
- _manipulation_check: ManipulationVerdict, ManipulationCheckResult,
  manipulation_check (formerly src/analysis/stats.py, expanded
  this into a package).
- tost: TOST equivalence test.

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
