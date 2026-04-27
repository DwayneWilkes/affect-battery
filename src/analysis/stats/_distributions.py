"""Cumulative-distribution-function helpers shared across stats modules.

`normal_cdf` and `student_t_cdf` are imported by tost.py and exp3a.py
so any test needing a normal or t-tail uses one canonical implementation.
"""

from __future__ import annotations

import math

from scipy.stats import t as _student_t


def normal_cdf(z: float) -> float:
    """Standard-normal CDF Phi(z) via erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def student_t_cdf(z: float, df: float) -> float:
    """Student-t CDF; thin wrapper over scipy.stats.t for naming parity
    with normal_cdf at the call site.
    """
    return float(_student_t.cdf(z, df=df))
