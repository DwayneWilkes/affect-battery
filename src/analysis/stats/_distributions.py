"""Cumulative-distribution-function helpers shared across stats modules.

Per review-finding #13: `normal_cdf` was duplicated across _effect_size,
tost, and exp3a. Extracted here so any future test that needs a normal
or t-tail uses one canonical implementation.
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
