"""Multiple-comparisons correction functions.

Pure math utilities implementing the correction policy from the
scoring-pipeline spec "Multiple-comparisons correction policy" Requirement
and design D8 of the affect-battery-task-difficulty-calibration change:

- Holm-Bonferroni (Holm, 1979) for the manipulation-check family.
  Controls family-wise error rate (FWER); appropriate for a fixed-size
  confirmatory family.
- Benjamini-Hochberg (Benjamini & Hochberg, 1995) for exploratory pairwise
  contrasts. Controls false-discovery rate (FDR); appropriate for a large
  screening family.

Both functions are pure: no I/O, no state. Input is a list of raw p-values;
output is a list of corrected q-values of the same length, in the same
positional order as the input (so position i of the output corresponds to
position i of the input).

References
----------
Holm, S. (1979). "A simple sequentially rejective multiple test procedure."
    Scandinavian Journal of Statistics, 6(2), 65-70.

Benjamini, Y., & Hochberg, Y. (1995). "Controlling the false discovery rate:
    a practical and powerful approach to multiple testing." Journal of the
    Royal Statistical Society Series B, 57(1), 289-300.
"""

from __future__ import annotations

import math


def _validate(pvals: list[float]) -> None:
    for p in pvals:
        # NaN fails any comparison, so check that explicitly.
        if not isinstance(p, (int, float)) or math.isnan(p):
            raise ValueError(f"p-value must be a finite number in [0, 1]; got {p!r}")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"p-value must lie in [0, 1]; got {p!r}")


def apply_holm_correction(pvals: list[float]) -> list[float]:
    """Return Holm-Bonferroni corrected q-values for ``pvals``.

    The Holm (1979) step-down procedure sorts p-values ascending and assigns
    q_(k) = max over j <= k of (m - j + 1) * p_(j), capped at 1.0. This
    cumulative max enforces the monotone-non-decreasing constraint on the
    corrected series. Output preserves input order: output[i] corresponds to
    input[i].

    Parameters
    ----------
    pvals : list[float]
        Raw p-values in [0, 1].

    Returns
    -------
    list[float]
        Corrected q-values of the same length and positional order.

    Raises
    ------
    ValueError
        If any p-value is outside [0, 1] or is NaN.
    """
    _validate(pvals)
    m = len(pvals)
    if m == 0:
        return []
    # Sort ascending, carry original indices.
    order = sorted(range(m), key=lambda i: pvals[i])
    qsorted = [0.0] * m
    running = 0.0
    for k, idx in enumerate(order):  # k is zero-indexed rank
        candidate = (m - k) * pvals[idx]
        if candidate > running:
            running = candidate
        qsorted[k] = min(running, 1.0)
    # Unsort back to input order.
    q_by_input = [0.0] * m
    for k, idx in enumerate(order):
        q_by_input[idx] = qsorted[k]
    return q_by_input


def apply_bh_correction(pvals: list[float]) -> list[float]:
    """Return Benjamini-Hochberg corrected q-values for ``pvals``.

    The BH (1995) step-up procedure sorts p-values ascending and assigns
    raw q_(k) = p_(k) * m / k, then enforces monotone non-decrease by taking
    a running minimum from the largest rank down. Output is capped at 1.0
    and preserves input order.

    Parameters
    ----------
    pvals : list[float]
        Raw p-values in [0, 1].

    Returns
    -------
    list[float]
        BH-corrected q-values of the same length and positional order.

    Raises
    ------
    ValueError
        If any p-value is outside [0, 1] or is NaN.
    """
    _validate(pvals)
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    # Raw q_(k) per rank (1-indexed k).
    raw = [pvals[order[k]] * m / (k + 1) for k in range(m)]
    # Step-up monotonicity: running minimum from highest rank down.
    qsorted = [0.0] * m
    running = math.inf
    for k in range(m - 1, -1, -1):
        if raw[k] < running:
            running = raw[k]
        qsorted[k] = min(running, 1.0)
    q_by_input = [0.0] * m
    for k, idx in enumerate(order):
        q_by_input[idx] = qsorted[k]
    return q_by_input
