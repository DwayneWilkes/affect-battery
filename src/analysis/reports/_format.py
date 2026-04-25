"""Formatting helpers shared across per-experiment report renderers.

`fmt_value`, `fmt_p`, and `fmt_d` cover the value-formatting needs of
every per-experiment report. New reports should import from here rather
than redefining a local helper.
"""

from __future__ import annotations

import math


def fmt_value(v, precision: int = 3, signed: bool = False) -> str:
    """Format a value for a markdown table cell.

    None -> '—'.  Floats get f'{:.{precision}f}' or signed equivalent;
   /- inf render as '+inf' / '-inf'.  Everything else falls through
    to str().
    """
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if v == math.inf:
            return "+inf"
        if v == -math.inf:
            return "-inf"
        if signed:
            return f"{v:+.{precision}f}" if abs(v) >= 0.5 * 10 ** -precision else f"{v:.{precision + 1}f}"
        return f"{v:.{precision}f}"
    return str(v)


def fmt_p(p: float | None) -> str:
    """Format a p-value: '<0.001' floor, three decimals otherwise."""
    if p is None:
        return "—"
    if not isinstance(p, float):
        return str(p)
    if math.isnan(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_d(d: float | None) -> str:
    """Format a Cohen's d (or other signed effect size); two decimals,
    explicit sign, +/- inf for perfect-separation."""
    if d is None:
        return "—"
    if not isinstance(d, float):
        return str(d)
    if math.isnan(d):
        return "nan"
    if d == math.inf:
        return "+inf"
    if d == -math.inf:
        return "-inf"
    return f"{d:+.2f}"
