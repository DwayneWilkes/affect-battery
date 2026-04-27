"""Coverage for shared report-formatting helpers (`fmt_value` / `fmt_p` /
`fmt_d`). Each renderer is tiny but the helpers cover several edge
cases (None, NaN, +/-inf, signed-vs-unsigned, sub-precision) that the
per-experiment report tests don't reach with realistic inputs.
"""

from __future__ import annotations

import math


class TestFmtValue:
    def test_none_returns_em_dash(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(None) == "—"

    def test_bool_passes_through_str(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(True) == "True"
        assert fmt_value(False) == "False"

    def test_nan(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(math.nan) == "nan"

    def test_positive_infinity(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(math.inf) == "+inf"

    def test_negative_infinity(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(-math.inf) == "-inf"

    def test_unsigned_default_precision(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(0.123456) == "0.123"

    def test_signed_renders_with_sign(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value(0.5, signed=True).startswith("+")
        assert fmt_value(-0.5, signed=True).startswith("-")

    def test_signed_below_precision_floor_uses_extra_digit(self):
        from src.analysis.reports._format import fmt_value
        # |v| < 0.5 * 10^-3 = 0.0005 — falls into the precision+1 branch
        out = fmt_value(0.0001, precision=3, signed=True)
        # No leading +/- sign because the small-value branch is unsigned 4dp
        assert "0.0001" in out

    def test_str_passthrough(self):
        from src.analysis.reports._format import fmt_value
        assert fmt_value("supported") == "supported"


class TestFmtP:
    def test_none_returns_em_dash(self):
        from src.analysis.reports._format import fmt_p
        assert fmt_p(None) == "—"

    def test_below_floor(self):
        from src.analysis.reports._format import fmt_p
        assert fmt_p(0.0001) == "<0.001"

    def test_three_decimals(self):
        from src.analysis.reports._format import fmt_p
        assert fmt_p(0.0421) == "0.042"

    def test_nan(self):
        from src.analysis.reports._format import fmt_p
        assert fmt_p(math.nan) == "nan"

    def test_non_float_passes_through(self):
        from src.analysis.reports._format import fmt_p
        assert fmt_p("not_run") == "not_run"


class TestFmtD:
    def test_none_returns_em_dash(self):
        from src.analysis.reports._format import fmt_d
        assert fmt_d(None) == "—"

    def test_signed_two_decimals(self):
        from src.analysis.reports._format import fmt_d
        assert fmt_d(0.5) == "+0.50"
        assert fmt_d(-1.25) == "-1.25"

    def test_positive_infinity(self):
        from src.analysis.reports._format import fmt_d
        assert fmt_d(math.inf) == "+inf"

    def test_negative_infinity(self):
        from src.analysis.reports._format import fmt_d
        assert fmt_d(-math.inf) == "-inf"

    def test_nan(self):
        from src.analysis.reports._format import fmt_d
        assert fmt_d(math.nan) == "nan"

    def test_non_float_passes_through(self):
        from src.analysis.reports._format import fmt_d
        assert fmt_d("degenerate") == "degenerate"
