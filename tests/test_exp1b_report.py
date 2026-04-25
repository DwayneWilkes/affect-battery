"""Task 4.4 Red — Exp 1b report with null-confirms-context-attention framing.

Per conditioning-protocol spec "Null result is reported as confirming
context-attention": when session-2 effect is null (within +/- 0.10 by
TOST), the report labels the outcome as
"context-attention mechanism confirmed (paper §3.2.2 expected)" rather
than "H1b failed".
"""

from __future__ import annotations


class TestRenderExp1bReport:
    def test_null_reported_as_context_attention_confirmed(self, tmp_path):
        from src.analysis.reports.exp1b import render_exp1b_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "three_way_comparison": {
                "strong_negative": {
                    "session_1_effect_size": -1.5,
                    "session_2_effect_size": -0.02,  # essentially null
                    "no_conditioning_baseline": 0.85,
                    "session_1_n_runs": 50,
                    "session_2_n_runs": 50,
                    "session_1_mean_accuracy": 0.30,
                    "session_2_mean_accuracy": 0.83,
                    "session_2_directional_p": 0.55,
                    "session_2_tost_p": 0.001,
                    "session_2_equivalent": True,
                },
            },
        }
        report_path = tmp_path / "exp1b_report.md"
        render_exp1b_report(analysis, output_path=report_path)
        content = report_path.read_text()
        # Null framing per spec — DO NOT label as "H1b failed"
        assert "context-attention" in content.lower()
        assert "§3.2.2" in content or "section 3.2.2" in content.lower()
        assert "h1b failed" not in content.lower()

    def test_non_null_session_2_does_not_invoke_null_framing(self, tmp_path):
        """If session-2 effect is large, the report should NOT use the
        context-attention-confirmed framing."""
        from src.analysis.reports.exp1b import render_exp1b_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "three_way_comparison": {
                "strong_negative": {
                    "session_1_effect_size": -1.5,
                    "session_2_effect_size": -1.4,  # large effect persists across sessions
                    "no_conditioning_baseline": 0.85,
                    "session_1_n_runs": 50,
                    "session_2_n_runs": 50,
                    "session_1_mean_accuracy": 0.30,
                    "session_2_mean_accuracy": 0.32,
                    "session_2_directional_p": 0.99,
                    "session_2_tost_p": 0.99,
                    "session_2_equivalent": False,
                },
            },
        }
        report_path = tmp_path / "exp1b_report.md"
        render_exp1b_report(analysis, output_path=report_path)
        content = report_path.read_text().lower()
        assert "context-attention mechanism confirmed" not in content
        # The cross-session persistence is itself a notable finding to call out
        assert "session 2" in content or "session_2" in content

    def test_three_way_table_renders(self, tmp_path):
        from src.analysis.reports.exp1b import render_exp1b_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "three_way_comparison": {
                "strong_negative": {
                    "session_1_effect_size": -1.5,
                    "session_2_effect_size": -0.05,
                    "no_conditioning_baseline": 0.85,
                    "session_1_n_runs": 50,
                    "session_2_n_runs": 50,
                    "session_1_mean_accuracy": 0.30,
                    "session_2_mean_accuracy": 0.83,
                    "session_2_directional_p": 0.50,
                    "session_2_tost_p": 0.01,
                    "session_2_equivalent": True,
                },
            },
        }
        report_path = tmp_path / "exp1b_report.md"
        render_exp1b_report(analysis, output_path=report_path)
        content = report_path.read_text()
        # Three-way comparison table content: condition + both effect sizes
        assert "strong_negative" in content
        assert "Exp 1b" in content
