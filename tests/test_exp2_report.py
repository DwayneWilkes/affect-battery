"""Exp 2 report with §10 decay-shape caveat.

Per persistence-dynamics spec "Decay-shape interpretation is hedged":
when decay-shape comparison results are presented, the accompanying
text MUST include the §10 caveat that curve shape alone does not
distinguish mood-inertia from context-attention decay.
"""

from __future__ import annotations


class TestRenderExp2Report:
    def test_section_10_caveat_present(self, tmp_path):
        from src.analysis.reports.exp2 import render_exp2_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "n_values": [1, 3, 5, 10],
            "by_condition": {
                "strong_negative": {
                    "turn_accuracies_mean": [0.2, 0.4, 0.6, 0.7],
                    "decay_fit": {
                        "exponential": {"amplitude": -0.6, "tau": 4.0, "aic": -10.0, "bic": -9.0},
                        "linear": {"slope": 0.05, "aic": -5.0, "bic": -4.5},
                    },
                    "recovery_metrics": {
                        "neg_time_to_baseline": 7.5,
                        "neg_auc": -2.3,
                    },
                },
                "strong_positive": {
                    "turn_accuracies_mean": [0.9, 0.95, 1.0, 1.0],
                    "decay_fit": {
                        "exponential": {"amplitude": 0.1, "tau": 3.0, "aic": -12.0, "bic": -11.0},
                        "linear": {"slope": 0.01, "aic": -8.0, "bic": -7.5},
                    },
                    "recovery_metrics": {
                        "pos_time_to_baseline": 1.0,
                        "pos_auc": 0.5,
                    },
                },
            },
            "asymmetry_ratio": 4.6,
            "baseline": 0.8,
        }

        report_path = tmp_path / "exp2_report.md"
        render_exp2_report(analysis, output_path=report_path)
        content = report_path.read_text()

        assert "§10" in content or "section 10" in content.lower()
        # The substantive caveat content per spec scenario
        assert "context-attention" in content.lower()
        assert "mood-inertia" in content.lower() or "shape alone" in content.lower()

    def test_decay_table_renders(self, tmp_path):
        from src.analysis.reports.exp2 import render_exp2_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "n_values": [1, 3, 5, 10],
            "by_condition": {
                "strong_negative": {
                    "turn_accuracies_mean": [0.2, 0.4, 0.6, 0.7],
                    "decay_fit": {
                        "exponential": {"amplitude": -0.6, "tau": 4.0, "aic": -10.0, "bic": -9.0},
                        "linear": {"slope": 0.05, "aic": -5.0, "bic": -4.5},
                    },
                    "recovery_metrics": {},
                },
            },
            "asymmetry_ratio": None,
            "baseline": 0.8,
        }
        report_path = tmp_path / "exp2_report.md"
        render_exp2_report(analysis, output_path=report_path)
        content = report_path.read_text()
        assert "Exp 2" in content
        assert "strong_negative" in content
        assert "AIC" in content or "aic" in content.lower()
