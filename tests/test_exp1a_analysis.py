"""Tasks 3.4 + 3.5 Red — Exp 1a H1 analysis + report.

Per power-analysis spec H1 + scoring-pipeline spec per-experiment views.
Simplified for the weekend ship: Cohen's d per condition vs
no-conditioning baseline + Holm-Bonferroni correction across the
primary family. Full lme4/pymer4 mixed-effects fit is a follow-up
task (logged in commit message).
"""

from pathlib import Path

import pytest


class TestAnalyzeExp1a:
    def test_per_condition_effect_sizes(self):
        from src.analysis.exp1a import analyze_exp1a_corpus

        # Synthetic corpus: 3 conditions × 5 transfer-correctness scores
        corpus = [
            {"condition": "no_conditioning", "transfer_correct": [True, True, True, False, True]},  # acc 0.8
            {"condition": "no_conditioning", "transfer_correct": [True, True, False, True, True]},  # 0.8
            {"condition": "strong_negative", "transfer_correct": [False, True, False, False, True]},  # 0.4
            {"condition": "strong_negative", "transfer_correct": [True, False, False, False, False]},  # 0.2
            {"condition": "strong_positive", "transfer_correct": [True, True, True, True, True]},  # 1.0
            {"condition": "strong_positive", "transfer_correct": [True, True, True, False, True]},  # 0.8
        ]

        analysis = analyze_exp1a_corpus(corpus, model="dry-run")
        per_condition = analysis["per_condition_vs_baseline"]
        # All non-baseline conditions present
        assert "strong_negative" in per_condition
        assert "strong_positive" in per_condition
        # No-conditioning is the baseline; not present as comparison key
        assert "no_conditioning" not in per_condition

    def test_holm_correction_applied(self):
        from src.analysis.exp1a import analyze_exp1a_corpus

        corpus = [
            {"condition": "no_conditioning", "transfer_correct": [True] * 10},
            {"condition": "no_conditioning", "transfer_correct": [True] * 10},
            {"condition": "strong_negative", "transfer_correct": [False] * 10},
            {"condition": "strong_negative", "transfer_correct": [False] * 10},
        ]
        analysis = analyze_exp1a_corpus(corpus, model="dry-run")
        # Each non-baseline condition has both raw + corrected p-values
        sn = analysis["per_condition_vs_baseline"]["strong_negative"]
        assert "p_raw" in sn
        assert "p_holm_corrected" in sn

    def test_handles_no_baseline_gracefully(self):
        """If no_conditioning runs are absent, return UNAVAILABLE verdict."""
        from src.analysis.exp1a import analyze_exp1a_corpus

        corpus = [
            {"condition": "strong_negative", "transfer_correct": [True, False]},
        ]
        analysis = analyze_exp1a_corpus(corpus, model="dry-run")
        assert analysis["verdict"] == "unavailable_no_baseline"


class TestRenderExp1aReport:
    def test_writes_markdown_report(self, tmp_path):
        from src.analysis.reports.exp1a import render_exp1a_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "per_condition_vs_baseline": {
                "strong_negative": {
                    "n_runs": 5,
                    "mean_accuracy": 0.30,
                    "baseline_mean": 0.80,
                    "cohens_d": -1.5,
                    "p_raw": 0.001,
                    "p_holm_corrected": 0.005,
                },
            },
        }
        report_path = tmp_path / "exp1a_report.md"
        render_exp1a_report(analysis, output_path=report_path)
        assert report_path.exists()
        content = report_path.read_text()
        assert "Exp 1a" in content
        assert "strong_negative" in content
        assert "cohen" in content.lower()  # Cohen's d label

    def test_report_cites_holm_correction(self, tmp_path):
        from src.analysis.reports.exp1a import render_exp1a_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "per_condition_vs_baseline": {
                "strong_negative": {
                    "n_runs": 5, "mean_accuracy": 0.30, "baseline_mean": 0.80,
                    "cohens_d": -1.5, "p_raw": 0.001, "p_holm_corrected": 0.005,
                },
            },
        }
        report_path = tmp_path / "exp1a_report.md"
        render_exp1a_report(analysis, output_path=report_path)
        content = report_path.read_text()
        assert "Holm" in content
