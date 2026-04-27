"""Aggregate cross-experiment report.

Per design.md:D11 Phase 9: top-level results/AGGREGATE_REPORT.md
contains sections for each of Exp 1a, 1b, 2, 3a, 3b, 3c plus H4
cross-experiment.
"""

from __future__ import annotations


class TestAggregateReport:
    def test_aggregate_has_all_experiments(self, tmp_path):
        from src.analysis.reports.aggregate import render_aggregate

        all_results = {
            "exp1a": {"model": "Llama-3-8B-Instruct", "verdict": "complete"},
            "exp1b": {"model": "Llama-3-8B-Instruct", "verdict": "complete"},
            "exp2": {"model": "Llama-3-8B-Instruct", "verdict": "complete"},
            "exp3a": {"model": "Llama-3-8B-Instruct", "verdict": "complete"},
            "exp3b": {"model": "Llama-3-8B-Instruct", "verdict": "complete"},
            "exp3c": {"model": "Llama-3-8B-Instruct", "verdict": "complete"},
            "h4": {"model_family": "Llama-3-8B"},
            "primary_family_corrections": {
                "H1": {"raw": 0.001, "corrected": 0.005, "family": "primary"},
                "H2": {"raw": 0.04, "corrected": 0.08, "family": "primary"},
            },
        }
        report_path = tmp_path / "AGGREGATE_REPORT.md"
        render_aggregate(all_results, output_path=report_path)
        content = report_path.read_text()
        for exp in ("Exp 1a", "Exp 1b", "Exp 2", "Exp 3a", "Exp 3b", "Exp 3c"):
            assert exp in content, f"missing section for {exp}"
        # H4 cross-experiment section present
        assert "H4" in content
        # Family-wise corrections summary
        assert "Holm" in content or "primary family" in content.lower()
