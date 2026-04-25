"""H4 joint-outcome 2x2 report.

Per asymmetry-contrast spec "Honest reporting of inconclusive outcomes"
+ "Joint-outcome table is always present": the H4 report enumerates a
2x2 joint-outcome table covering the cross of:
  {base asymmetry_ratio > 1 vs <= 1}
  x {asymmetry_delta_ratio > 1 vs <= 1}
The 2x2 table renders even when both axes resolve to "inconclusive"
(spec: honest reporting).
"""

from __future__ import annotations


class TestH4Report:
    def test_2x2_joint_table_present(self, tmp_path):
        from src.analysis.reports.h4 import render_h4_report

        result = {
            "model_family": "Llama-3-8B",
            "base_model": "Meta-Llama-3-8B",
            "instruct_model": "Meta-Llama-3-8B-Instruct",
            "ratio_base": 1.5,
            "ratio_instruct": 3.0,
            "asymmetry_delta_ratio": 2.0,
            "test_a_primary_delta_ratio_gt_1": True,
            "test_b_secondary_diff_instruct_gt_diff_base": True,
            "per_model_verdicts": {
                "Meta-Llama-3-8B": "supported",
                "Meta-Llama-3-8B-Instruct": "supported",
            },
        }
        report_path = tmp_path / "h4_report.md"
        render_h4_report(result, output_path=report_path)
        content = report_path.read_text()
        # All four cells of the 2x2 enumerated
        assert "ratio_base > 1" in content
        assert "ratio_base <= 1" in content
        assert "delta_ratio > 1" in content
        assert "delta_ratio <= 1" in content
        # Both pre-registered tests reported
        assert "test_a" in content.lower() or "primary" in content.lower()

    def test_inconclusive_still_renders_table(self, tmp_path):
        from src.analysis.reports.h4 import render_h4_report

        result = {
            "model_family": "Llama-3-8B",
            "base_model": "B",
            "instruct_model": "I",
            "ratio_base": 0.95,
            "ratio_instruct": 1.05,
            "asymmetry_delta_ratio": 1.10,
            "test_a_primary_delta_ratio_gt_1": True,
            "test_b_secondary_diff_instruct_gt_diff_base": False,
            "per_model_verdicts": {
                "B": "inconclusive",
                "I": "inconclusive",
            },
        }
        report_path = tmp_path / "h4_report.md"
        render_h4_report(result, output_path=report_path)
        content = report_path.read_text()
        assert "inconclusive" in content.lower()
        assert "ratio_base > 1" in content
        assert "ratio_base <= 1" in content
