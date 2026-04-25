"""Exp 3b + 3c reports with §10 caveats.

Per cognitive-scope-measurement spec "Every 3b report cites §10" +
conservative-shift-measurement spec "Every 3c report cites §10":
both reports MUST include a §10 caveat. 3b cites the proxy-not-direct
caveat (semantic-diversity proxy ≠ direct cognitive-scope measurement).
3c cites the conservative-shift-vs-mood-as-information caveat (paper
warns that conservative-shift output may reflect mood-as-information
heuristic rather than genuine epistemic recalibration).
"""

from __future__ import annotations


class TestExp3bReport:
    def test_report_cites_section_10_proxy(self, tmp_path):
        from src.analysis.reports.exp3b import render_exp3b_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "by_condition": {
                "strong_negative": {
                    "embedding_variance": 0.12,
                    "ngram_ratio": 0.45,
                    "n_generations": 30,
                },
            },
        }
        report_path = tmp_path / "exp3b_report.md"
        render_exp3b_report(analysis, output_path=report_path)
        content = report_path.read_text().lower()
        assert "§10" in report_path.read_text() or "section 10" in content
        # Spec-required framing: proxy, not direct
        assert "proxy" in content or "not direct" in content


class TestExp3cReport:
    def test_report_cites_section_10_mood_as_information(self, tmp_path):
        from src.analysis.reports.exp3c import render_exp3c_report

        analysis = {
            "model": "dry-run",
            "verdict": "complete",
            "by_condition_difficulty": {
                ("strong_negative", "hard"): {
                    "n_items": 20,
                    "hedging_rate_per_100w": 4.5,
                    "refusal_rate": 0.10,
                    "mean_response_length": 80,
                },
            },
        }
        report_path = tmp_path / "exp3c_report.md"
        render_exp3c_report(analysis, output_path=report_path)
        content = report_path.read_text()
        lower = content.lower()
        assert "§10" in content or "section 10" in lower
        # Spec-required framing: mood-as-information caveat
        assert "mood-as-information" in lower or "mood as information" in lower
