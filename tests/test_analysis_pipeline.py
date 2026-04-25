"""Test the end-to-end analysis pipeline (review-finding #11).

The analyze_* and render_*_report functions exist; this regression test
asserts they're reachable from production via analyze_results_dir() so
the next refactor can't accidentally re-orphan them.
"""

from __future__ import annotations

import json


def _write_run(out_dir, run_idx: int, condition: str, transfer_correct: list[bool]):
    """Drop a minimal Exp 1a-style result JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"run_{run_idx}.json").write_text(json.dumps({
        "config": {"condition": condition},
        "run_number": run_idx,
        "experiment_type": "exp1a",
        "model": "dry-run",
        "condition": condition,
        "transfer_correct": transfer_correct,
        "checksum": "0" * 16,
    }))


class TestAnalysisPipeline:
    def test_analyze_results_dir_renders_reports(self, tmp_path):
        from src.analysis.pipeline import analyze_results_dir

        # Synthesize one Exp 1a run per condition
        exp1a_dir = tmp_path / "exp1a"
        _write_run(exp1a_dir, 0, "no_conditioning", [True, True, True, True])
        _write_run(exp1a_dir, 1, "no_conditioning", [True, True, True, False])
        _write_run(exp1a_dir, 2, "strong_negative", [False, False, False, True])
        _write_run(exp1a_dir, 3, "strong_negative", [False, True, False, False])

        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")

        assert "exp1a" in rendered
        assert "aggregate" in rendered
        assert rendered["exp1a"].exists()
        assert rendered["aggregate"].exists()
        # Per-experiment report mentions the condition
        content = rendered["exp1a"].read_text()
        assert "strong_negative" in content
        # Aggregate report mentions the experiment section
        agg = rendered["aggregate"].read_text()
        assert "Exp 1a" in agg

    def test_empty_results_dir_still_renders_aggregate(self, tmp_path):
        """An empty results dir produces an aggregate report with 'No data'
        entries rather than crashing."""
        from src.analysis.pipeline import analyze_results_dir

        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        assert rendered["aggregate"].exists()
        content = rendered["aggregate"].read_text()
        assert "Exp 1a" in content
