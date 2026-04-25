"""Test the end-to-end analysis pipeline .

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

    def test_pipeline_renders_h4_when_multiple_models_present(self, tmp_path):
        """A4 + A5 + A6: when Exp 1a corpus contains both base + instruct
        models with proper conditioning patterns, the pipeline should
        render h4_report.md and include primary_family_corrections in
        the aggregate payload."""
        from src.analysis.pipeline import analyze_results_dir

        exp1a_dir = tmp_path / "exp1a"
        exp1a_dir.mkdir(parents=True)

        # Synthesize per-(model, condition) runs with realistic accuracy
        # gaps so manipulation_check passes.
        models_data = {
            "Meta-Llama-3-8B": {
                "no_conditioning": ([True, True, True, True, True], [True, True, True, True, False]),
                "strong_positive": ([True, True, True, True, False], [True, True, True, True, True]),
                "strong_negative": ([True, False, False, False, False], [False, False, False, True, True]),
            },
            "Meta-Llama-3-8B-Instruct": {
                "no_conditioning": ([True, True, True, True, True], [True, True, True, True, False]),
                "strong_positive": ([True, True, True, True, True], [True, True, True, True, True]),
                "strong_negative": ([False, False, False, False, False], [False, False, True, False, False]),
            },
        }
        idx = 0
        for model, conds in models_data.items():
            for cond, (tc, cc) in conds.items():
                for _ in range(5):
                    payload = {
                        "config": {"condition": cond, "model_name": model},
                        "run_number": idx,
                        "experiment_type": "exp1a",
                        "model": model,
                        "condition": cond,
                        "transfer_correct": tc,
                        "conditioning_correct": cc,
                        "checksum": "0" * 16,
                    }
                    (exp1a_dir / f"run_{idx}.json").write_text(json.dumps(payload))
                    idx += 1

        rendered = analyze_results_dir(
            results_dir=tmp_path,
            model="Llama-3-8B-family",
            base_model="Meta-Llama-3-8B",
            instruct_model="Meta-Llama-3-8B-Instruct",
        )
        assert "exp1a" in rendered
        assert "h4" in rendered, "h4 report should render when both models present"
        assert rendered["h4"].exists()

        # Aggregate should mention H4 contrast
        agg_content = rendered["aggregate"].read_text()
        assert "H4" in agg_content
        # Should also include a corrections table since we have H1 from
        # exp1a (and H4 from the contrast)
        assert "Holm" in agg_content or "corrected" in agg_content.lower()

    def test_empty_results_dir_still_renders_aggregate(self, tmp_path):
        """An empty results dir produces an aggregate report with 'No data'
        entries rather than crashing."""
        from src.analysis.pipeline import analyze_results_dir

        rendered = analyze_results_dir(results_dir=tmp_path, model="dry-run")
        assert rendered["aggregate"].exists()
        content = rendered["aggregate"].read_text()
        assert "Exp 1a" in content
