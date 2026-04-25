"""A1 Red — Exp 2 cross-N aggregation.

Per persistence-dynamics spec + aggregate Exp 2 result
JSONs across the N-values sweep into the analysis dict that
render_exp2_report expects.
"""

from __future__ import annotations


def _make_run(condition: str, n_value: int, turn_accuracies: list[float]) -> dict:
    return {
        "experiment_type": "exp2",
        "model": "dry-run",
        "condition": condition,
        "body": {
            "n_value": n_value,
            "turn_accuracies": turn_accuracies,
        },
    }


class TestAnalyzeExp2Corpus:
    def test_aggregates_across_n_values(self):
        from src.analysis.exp2 import analyze_exp2_corpus

        # 4 N values × 4 conditions, 2 runs per cell. Per persistence-
        # dynamics spec: NEUTRAL conditioning is the recovery control;
        # NO_CONDITIONING provides a separate scalar reference.
        corpus: list[dict] = []
        for cond, accs_by_n in [
            ("no_conditioning", {1: 0.85, 3: 0.85, 5: 0.85, 10: 0.85}),
            ("neutral", {1: 0.85, 3: 0.85, 5: 0.85, 10: 0.85}),
            ("strong_negative", {1: 0.20, 3: 0.40, 5: 0.60, 10: 0.70}),
            ("strong_positive", {1: 0.90, 3: 0.95, 5: 1.00, 10: 1.00}),
        ]:
            for n, mean_acc in accs_by_n.items():
                for _ in range(2):
                    corpus.append(_make_run(cond, n, [mean_acc] * n))

        analysis = analyze_exp2_corpus(corpus, model="dry-run")

        assert analysis["verdict"] == "complete"
        assert analysis["n_values"] == [1, 3, 5, 10]
        # baseline pulled from no_conditioning corpus (scalar reference)
        assert abs(analysis["baseline"] - 0.85) < 1e-6
        # Control curve from NEUTRAL conditioning runs
        assert analysis["control_curve"] == [0.85, 0.85, 0.85, 0.85]
        # Both non-baseline conditions present in by_condition
        assert "strong_negative" in analysis["by_condition"]
        assert "strong_positive" in analysis["by_condition"]
        # Decay fits + recovery metrics populated
        sn = analysis["by_condition"]["strong_negative"]
        assert "decay_fit" in sn
        assert "exponential" in sn["decay_fit"]
        assert "linear" in sn["decay_fit"]
        assert "recovery_metrics" in sn
        # asymmetry_ratio = |neg_auc| / |pos_auc|
        assert analysis["asymmetry_ratio"] is not None
        assert analysis["asymmetry_ratio"] > 0

    def test_no_neutral_control_returns_unavailable(self):
        from src.analysis.exp2 import analyze_exp2_corpus

        corpus = [_make_run("strong_negative", 5, [0.5, 0.5, 0.5, 0.5, 0.5])]
        analysis = analyze_exp2_corpus(corpus, model="dry-run")
        assert analysis["verdict"] == "unavailable_no_control"

    def test_empty_corpus_returns_unavailable(self):
        from src.analysis.exp2 import analyze_exp2_corpus

        analysis = analyze_exp2_corpus([], model="dry-run")
        assert analysis["verdict"] == "unavailable_no_control"
