"""A2 Red — Exp 3b cross-condition aggregation."""

from __future__ import annotations


def _make_run(condition: str, prompt_id: str, generations: list[str]) -> dict:
    return {
        "experiment_type": "exp3b",
        "model": "dry-run",
        "condition": condition,
        "body": {
            "prompt_id": prompt_id,
            "generations": generations,
            "per_generation_seeds": list(range(len(generations))),
        },
    }


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    """Hash-based stand-in: text -> 8-dim unit vector."""
    import hashlib
    import math
    vectors = []
    for t in texts:
        digest = hashlib.sha256(t.encode("utf-8")).digest()[:8]
        v = [b / 255.0 for b in digest]
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        vectors.append(v)
    return vectors


class TestAnalyzeExp3bCorpus:
    def test_aggregates_by_condition(self):
        from src.analysis.exp3b import analyze_exp3b_corpus

        corpus = [
            _make_run("strong_negative", "p1", ["A.", "A.", "A."]),  # zero variance
            _make_run("strong_negative", "p2", ["B.", "B.", "B."]),
            _make_run("strong_positive", "p1", ["X.", "Y.", "Z."]),  # high variance
            _make_run("strong_positive", "p2", ["U.", "V.", "W."]),
        ]
        analysis = analyze_exp3b_corpus(
            corpus, model="dry-run", embedder=_fake_embedder,
        )
        assert analysis["verdict"] == "complete"
        assert "strong_negative" in analysis["by_condition"]
        assert "strong_positive" in analysis["by_condition"]
        # n_generations counts all generations across all prompts in cond
        assert analysis["by_condition"]["strong_negative"]["n_generations"] == 6
        assert analysis["by_condition"]["strong_positive"]["n_generations"] == 6
        # Embedding variance + n-gram ratio populated
        assert "embedding_variance" in analysis["by_condition"]["strong_negative"]
        assert "ngram_ratio" in analysis["by_condition"]["strong_negative"]

    def test_empty_corpus_returns_empty_by_condition(self):
        from src.analysis.exp3b import analyze_exp3b_corpus

        analysis = analyze_exp3b_corpus(
            [], model="dry-run", embedder=_fake_embedder,
        )
        assert analysis["by_condition"] == {}
