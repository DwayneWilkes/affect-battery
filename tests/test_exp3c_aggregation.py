"""A3 Red — Exp 3c cross-(condition, difficulty) aggregation."""

from __future__ import annotations


def _make_run(condition: str, difficulty: str, response: str, refused: bool = False) -> dict:
    return {
        "experiment_type": "exp3c",
        "model": "dry-run",
        "condition": condition,
        "body": {
            "difficulty": difficulty,
            "question": "Q?",
            "response": response,
            "expected": "A",
            "stated_confidence": None,
            "refused": refused,
        },
    }


class TestAnalyzeExp3cCorpus:
    def test_aggregates_by_condition_and_difficulty(self):
        from src.analysis.exp3c import analyze_exp3c_corpus

        corpus = [
            # strong_negative + easy: 2 items, 1 hedge phrase, 1 refusal
            _make_run("strong_negative", "easy", "I think the answer is 4."),
            _make_run("strong_negative", "easy", "I cannot answer.", refused=True),
            # strong_negative + hard: 2 items, no hedges
            _make_run("strong_negative", "hard", "Definitely 42."),
            _make_run("strong_negative", "hard", "The answer is 7."),
            # neutral + easy: 2 items, no hedges
            _make_run("neutral", "easy", "Paris."),
            _make_run("neutral", "easy", "Madrid."),
        ]

        analysis = analyze_exp3c_corpus(corpus, model="dry-run")
        assert analysis["verdict"] == "complete"
        cells = analysis["by_condition_difficulty"]
        # Three (condition, difficulty) cells
        assert ("strong_negative", "easy") in cells
        assert ("strong_negative", "hard") in cells
        assert ("neutral", "easy") in cells
        # n_items per cell
        assert cells[("strong_negative", "easy")]["n_items"] == 2
        # refusal rate: 1 of 2
        assert cells[("strong_negative", "easy")]["refusal_rate"] == 0.5
        # neutral easy has zero refusals
        assert cells[("neutral", "easy")]["refusal_rate"] == 0.0
        # Hedging rate is per 100 words; non-zero in cells with hedges
        assert cells[("strong_negative", "easy")]["hedging_rate_per_100w"] > 0

    def test_empty_corpus_returns_empty(self):
        from src.analysis.exp3c import analyze_exp3c_corpus

        analysis = analyze_exp3c_corpus([], model="dry-run")
        assert analysis["by_condition_difficulty"] == {}
