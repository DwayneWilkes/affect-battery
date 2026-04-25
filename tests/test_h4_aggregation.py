"""A4 + A5 Red — H4 aggregation + manipulation-check from Exp 1a corpus.

Per asymmetry-contrast spec + review-findings A4/A5: stitch per-model
H4 aggregation from Exp 1a corpus (Cohen's d for strong_positive vs
strong_negative against no_conditioning baseline), apply manipulation-
check to determine model exclusion, then run the base-vs-instruct
contrast on the surviving models.
"""

from __future__ import annotations


def _exp1a_run(model: str, condition: str, transfer_correct: list[bool],
               conditioning_correct: list[bool] | None = None) -> dict:
    cc = conditioning_correct if conditioning_correct is not None else [True] * 5
    return {
        "experiment_type": "exp1a",
        "model": model,
        "condition": condition,
        "conditioning_correct": cc,
        "transfer_correct": transfer_correct,
    }


class TestManipulationCheckFromCorpus:
    def test_passes_when_pos_and_neg_diverge_from_baseline(self):
        from src.analysis.h4 import manipulation_check_from_corpus

        # Synthesize: pos arm conditioning ~0.95, baseline ~0.85, neg arm ~0.60.
        # Both directions exceed 2pp threshold => PASS.
        corpus = []
        for _ in range(5):
            corpus.append(_exp1a_run("M", "strong_positive", [True] * 4,
                                     conditioning_correct=[True] * 5))
            corpus.append(_exp1a_run("M", "no_conditioning", [True] * 4,
                                     conditioning_correct=[True, True, True, True, False]))
            corpus.append(_exp1a_run("M", "strong_negative", [True] * 4,
                                     conditioning_correct=[False, False, False, True, True]))

        verdicts = manipulation_check_from_corpus(corpus)
        assert "M" in verdicts
        # PASS or PARTIAL acceptable; FAIL is the bad case
        assert verdicts["M"].verdict.value in {"pass", "partial"}


class TestAnalyzeH4Corpus:
    def test_per_model_aggregates_and_base_vs_instruct(self):
        from src.analysis.h4 import analyze_h4_corpus

        # Two models: a base + an instruct. Each shows asymmetric effects:
        # negative effect larger than positive (so ratio_geomean > 1).
        # Conditioning_correct patterns are chosen so manipulation_check
        # PASSes (clear pos vs neg vs no-cond accuracy gap).
        models = {
            "Meta-Llama-3-8B": {
                "no_conditioning": (
                    [True, True, True, True, True],   # transfer 1.0
                    [True, True, True, True, False],  # cond 0.8
                ),
                "strong_positive": (
                    [True, True, True, True, False],  # transfer 0.8
                    [True, True, True, True, True],   # cond 1.0 (boost)
                ),
                "strong_negative": (
                    [True, False, False, False, False],  # transfer 0.2
                    [False, False, False, True, True],   # cond 0.4 (deficit)
                ),
            },
            "Meta-Llama-3-8B-Instruct": {
                "no_conditioning": (
                    [True, True, True, True, True],
                    [True, True, True, True, False],  # cond 0.8
                ),
                "strong_positive": (
                    [True, True, True, True, True],   # transfer 1.0
                    [True, True, True, True, True],   # cond 1.0
                ),
                "strong_negative": (
                    [False, False, False, False, False],  # transfer 0.0
                    [False, False, True, False, False],   # cond 0.2
                ),
            },
        }
        corpus: list[dict] = []
        for model, conds in models.items():
            for cond, (tc, cc) in conds.items():
                for _ in range(5):
                    corpus.append(_exp1a_run(model, cond, tc, conditioning_correct=cc))

        result = analyze_h4_corpus(
            corpus=corpus,
            base_model="Meta-Llama-3-8B",
            instruct_model="Meta-Llama-3-8B-Instruct",
        )
        assert "per_model_aggregates" in result
        assert "Meta-Llama-3-8B" in result["per_model_aggregates"]
        assert "Meta-Llama-3-8B-Instruct" in result["per_model_aggregates"]
        # base-vs-instruct contrast computed
        assert "asymmetry_delta_ratio" in result
        # Per-model verdicts assigned
        assert "per_model_verdicts" in result
        # delta_ratio > 1 since instruct shows more asymmetry than base
        if result["asymmetry_delta_ratio"] is not None:
            assert result["asymmetry_delta_ratio"] > 1.0

    def test_empty_corpus_returns_unavailable(self):
        from src.analysis.h4 import analyze_h4_corpus

        result = analyze_h4_corpus(
            corpus=[],
            base_model="A",
            instruct_model="B",
        )
        assert result["verdict"] == "unavailable_no_data"
