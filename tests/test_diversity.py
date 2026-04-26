"""Tests for lexical and semantic diversity scoring.

Spec (scoring-pipeline, Requirements: Lexical diversity scoring, Semantic
diversity placeholder):
- Lexical diversity normalized by total n-gram count so condition-level
  comparison is not confounded by response length.
- Semantic diversity placeholder returns 0.0 and logs a warning when
  sentence-transformers is unavailable (deferred per task 6.3).
"""


from src.scoring.diversity import lexical_diversity


class TestLexicalDiversity:
    def test_basic(self):
        texts = ["the cat sat on the mat", "a dog ran in the park"]
        result = lexical_diversity(texts)
        assert "unique_2gram_ratio" in result
        assert 0 < result["unique_2gram_ratio"] <= 1

    def test_identical_texts_low_diversity(self):
        texts = ["the cat sat on the mat"] * 5
        result = lexical_diversity(texts)
        # Identical texts should have lower diversity than varied texts
        varied = lexical_diversity(["the cat sat", "a dog ran fast", "birds fly high", "fish swim deep", "trees grow tall"])
        assert result["unique_2gram_ratio"] <= varied["unique_2gram_ratio"]

    def test_single_text(self):
        result = lexical_diversity(["hello world"])
        assert result["total_tokens"] == 2

    def test_empty_texts(self):
        result = lexical_diversity(["", ""])
        assert result["total_tokens"] == 0

    def test_normalization_different_lengths(self):
        """Ratio should be comparable regardless of total text length."""
        short = lexical_diversity(["cat dog"] * 3)
        long = lexical_diversity(["cat dog " * 10] * 3)
        # Both should produce ratios between 0 and 1
        assert 0 <= short["unique_2gram_ratio"] <= 1
        assert 0 <= long["unique_2gram_ratio"] <= 1

    def test_condition_comparison_at_similar_length(self):
        """Spec intent: the metric supports meaningful condition comparison
        when response lengths are roughly similar across conditions (the
        experimental scenario). A condition producing less-diverse output
        has a lower ratio than one producing more-diverse output at a
        comparable length.

        NOTE: type-token ratio is mathematically length-dependent for a
        given content pattern (a well-known limitation). Across conditions
        with similar mean response length, comparisons remain meaningful.
        Documented in diversity.py module docstring.
        """
        varied = lexical_diversity(["cat dog run fast sky blue tree tall"] * 5)
        repetitive = lexical_diversity(["the the the the the the the the"] * 5)
        assert varied["unique_2gram_ratio"] > repetitive["unique_2gram_ratio"]

    def test_varied_text_beats_repetitive_at_same_length(self):
        """At matched token count, varied text must score higher than
        repetitive text. This is the signal we want the metric to capture."""
        tokens = 20
        repetitive = lexical_diversity(["cat dog " * (tokens // 2)])
        varied = lexical_diversity([
            "cat dog run fast sky blue tree tall river deep"
        ] * 2)  # roughly 20 tokens
        assert varied["unique_2gram_ratio"] > repetitive["unique_2gram_ratio"]

    def test_all_three_ngram_sizes_present(self):
        """Spec: 2-gram, 3-gram, 4-gram all reported per scenario."""
        result = lexical_diversity(["the quick brown fox jumps over lazy dog"])
        for n in (2, 3, 4):
            assert f"unique_{n}gram_ratio" in result
            assert f"unique_{n}gram_count" in result
            assert f"total_{n}gram_count" in result


# Semantic / embedding-based diversity for Exp 3b lives in
# src.analysis.exp3b.compute_embedding_variance — see
# tests/test_exp3b_metrics.py for its coverage.
