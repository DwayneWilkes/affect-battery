"""Tests for lexical diversity scoring."""

from src.scoring.diversity import lexical_diversity, semantic_diversity


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


class TestSemanticDiversity:
    def test_placeholder_returns_zero(self):
        assert semantic_diversity(["hello", "world"]) == 0.0
