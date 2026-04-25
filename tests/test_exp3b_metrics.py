"""Task 7.2 Red — Exp 3b metrics (embedding variance + n-gram ratio).

Per cognitive-scope-measurement spec "Metrics" + "Semantic diversity is
primary": compute_embedding_variance averages cosine distances from each
generation's embedding to the unit-normalized centroid; compute_ngram_ratio
counts unique n-grams / total n-grams across the generation set.
"""

from __future__ import annotations


class TestEmbeddingVariance:
    def test_constant_text_has_zero_variance(self):
        from src.analysis.exp3b import compute_embedding_variance

        # When generations are identical, variance is 0.
        gens = ["The same sentence."] * 5
        variance = compute_embedding_variance(gens, embedder=_fake_embedder)
        assert abs(variance) < 1e-6

    def test_diverse_text_has_positive_variance(self):
        from src.analysis.exp3b import compute_embedding_variance

        gens = [
            "The cat sat on the mat.",
            "Quantum mechanics describes subatomic behavior.",
            "Bananas are yellow.",
        ]
        variance = compute_embedding_variance(gens, embedder=_fake_embedder)
        assert variance > 0


class TestNGramRatio:
    def test_unique_ngram_ratio_zero_to_one(self):
        from src.analysis.exp3b import compute_ngram_ratio

        # All identical -> ratio close to 1/N (only one unique sequence)
        gens = ["a b c d e"] * 5
        ratio = compute_ngram_ratio(gens, n=2)
        # 4 bigrams x 5 generations = 20 total bigrams; 4 unique.
        # ratio = 4 / 20 = 0.2
        assert ratio == 0.2

    def test_all_distinct_ngrams_ratio_is_one(self):
        from src.analysis.exp3b import compute_ngram_ratio

        gens = [
            "a b c",
            "d e f",
            "g h i",
        ]
        ratio = compute_ngram_ratio(gens, n=2)
        # 2 bigrams per generation x 3 = 6 total; 6 unique => ratio = 1.0
        assert ratio == 1.0

    def test_invalid_n_raises(self):
        import pytest
        from src.analysis.exp3b import compute_ngram_ratio
        with pytest.raises(ValueError, match="n must be >= 1"):
            compute_ngram_ratio(["one two"], n=0)


# ---------------------------------------------------------------------------
# Tiny fake embedder for unit tests (avoids loading sentence-transformers).
# Maps each unique character set to a vector via hashing; identical inputs
# get identical embeddings; different inputs get different embeddings.
# ---------------------------------------------------------------------------
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
