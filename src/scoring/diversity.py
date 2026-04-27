"""Lexical diversity scoring.

Type-token ratio over n-grams (n in {2,3,4}). TTR is length-dependent;
cross-condition comparisons are valid when conditions have similar mean
response length, otherwise use MTLD/HDD for very different lengths.

Semantic / embedding-based diversity for Exp 3b is computed by
`src.analysis.exp3b.compute_embedding_variance`, which uses
sentence-transformers and is the canonical entry point for the
cognitive-scope analysis.
"""

from __future__ import annotations


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def lexical_diversity(
    texts: list[str],
    n_gram_sizes: tuple[int, ...] = (2, 3, 4),
) -> dict:
    """Compute length-normalized unique n-gram ratios across multiple texts.

    Higher values indicate more diverse / varied output.
    """
    results: dict = {}
    total_tokens = 0
    for n in n_gram_sizes:
        all_ngrams: list[tuple] = []
        total_tokens = 0
        for text in texts:
            tokens = text.lower().split()
            total_tokens += len(tokens)
            all_ngrams.extend(_ngrams(tokens, n))

        unique = len(set(all_ngrams))
        total = max(len(all_ngrams), 1)
        results[f"unique_{n}gram_ratio"] = unique / total
        results[f"unique_{n}gram_count"] = unique
        results[f"total_{n}gram_count"] = total

    results["total_tokens"] = total_tokens
    return results
