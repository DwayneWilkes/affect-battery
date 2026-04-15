"""Semantic and lexical diversity scoring for Experiment 3b."""

from collections import Counter


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def lexical_diversity(texts: list[str], n_gram_sizes: tuple[int, ...] = (2, 3, 4)) -> dict:
    """Compute length-normalized unique n-gram ratios across multiple texts.
    
    Higher values indicate more diverse/varied output.
    """
    results = {}
    for n in n_gram_sizes:
        all_ngrams = []
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


def semantic_diversity(texts: list[str]) -> float:
    """Placeholder for embedding-based semantic diversity.
    
    Requires sentence-transformers (optional dependency).
    Returns cosine distance variance from centroid across texts.
    
    TODO: Implement with sentence-transformers when embeddings dependency is added.
    """
    return 0.0
