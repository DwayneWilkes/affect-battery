"""Lexical and semantic diversity scoring for Experiment 3b.

Lexical diversity is type-token ratio over n-grams (n in {2,3,4}). TTR is
length-dependent; cross-condition comparisons are valid when conditions
have similar mean response length. Use MTLD/HDD for very different lengths.

Semantic diversity is a deferred placeholder requiring sentence-transformers.
"""

import logging
from collections import Counter

log = logging.getLogger(__name__)


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


_SEMANTIC_DIVERSITY_WARNING_EMITTED = False


def semantic_diversity(texts: list[str]) -> float:
    """Deferred placeholder: returns 0.0 and warns. Full implementation
    (cosine distance variance from centroid) needs sentence-transformers."""
    global _SEMANTIC_DIVERSITY_WARNING_EMITTED
    if not _SEMANTIC_DIVERSITY_WARNING_EMITTED:
        log.warning(
            "semantic_diversity is deferred (sentence-transformers not installed). "
            "Returning 0.0 placeholder."
        )
        _SEMANTIC_DIVERSITY_WARNING_EMITTED = True
    else:
        log.warning("semantic_diversity placeholder called; returning 0.0.")
    return 0.0
