"""Semantic and lexical diversity scoring for Experiment 3b.

Lexical diversity uses unique-n-gram / total-n-gram (type-token ratio,
TTR) for n in {2, 3, 4}. Note: TTR is mathematically length-dependent --
a repetitive text's ratio falls as length grows. For our experimental
scenario (many generations per condition, similar mean response length
across conditions), condition-level comparisons remain meaningful because
both conditions are at similar lengths. For studies where response length
differs substantially across conditions, consider a length-invariant
metric (MTLD, HDD, moving-average TTR) instead.

Semantic diversity is deferred: full implementation requires the
sentence-transformers optional dependency. The current placeholder logs
a warning and returns 0.0 so callers downstream do not crash.
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
    """Placeholder for embedding-based semantic diversity.

    The full implementation (cosine distance variance from centroid across
    generated texts) requires the sentence-transformers optional dependency
    and is deferred until after the pilot per spec task 6.3. The placeholder
    logs a warning on first call so the caller knows the metric is not
    actually computed, then returns 0.0 so downstream analysis does not
    crash.
    """
    global _SEMANTIC_DIVERSITY_WARNING_EMITTED
    if not _SEMANTIC_DIVERSITY_WARNING_EMITTED:
        log.warning(
            "semantic_diversity is deferred (sentence-transformers dependency "
            "not installed). Returning 0.0 placeholder; install sentence-"
            "transformers and implement embedding logic per spec task 6.3."
        )
        _SEMANTIC_DIVERSITY_WARNING_EMITTED = True
    else:
        log.warning("semantic_diversity placeholder called; returning 0.0.")
    return 0.0
