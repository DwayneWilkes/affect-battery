"""Exp 3b cognitive-scope metrics.

Per cognitive-scope-measurement spec "Metrics" + "Semantic diversity is
primary": two metrics over a set of open-ended generations.

1. Embedding variance: project each generation to a sentence embedding
   (default: sentence-transformers/all-mpnet-base-v2 v2.0.0 per spec),
   average them to a centroid (NOT unit-normalized — we use cosine
   distances to the centroid directly), compute mean cosine distance
   from each generation embedding to the centroid.

2. n-gram ratio: count unique n-grams across the generation set,
   divide by total n-gram occurrences. Range [0, 1]; 0 = all
   generations identical, 1 = no n-gram repeats anywhere.

The embedder is injectable so tests can use a fake. The default
embedder lazily loads the pinned model on first call.
"""

from __future__ import annotations

import math
from typing import Callable


# Pinned model identifier for the production embedder.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_VERSION = "v2.0.0"


_embedder_cache = None


def _default_embedder(texts: list[str]) -> list[list[float]]:
    """Lazy-load the pinned embedding model and embed `texts`.

    The model is cached at module level so repeated calls don't reload.
    Tests should pass a `fake_embedder` to avoid the heavy load.
    """
    global _embedder_cache
    if _embedder_cache is None:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        _embedder_cache = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = _embedder_cache.encode(texts, normalize_embeddings=True)
    return [list(e) for e in embeddings]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    cos_sim = dot / (na * nb)
    # Clamp for numerical stability before subtracting.
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim


def compute_embedding_variance(
    generations: list[str],
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> float:
    """Mean cosine distance from generation embeddings to their centroid."""
    if not generations:
        return 0.0
    embed_fn = embedder if embedder is not None else _default_embedder
    embeddings = embed_fn(list(generations))
    n = len(embeddings)
    if n == 0:
        return 0.0
    dim = len(embeddings[0])
    centroid = [sum(e[i] for e in embeddings) / n for i in range(dim)]
    distances = [_cosine_distance(e, centroid) for e in embeddings]
    return sum(distances) / len(distances)


def compute_ngram_ratio(generations: list[str], n: int) -> float:
    """Unique n-grams / total n-grams across the generation set."""
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if not generations:
        return 0.0
    all_ngrams: list[tuple[str, ...]] = []
    for gen in generations:
        tokens = gen.split()
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def analyze_exp3b_corpus(
    corpus: list[dict],
    model: str,
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
    ngram_n: int = 2,
) -> dict:
    """Cross-condition aggregation pipeline for Exp 3b.

    Group result-JSONs by condition, flatten generations across prompts
    within each condition, compute embedding_variance + ngram_ratio.
    Embedder is injectable for tests; production uses the pinned mpnet.
    """
    by_cond_gens: dict[str, list[str]] = {}
    for run in corpus:
        cond = run.get("condition")
        if cond is None:
            continue
        gens = (run.get("body") or {}).get("generations", [])
        by_cond_gens.setdefault(cond, []).extend(gens)

    by_condition: dict[str, dict] = {}
    for cond, gens in by_cond_gens.items():
        if not gens:
            continue
        by_condition[cond] = {
            "n_generations": len(gens),
            "embedding_variance": compute_embedding_variance(
                gens, embedder=embedder,
            ),
            "ngram_ratio": compute_ngram_ratio(gens, n=ngram_n),
        }

    return {
        "model": model,
        "verdict": "complete" if by_condition else "unavailable_no_data",
        "by_condition": by_condition,
    }
