"""Exp 3c cross-(condition, difficulty) aggregation pipeline (A3).

Aggregator over Exp 3c result-JSONs. Per (condition, difficulty) cell:
- n_items: count of result rows
- hedging_rate_per_100w: total primary hedges / total words * 100
- refusal_rate: fraction of rows with body.refused=True
- mean_response_length: average word count

Produces the dict shape that render_exp3c_report expects.
"""

from __future__ import annotations

from src.scoring.hedging import hedge_summary


def _word_count(text: str) -> int:
    return len(text.split())


def analyze_exp3c_corpus(corpus: list[dict], model: str) -> dict:
    """Cross-(condition, difficulty) aggregation for Exp 3c."""
    cells: dict[tuple[str, str], dict] = {}
    by_key: dict[tuple[str, str], list[dict]] = {}
    for run in corpus:
        cond = run.get("condition")
        body = run.get("body") or {}
        diff = body.get("difficulty")
        if cond is None or diff is None:
            continue
        by_key.setdefault((cond, diff), []).append(run)

    for key, rows in by_key.items():
        total_words = 0
        total_primary_hedges = 0
        refusal_count = 0
        lengths: list[int] = []
        for run in rows:
            response = (run.get("body") or {}).get("response", "")
            if (run.get("body") or {}).get("refused"):
                refusal_count += 1
            wc = _word_count(response)
            lengths.append(wc)
            total_words += wc
            summary = hedge_summary(response)
            total_primary_hedges += summary["total_primary"]

        cells[key] = {
            "n_items": len(rows),
            "hedging_rate_per_100w": (
                100.0 * total_primary_hedges / total_words
                if total_words > 0 else 0.0
            ),
            "refusal_rate": refusal_count / len(rows),
            "mean_response_length": (
                sum(lengths) / len(lengths) if lengths else 0.0
            ),
        }

    return {
        "model": model,
        "verdict": "complete" if cells else "unavailable_no_data",
        "by_condition_difficulty": cells,
    }
