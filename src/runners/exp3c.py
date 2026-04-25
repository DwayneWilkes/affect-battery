"""Exp 3c — conservative shift (paper §3.4.3).

Per conservative-shift-measurement spec "Conservative-shift protocol" +
"Question-difficulty stratification" each run
performs the 5-turn affective conditioning protocol BEFORE asking the
factual-QA items so the responses reflect the conditioning. Skipping
conditioning (the prior implementation did) made hedging / refusal /
length metrics meaningless under condition stratification.

Items within a single run are dispatched concurrently via
asyncio.gather ; they share the conditioning
history but each lands as a separate one-shot user turn from the
model's perspective (the conditioning_messages list is reused but
NOT mutated across items).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path

from src.runner import (
    Exp3cBody,
    ExperimentType,
    RunResult,
    save_result,
    run_conditioning_phase,
)


VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})

# Refusal heuristic prefixes; lifted out so future tuning is one place.
REFUSAL_PREFIXES = (
    "i cannot", "i can't", "i'm not able", "i don't",
    "i am not able", "i am unable", "i refuse",
)


def _is_refusal(response: str) -> bool:
    text = response.lower().strip()
    return any(text.startswith(prefix) for prefix in REFUSAL_PREFIXES)


async def run_exp3c(
    config,
    client,
    items: list[dict],
    output_dir: Path | None = None,
    **kwargs,
):
    """Run Exp 3c across `items` (each with difficulty/question/expected)."""
    if config.experiment_type != ExperimentType.CONSERVATIVE_SHIFT:
        raise ValueError(
            f"run_exp3c requires config.experiment_type=CONSERVATIVE_SHIFT; "
            f"got {config.experiment_type!r}"
        )
    if not items:
        raise ValueError("items must be non-empty")

    for item in items:
        if item.get("difficulty") not in VALID_DIFFICULTIES:
            raise ValueError(
                f"invalid difficulty {item.get('difficulty')!r}; "
                f"must be one of {sorted(VALID_DIFFICULTIES)}"
            )

    base_dir = Path(output_dir) if output_dir else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    base_seed = config.seed or 0
    for run_num in range(config.num_runs):
        # Phase 1: conditioning.
        seed = base_seed + run_num
        cond_messages, conditioning_responses, conditioning_correct = (
            await run_conditioning_phase(config, client, seed)
        )

        # Phase 2: per-item completions, dispatched concurrently. Each item
        # appends its question to a copy of the conditioning history.
        async def _ask(item: dict) -> tuple[dict, str]:
            messages = cond_messages + [{"role": "user", "content": item["question"]}]
            response = await client.complete(messages, temperature=config.temperature)
            return item, response

        results_per_item = await asyncio.gather(*[_ask(it) for it in items])

        for item_idx, (item, response) in enumerate(results_per_item):
            body = Exp3cBody(
                difficulty=item["difficulty"],
                question=item["question"],
                response=response,
                expected=item["expected"],
                stated_confidence=None,
                refused=_is_refusal(response),
            )
            result = RunResult(
                config=asdict(config),
                # Composite run_number so each (run_num, item) writes a
                # unique result file under save_result's naming scheme.
                run_number=run_num * 10_000 + item_idx,
                experiment_type=config.experiment_type.value,
                model=config.model_name,
                condition=config.condition.value,
                conditioning_responses=conditioning_responses,
                conditioning_correct=conditioning_correct,
                start_time=time.time(),
                end_time=time.time(),
                body=body,
            )
            result.compute_checksum()
            if base_dir is not None:
                save_result(result, base_dir)
            yield result
