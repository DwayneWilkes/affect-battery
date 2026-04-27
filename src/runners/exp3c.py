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
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.runner import (
    Exp3cBody,
    ExperimentType,
    RunResult,
    _BudgetedClient,
    _TokenBucket,
    _cached_run_path,
    is_valid_cached_result,
    save_result,
    run_conditioning_phase,
)
from src.util import CHECKSUM_KEY


VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})

# Refusal heuristic prefixes; lifted out so future tuning is one place.
REFUSAL_PREFIXES = (
    "i cannot", "i can't", "i'm not able", "i don't",
    "i am not able", "i am unable", "i refuse",
)


def _is_refusal(response: str) -> bool:
    text = response.lower().strip()
    return any(text.startswith(prefix) for prefix in REFUSAL_PREFIXES)


def _yield_cached_cell(cell_path: Path) -> RunResult:
    data = json.loads(cell_path.read_text())
    payload = {k: v for k, v in data.items() if k != CHECKSUM_KEY}
    cached = RunResult(**payload)
    cached.checksum = data.get(CHECKSUM_KEY, "")
    return cached


async def run_exp3c(
    config,
    client,
    items: list[dict],
    output_dir: Path | None = None,
    budget=None,
    rate_limit_rps: float | None = None,
    cancel_event=None,
    **kwargs,
):
    """Run Exp 3c across `items` (each with difficulty/question/expected).

    Honors the same budget / rate-limit / cancel kwargs as run_batch.
    """
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

    rate_limiter = _TokenBucket(rate_limit_rps) if rate_limit_rps else None
    budgeted_client = _BudgetedClient(client, budget, rate_limiter)

    base_seed = config.seed or 0
    expected_transfer_hash = (
        getattr(config, "transfer_bank_hash", None) or None
    )
    for run_num in range(config.num_runs):
        if cancel_event is not None and cancel_event.is_set():
            break

        # Per-cell cache classification: every (run_num, item_idx) pair is
        # checked independently. Cached cells are yielded straight from
        # disk; only the missing cells trigger conditioning + API work.
        # Restarts are "free" in the sense that completed cells are never
        # redone, even if other cells from the same run_num are missing.
        cached_cell_paths: dict[int, Path] = {}
        missing_indices: list[int] = []
        if base_dir is not None:
            for item_idx in range(len(items)):
                composite_run_number = run_num * 10_000 + item_idx
                cell_path = _cached_run_path(base_dir, config, composite_run_number)
                if is_valid_cached_result(
                    cell_path,
                    expected_transfer_bank_hash=expected_transfer_hash,
                ):
                    cached_cell_paths[item_idx] = cell_path
                else:
                    missing_indices.append(item_idx)
        else:
            missing_indices = list(range(len(items)))

        # Yield cached cells first (no API calls). Each cell carries its
        # own conditioning_responses from when it was originally written,
        # so mixing cached + freshly-run cells in the same run_num is
        # safe — the analyzer reads each cell's own conditioning history.
        for item_idx in sorted(cached_cell_paths):
            yield _yield_cached_cell(cached_cell_paths[item_idx])

        if not missing_indices:
            continue

        if cancel_event is not None and cancel_event.is_set():
            break

        # Phase 1: conditioning (only when there is missing work to do).
        seed = base_seed + run_num
        cond_messages, conditioning_responses, conditioning_correct = (
            await run_conditioning_phase(config, budgeted_client, seed)
        )

        # Phase 2: per-item completions for ONLY the missing cells,
        # dispatched concurrently. Each item appends its question to a
        # copy of the conditioning history.
        async def _ask(item: dict) -> tuple[dict, str]:
            messages = cond_messages + [{"role": "user", "content": item["question"]}]
            # exp3c factual QA + hedging measurement: 256 tokens is
            # plenty for the longest legitimate hedge-rich answer.
            response = await budgeted_client.complete(
                messages, temperature=config.temperature, max_tokens=256,
            )
            return item, response

        missing_items = [items[i] for i in missing_indices]
        results_per_item = await asyncio.gather(*[_ask(it) for it in missing_items])

        for (item_idx, (item, response)) in zip(missing_indices, results_per_item):
            body = Exp3cBody(
                difficulty=item["difficulty"],
                question=item["question"],
                response=response,
                expected=item["expected"],
                stated_confidence=None,
                refused=_is_refusal(response),
                expected_aliases=list(item.get("answer_aliases") or []),
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
