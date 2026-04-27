"""Exp 3b — cognitive scope (paper §3.4.2).

Per cognitive-scope-measurement spec "Cognitive-scope protocol" +
each (run_num, prompt) pair runs the 5-turn affective
conditioning protocol FIRST, then samples n_generations completions
from the prompt with the conditioning history attached. Without the
conditioning phase the generations are unconditioned, which would make
every condition-stratified diversity metric meaningless.

Generations within a single (run_num, prompt) are independent and run
concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.runner import (
    Exp3bBody,
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


def _yield_cached_cell(cell_path: Path) -> RunResult:
    data = json.loads(cell_path.read_text())
    payload = {k: v for k, v in data.items() if k != CHECKSUM_KEY}
    cached = RunResult(**payload)
    cached.checksum = data.get(CHECKSUM_KEY, "")
    return cached


async def run_exp3b(
    config,
    client,
    prompts: list[dict],
    n_generations: int = 10,
    output_dir: Path | None = None,
    budget=None,
    rate_limit_rps: float | None = None,
    cancel_event=None,
    **kwargs,
):
    """Run Exp 3b across `prompts` x `n_generations`.

    Honors the same budget / rate-limit / cancel kwargs as run_batch:
    budget is a BatchBudget, rate_limit_rps throttles via a token
    bucket, cancel_event (asyncio.Event) drains the run gracefully on
    SIGINT.
    """
    if config.experiment_type != ExperimentType.COGNITIVE_SCOPE:
        raise ValueError(
            f"run_exp3b requires config.experiment_type=COGNITIVE_SCOPE; "
            f"got {config.experiment_type!r}"
        )
    if not prompts:
        raise ValueError("prompts must be non-empty")
    if n_generations <= 0:
        raise ValueError(f"n_generations must be > 0; got {n_generations}")

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

        # Per-cell cache classification: each (run_num, prompt_idx) pair
        # is checked independently. Cached cells yield straight from
        # disk; only missing cells trigger conditioning + API work.
        cached_cell_paths: dict[int, Path] = {}
        missing_indices: list[int] = []
        if base_dir is not None:
            for p_idx in range(len(prompts)):
                composite_run_number = run_num * 10_000 + p_idx
                cell_path = _cached_run_path(base_dir, config, composite_run_number)
                if is_valid_cached_result(
                    cell_path,
                    expected_transfer_bank_hash=expected_transfer_hash,
                ):
                    cached_cell_paths[p_idx] = cell_path
                else:
                    missing_indices.append(p_idx)
        else:
            missing_indices = list(range(len(prompts)))

        for p_idx in sorted(cached_cell_paths):
            yield _yield_cached_cell(cached_cell_paths[p_idx])

        if not missing_indices:
            continue

        if cancel_event is not None and cancel_event.is_set():
            break

        # Phase 1: conditioning. Per (run_num) so the
        # conditioning history is consistent across all prompts in this run.
        seed = base_seed + run_num
        cond_messages, conditioning_responses, conditioning_correct = (
            await run_conditioning_phase(config, budgeted_client, seed)
        )

        for p_idx in missing_indices:
            if cancel_event is not None and cancel_event.is_set():
                break
            prompt = prompts[p_idx]
            # Phase 2: n_generations independent completions following the
            # conditioning history. Run concurrently via asyncio.gather
            # . Append the prompt as the next user turn.
            generation_messages = cond_messages + [
                {"role": "user", "content": prompt["text"]}
            ]
            seeds = [
                base_seed + run_num * 100_000 + p_idx * 1_000 + g_idx
                for g_idx in range(n_generations)
            ]
            # exp3b open-ended generation: 512 tokens accommodates the
            # longest creative response without runaway. The semantic-
            # diversity metric needs the full response, but anything
            # beyond ~500 tokens is almost certainly the model going
            # off the rails (we've never seen legitimate generations
            # longer than ~400 tokens in pilots).
            tasks = [
                budgeted_client.complete(
                    generation_messages,
                    temperature=config.temperature,
                    max_tokens=512,
                )
                for _ in range(n_generations)
            ]
            generations = list(await asyncio.gather(*tasks))

            result = RunResult(
                config=asdict(config),
                # Composite run_number so each (run_num, prompt) writes a
                # unique result file under save_result's naming scheme.
                run_number=run_num * 10_000 + p_idx,
                experiment_type=config.experiment_type.value,
                model=config.model_name,
                condition=config.condition.value,
                conditioning_responses=conditioning_responses,
                conditioning_correct=conditioning_correct,
                start_time=time.time(),
                end_time=time.time(),
                body=Exp3bBody(
                    prompt_id=prompt["id"],
                    generations=generations,
                    per_generation_seeds=seeds,
                ),
            )
            result.compute_checksum()
            if base_dir is not None:
                save_result(result, base_dir)
            yield result
