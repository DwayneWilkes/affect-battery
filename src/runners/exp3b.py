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
import time
from dataclasses import asdict
from pathlib import Path

from src.runner import (
    Exp3bBody,
    ExperimentType,
    RunResult,
    save_result,
    run_conditioning_phase,
)


async def run_exp3b(
    config,
    client,
    prompts: list[dict],
    n_generations: int = 10,
    output_dir: Path | None = None,
    **kwargs,
):
    """Run Exp 3b across `prompts` x `n_generations`."""
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

    base_seed = config.seed or 0
    for run_num in range(config.num_runs):
        # Phase 1: conditioning. Per (run_num) so the
        # conditioning history is consistent across all prompts in this run.
        seed = base_seed + run_num
        cond_messages, conditioning_responses, conditioning_correct = (
            await run_conditioning_phase(config, client, seed)
        )

        for p_idx, prompt in enumerate(prompts):
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
            tasks = [
                client.complete(generation_messages, temperature=config.temperature)
                for _ in range(n_generations)
            ]
            generations = list(await asyncio.gather(*tasks))

            result = RunResult(
                config=asdict(config),
                run_number=run_num,
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
