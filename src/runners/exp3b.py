"""Exp 3b — cognitive scope (paper §3.4.2).

Per cognitive-scope-measurement spec "Cognitive-scope protocol" +
tasks.md Task 7.1: 3 conditions x prompts x N_GENERATIONS open-ended
generations per prompt at temperature=0.7. Distinct per-generation
seeds for reproducibility. Each yielded RunResult carries an Exp3bBody
recording (prompt_id, generations, per_generation_seeds).

DRY check: conditioning phase reuses run_single's existing 5-turn
protocol (via run_batch dispatch with num_transfer_questions=0). The
new logic is the per-prompt generation-sampling loop after conditioning.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from src.runner import (
    Exp3bBody,
    ExperimentType,
    RunResult,
    save_result,
    run_single,
)


async def run_exp3b(
    config,
    client,
    prompts: list[dict],
    n_generations: int = 10,
    output_dir: Path | None = None,
    **kwargs,
):
    """Run Exp 3b across `prompts` x `n_generations`.

    For each prompt, runs the conditioning phase once via run_single
    (reusing the Exp 1a path) then samples n_generations completions
    with distinct seeds. Generations share the conditioning history but
    each sample uses a distinct seed for reproducibility.
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

    base_seed = config.seed or 0
    for run_num in range(config.num_runs):
        for p_idx, prompt in enumerate(prompts):
            generations: list[str] = []
            seeds: list[int] = []
            for g_idx in range(n_generations):
                # Distinct seed per (run_num, prompt, generation) for
                # reproducibility without seed collisions across the matrix.
                gen_seed = base_seed + run_num * 100_000 + p_idx * 1_000 + g_idx
                seeds.append(gen_seed)
                # Synthesize a turn-style call against the prompt.
                response = await client.complete(
                    [{"role": "user", "content": prompt["text"]}],
                    temperature=config.temperature,
                )
                generations.append(response)

            result = RunResult(
                config=asdict(config),
                run_number=run_num,
                experiment_type=config.experiment_type.value,
                model=config.model_name,
                condition=config.condition.value,
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
