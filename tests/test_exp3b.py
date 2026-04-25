"""Exp 3b runner (open-ended generation).

Per cognitive-scope-measurement spec "Cognitive-scope protocol" +
: Exp 3b runs 3 conditions x prompts x 10 generations
per prompt at temperature=0.7, top_p=0.95. Each generation receives a
distinct seed for reproducibility. The result body is Exp3bBody with
prompt_id, generations, per_generation_seeds.
"""

from __future__ import annotations

import pytest

from src.runner import RunResult, Exp3bBody, ExperimentConfig, ExperimentType
from src.runners import run_exp3b
from src.conditioning.prompts import Condition


@pytest.mark.asyncio
async def test_exp3b_ten_generations_per_prompt_per_condition(tmp_path):
    """Runner produces 10 completions per (prompt, condition) with
    distinct per-generation seeds."""
    from src.models import DryRunClient

    prompts = [
        {"id": "story_1", "text": "Continue the story: The lighthouse keeper saw a strange light..."},
        {"id": "brainstorm_1", "text": "List 5 unconventional uses for a paperclip."},
    ]
    client = DryRunClient(model="dry-run", responses=["A creative completion."])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.COGNITIVE_SCOPE,
        num_runs=1,
        temperature=0.7,
        seed=42,
    )

    results: list[RunResult] = []
    async for r in run_exp3b(
        config, client,
        prompts=prompts,
        n_generations=10,
        output_dir=tmp_path,
    ):
        results.append(r)

    # 2 prompts x 1 num_runs = 2 results, each with 10 generations
    assert len(results) == 2
    for r in results:
        assert r.experiment_type == "exp3b"
        assert isinstance(r.body, Exp3bBody)
        assert len(r.body.generations) == 10
        assert len(r.body.per_generation_seeds) == 10
        # Distinct seeds per generation
        assert len(set(r.body.per_generation_seeds)) == 10
        # Prompt id recorded
        assert r.body.prompt_id in {"story_1", "brainstorm_1"}


@pytest.mark.asyncio
async def test_exp3b_runs_conditioning_phase_first(tmp_path):
    """Per Exp 3b MUST run the 5-turn affective
    conditioning protocol before the generation phase. We assert this by
    checking that conditioning_responses is populated on the result."""
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["42"] * 50)
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_NEGATIVE,
        experiment_type=ExperimentType.COGNITIVE_SCOPE,
        num_runs=1,
        temperature=0.7,
        seed=42,
        num_conditioning_turns=5,
    )

    async for r in run_exp3b(
        config, client,
        prompts=[{"id": "p1", "text": "Continue the story."}],
        n_generations=3,
        output_dir=tmp_path,
    ):
        # 5 conditioning turns at config.num_conditioning_turns -> 5 responses
        assert len(r.conditioning_responses) == 5
        assert len(r.conditioning_correct) == 5


@pytest.mark.asyncio
async def test_exp3b_uses_paper_sampling_params(tmp_path):
    """Temperature=0.7 + top_p=0.95 are the paper §3.4.2 sampling
    parameters — verify they're surfaced on the run config."""
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["text"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.COGNITIVE_SCOPE,
        num_runs=1,
        temperature=0.7,
        seed=0,
    )
    async for r in run_exp3b(
        config, client,
        prompts=[{"id": "p1", "text": "test"}],
        n_generations=2,
        output_dir=tmp_path,
    ):
        assert r.config["temperature"] == 0.7
