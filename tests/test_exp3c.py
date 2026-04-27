"""Exp 3c runner (factual QA, difficulty-stratified).

Per conservative-shift-measurement spec "Conservative-shift protocol" +
"Question-difficulty stratification": Exp 3c runs factual QA across
{easy, medium, hard} difficulty bins. Each result records the
difficulty + question + response + (optional) stated_confidence + refused
flag on Exp3cBody.
"""

from __future__ import annotations

import pytest

from src.runner import RunResult, Exp3cBody, ExperimentConfig, ExperimentType
from src.runners import run_exp3c
from src.conditioning.prompts import Condition


@pytest.mark.asyncio
async def test_exp3c_difficulty_stratified(tmp_path):
    """Runner produces results stratified by difficulty per model."""
    from src.models import DryRunClient

    items = [
        {"difficulty": "easy", "question": "What is 2+2?", "expected": "4"},
        {"difficulty": "easy", "question": "What is the capital of France?", "expected": "Paris"},
        {"difficulty": "medium", "question": "Who wrote Hamlet?", "expected": "Shakespeare"},
        {"difficulty": "hard", "question": "What is the speed of light in m/s?", "expected": "299792458"},
    ]
    client = DryRunClient(model="dry-run", responses=["I am not sure."])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_NEGATIVE,
        experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
        num_runs=1,
        seed=42,
    )

    results: list[RunResult] = []
    async for r in run_exp3c(
        config, client,
        items=items,
        output_dir=tmp_path,
    ):
        results.append(r)

    # One result per item
    assert len(results) == len(items)
    # All difficulties present
    difficulties = {r.body.difficulty for r in results}
    assert difficulties == {"easy", "medium", "hard"}
    for r in results:
        assert r.experiment_type == "exp3c"
        assert isinstance(r.body, Exp3cBody)
        assert r.body.question
        assert r.body.expected
        assert r.body.response


@pytest.mark.asyncio
async def test_exp3c_runs_conditioning_phase_first(tmp_path):
    """Per Exp 3c MUST run the 5-turn affective
    conditioning protocol before the QA phase. We assert this by checking
    that conditioning_responses is populated on the result."""
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["42"] * 50)
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_NEGATIVE,
        experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
        num_runs=1,
        seed=42,
        num_conditioning_turns=5,
    )
    items = [
        {"difficulty": "easy", "question": "Q1?", "expected": "A1"},
        {"difficulty": "medium", "question": "Q2?", "expected": "A2"},
    ]
    async for r in run_exp3c(config, client, items=items, output_dir=tmp_path):
        assert len(r.conditioning_responses) == 5
        assert len(r.conditioning_correct) == 5


@pytest.mark.asyncio
async def test_exp3c_invalid_difficulty_rejected(tmp_path):
    from src.models import DryRunClient

    bad_items = [
        {"difficulty": "trivial", "question": "?", "expected": "?"},
    ]
    client = DryRunClient(model="dry-run", responses=["x"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
        num_runs=1,
        seed=0,
    )
    with pytest.raises(ValueError, match="difficulty"):
        async for _ in run_exp3c(
            config, client, items=bad_items, output_dir=tmp_path,
        ):
            pass
