"""Exp 2 persistence-dynamics runner.

Per persistence-dynamics spec "Persistence protocol structure" + "N-values
sweep": Exp 2 runs Phase 1 (5-turn conditioning on 3-arm subset) + Phase 2
(N neutral turns on diverse tasks) for N ∈ {1,3,5,10}. Each run produces
Exp2Body with n_value (the sweep step) + turn_accuracies (per-turn
accuracy across the N neutral turns).
"""

from __future__ import annotations

import pytest

from src.runner import RunResult, Exp2Body, ExperimentConfig, ExperimentType
from src.runners import run_exp2
from src.conditioning.prompts import Condition


@pytest.mark.asyncio
async def test_exp2_runner_produces_exp2_body(tmp_path):
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["42"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_NEGATIVE,
        experiment_type=ExperimentType.PERSISTENCE,
        num_runs=1,
        temperature=0.7,
        seed=42,
        neutral_turns=3,  # N value
    )

    results: list[RunResult] = []
    async for r in run_exp2(config, client, output_dir=tmp_path):
        results.append(r)

    assert len(results) == 1
    r = results[0]
    assert r.experiment_type == "exp2"
    assert r.body is not None
    assert isinstance(r.body, Exp2Body)
    assert r.body.n_value == 3
    # turn_accuracies has one entry per neutral turn
    assert len(r.body.turn_accuracies) == 3


@pytest.mark.asyncio
async def test_exp2_n_values_sweep(tmp_path):
    """Run Exp 2 across N ∈ {1, 3, 5, 10}; assert one result per N value
    with turn_accuracies length matching the N value."""
    from src.models import DryRunClient

    n_values = [1, 3, 5, 10]
    client = DryRunClient(model="dry-run", responses=["42"])

    for n in n_values:
        config = ExperimentConfig(
            model_name="dry-run",
            condition=Condition.STRONG_NEGATIVE,
            experiment_type=ExperimentType.PERSISTENCE,
            num_runs=1,
            seed=0,
            neutral_turns=n,
        )
        results = []
        n_dir = tmp_path / f"n_{n}"
        n_dir.mkdir()
        async for r in run_exp2(config, client, output_dir=n_dir):
            results.append(r)
        assert len(results) == 1
        assert results[0].body.n_value == n
        assert len(results[0].body.turn_accuracies) == n
