"""Exp 1a runner end-to-end produces populated Exp1aBody.

Per invoking run_exp1a yields
RunResult with experiment_type='exp1a', body=Exp1aBody (not None),
populated conditioning + transfer fields. Schema valid; checksum
computed.
"""

import pytest

from src.runner import RunResult, Exp1aBody, ExperimentConfig, ExperimentType
from src.runners import run_exp1a
from src.conditioning.prompts import Condition


@pytest.mark.asyncio
async def test_exp1a_runner_produces_exp1a_body(tmp_path):
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["The answer is 42."])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=2,
        temperature=0.7,
        seed=42,
    )

    results: list[RunResult] = []
    async for r in run_exp1a(config, client, output_dir=tmp_path):
        results.append(r)

    assert len(results) == 2
    for r in results:
        assert r.experiment_type == "exp1a"
        assert r.body is not None
        assert isinstance(r.body, Exp1aBody)
        # Body fields populated from conditioning + transfer phases
        assert isinstance(r.body.conditioning_responses, list)
        assert isinstance(r.body.transfer_responses, list)


@pytest.mark.asyncio
async def test_exp1a_runner_writes_results_with_checksum(tmp_path):
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["42"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=1,
        seed=1,
    )
    async for r in run_exp1a(config, client, output_dir=tmp_path):
        assert r.checksum != ""
        # Checksum is a 16-char hex truncation per src.util.checksum_of_payload
        assert len(r.checksum) == 16


@pytest.mark.asyncio
async def test_exp1a_supports_full_paper_3_2_1_condition_set(tmp_path):
    """All 6 paper §3.2.1 arms accepted by the runner (one run each
    at small n)."""
    from src.models import DryRunClient

    six_arms = [
        Condition.STRONG_POSITIVE,
        Condition.MILD_NEGATIVE,
        Condition.STRONG_NEGATIVE,
        Condition.NEUTRAL,
        Condition.NO_CONDITIONING,
        Condition.ACCURATE_NEGATIVE,
    ]
    client = DryRunClient(model="dry-run", responses=["42"])
    for cond in six_arms:
        config = ExperimentConfig(
            model_name="dry-run",
            condition=cond,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            seed=0,
        )
        results = []
        async for r in run_exp1a(config, client, output_dir=tmp_path):
            results.append(r)
        assert len(results) == 1
        assert results[0].experiment_type == "exp1a"
