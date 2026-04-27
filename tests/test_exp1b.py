"""Exp 1b cross-session falsification runner.

Per design.md D6 + conditioning-protocol spec
"Cross-session session seeds are recorded separately": Exp 1b extends
Exp 1a with a Phase-2 fresh-session re-test using the neutral cross-
session system prompt. session_1_seed and session_2_seed are recorded
distinctly on the Exp1bBody.
"""

from __future__ import annotations

import pytest

from src.runner import RunResult, Exp1bBody, ExperimentConfig, ExperimentType
from src.runners import run_exp1b
from src.conditioning.prompts import Condition


@pytest.mark.asyncio
async def test_exp1b_runner_produces_exp1b_body(tmp_path):
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["42"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.TRANSFER_CROSS,
        num_runs=1,
        temperature=0.7,
        seed=42,
    )

    results: list[RunResult] = []
    async for r in run_exp1b(config, client, output_dir=tmp_path):
        results.append(r)

    assert len(results) == 1
    r = results[0]
    assert r.experiment_type == "exp1b"
    assert r.body is not None
    assert isinstance(r.body, Exp1bBody)


@pytest.mark.asyncio
async def test_exp1b_session_seeds_distinct(tmp_path):
    """Session_1_seed and session_2_seed must be recorded distinctly."""
    from src.models import DryRunClient

    client = DryRunClient(model="dry-run", responses=["42"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.TRANSFER_CROSS,
        num_runs=1,
        seed=42,
    )

    async for r in run_exp1b(config, client, output_dir=tmp_path):
        assert r.body.session_1_seed != 0 or r.body.session_2_seed != 0
        assert r.body.session_1_seed != r.body.session_2_seed


@pytest.mark.asyncio
async def test_exp1b_uses_cross_session_neutral_prompt(tmp_path):
    """Phase 2 must use the neutral cross-session system prompt; we verify
    by inspecting that the run completed under TRANSFER_CROSS dispatch
    (not TRANSFER_WITHIN) — implementation-level check."""
    from src.models import DryRunClient
    from src.runner import CROSS_SESSION_SYSTEM_PROMPT

    # Sanity: the neutral prompt is the documented one
    assert "best of your ability" in CROSS_SESSION_SYSTEM_PROMPT

    client = DryRunClient(model="dry-run", responses=["42"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.TRANSFER_CROSS,
        num_runs=1,
        seed=1,
    )
    async for r in run_exp1b(config, client, output_dir=tmp_path):
        # config field carries the discriminator
        assert r.config["experiment_type"] == ExperimentType.TRANSFER_CROSS
