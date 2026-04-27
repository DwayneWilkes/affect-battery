"""Exp 3a runner with 7-level intensity axis.

Single-turn paradigm: each (level, run) cell delivers
INTENSITY_LEVELS[level-1].feedback_text as the system message and one
disjoint sample from the configured bank as the user message. The
pilot-seed SHA gate runs before any cell dispatches.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.runner import RunResult, Exp3aBody, ExperimentConfig, ExperimentType
from src.runners import run_exp3a
from src.conditioning.prompts import Condition


def _write_seed(tmp_path, sha=None):
    """Write a valid intensity-pilot seed file (helper for tests)."""
    from src.probes.intensity_pilot import emit_seed

    pilot_result = {
        "alpha_overall": 0.85,
        "alpha_pairwise": {"r1__r2": 0.84, "r1__r3": 0.82, "r2__r3": 0.88},
        "decision": "proceed",
    }
    seed_path = tmp_path / "intensity_pilot_pass_2026-04-24.json"
    emit_seed(
        pilot_result,
        axis_id="primary_valence_axis",
        n_levels=7,
        pilot_date="2026-04-24",
        output_path=seed_path,
    )
    return seed_path


def _write_bank(tmp_path: Path, n: int = 50) -> Path:
    bank = tmp_path / "bank.yaml"
    bank.write_text(yaml.safe_dump({"items": [
        {"id": f"item_{i:03d}", "question": f"What is {i}?", "expected": str(i)}
        for i in range(n)
    ]}))
    return bank


@pytest.mark.asyncio
async def test_exp3a_runs_at_seven_levels(tmp_path):
    """Runner yields one result per (level, run) cell."""
    from src.models import DryRunClient

    seed_path = _write_seed(tmp_path)
    bank = _write_bank(tmp_path, n=50)
    client = DryRunClient(model="dry-run", responses=["42"])

    intensity_levels = [1, 2, 3, 4, 5, 6, 7]
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
        num_runs=1,
        seed=42,
        transfer_bank=str(bank),
    )

    results: list[RunResult] = []
    async for r in run_exp3a(
        config, client,
        intensity_levels=intensity_levels,
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "results",
    ):
        results.append(r)

    assert len(results) == 7
    seen_levels = {r.body.intensity_level for r in results}
    assert seen_levels == {1, 2, 3, 4, 5, 6, 7}
    for r in results:
        assert r.experiment_type == "exp3a"
        assert isinstance(r.body, Exp3aBody)


@pytest.mark.asyncio
async def test_exp3a_rejects_tampered_seed(tmp_path):
    """If the pilot-seed SHA does not match the canonicalized payload,
    run_exp3a refuses to start."""
    from src.models import DryRunClient

    seed_path = _write_seed(tmp_path)
    bank = _write_bank(tmp_path, n=50)
    payload = json.loads(seed_path.read_text())
    payload["alpha_overall"] = 0.99
    seed_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    client = DryRunClient(model="dry-run", responses=["42"])
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
        num_runs=1,
        seed=42,
        transfer_bank=str(bank),
    )

    with pytest.raises(ValueError, match="SHA"):
        async for _ in run_exp3a(
            config, client,
            intensity_levels=[1, 2, 3, 4, 5, 6, 7],
            pilot_seed_path=seed_path,
            output_dir=tmp_path / "results",
        ):
            pass
