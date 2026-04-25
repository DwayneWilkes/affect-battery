"""Task 6.3 Red — Exp 3a runner with 7-level intensity axis.

Per conditioning-protocol spec "Intensity-axis pilot-as-gate for Exp 3a"
+ tasks.md Task 6.3: Exp 3a runner reads pilot_seed JSON (from Task 6.2),
validates SHA matches, then iterates intensity levels x stimulus bank
x model. Each run record carries the level on Exp3aBody.
"""

from __future__ import annotations

import json
import pytest

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


@pytest.mark.asyncio
async def test_exp3a_runs_at_seven_levels(tmp_path):
    """Runner produces one result per intensity level for the configured
    bank x model combo."""
    from src.models import DryRunClient

    seed_path = _write_seed(tmp_path)
    client = DryRunClient(model="dry-run", responses=["42"])

    intensity_levels = [1, 2, 3, 4, 5, 6, 7]
    config = ExperimentConfig(
        model_name="dry-run",
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
        num_runs=1,
        seed=42,
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
    # Tamper: edit one field without recomputing SHA
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
    )

    with pytest.raises(ValueError, match="SHA"):
        async for _ in run_exp3a(
            config, client,
            intensity_levels=[1, 2, 3, 4, 5, 6, 7],
            pilot_seed_path=seed_path,
            output_dir=tmp_path / "results",
        ):
            pass
