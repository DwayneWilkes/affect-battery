"""Task 1.1 Red — variance probe runner.

Per power-analysis spec "Week-1 pilot feeds a simulation-based power
analysis" + design.md D3 variance-probe-override:

The variance probe runs a small-n manipulation-check on a single
instruct model + 2 conditions (strong-positive vs neutral) on
arithmetic. Output JSON has fields: variance_estimate, std_err,
observed_effect_size, n_per_condition + the cross-cutting metadata
that a power-analysis update step (Task 1.3) consumes.
"""

import json
from pathlib import Path

import pytest

from src.probes.variance import (
    VarianceProbeResult,
    run_variance_probe,
)


class TestVarianceProbeResultSchema:
    def test_result_has_variance_estimate_field(self):
        r = VarianceProbeResult(
            model="llama-3-8b-instruct",
            variance_estimate=0.15,
            std_err=0.04,
            observed_effect_size=0.32,
            n_per_condition=20,
            conditions=["STRONG_POSITIVE", "NEUTRAL"],
        )
        assert r.variance_estimate == 0.15

    def test_result_round_trips_via_asdict(self):
        from dataclasses import asdict
        r = VarianceProbeResult(
            model="m",
            variance_estimate=0.10,
            std_err=0.03,
            observed_effect_size=0.25,
            n_per_condition=20,
            conditions=["STRONG_POSITIVE", "NEUTRAL"],
        )
        d = asdict(r)
        assert d["variance_estimate"] == 0.10
        assert d["n_per_condition"] == 20


class TestRunVarianceProbe:
    @pytest.mark.asyncio
    async def test_dry_run_produces_result(self, tmp_path):
        from src.models import DryRunClient
        client = DryRunClient(model="llama-3-8b-instruct")
        result = await run_variance_probe(
            client=client,
            model_name="llama-3-8b-instruct",
            n_per_condition=5,
            output_dir=tmp_path,
        )
        assert isinstance(result, VarianceProbeResult)
        assert result.n_per_condition == 5
        assert result.variance_estimate >= 0.0

    @pytest.mark.asyncio
    async def test_writes_json_to_output_dir(self, tmp_path):
        from src.models import DryRunClient
        client = DryRunClient(model="llama-3-8b-instruct")
        await run_variance_probe(
            client=client,
            model_name="llama-3-8b-instruct",
            n_per_condition=5,
            output_dir=tmp_path,
        )
        json_files = list(tmp_path.glob("variance_probe_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "variance_estimate" in data
        assert "observed_effect_size" in data
        assert "n_per_condition" in data
        assert data["n_per_condition"] == 5

    @pytest.mark.asyncio
    async def test_uses_strong_positive_and_neutral(self, tmp_path):
        from src.models import DryRunClient
        client = DryRunClient(model="llama-3-8b-instruct")
        result = await run_variance_probe(
            client=client,
            model_name="llama-3-8b-instruct",
            n_per_condition=3,
            output_dir=tmp_path,
        )
        # Condition enum values are lowercase
        assert "strong_positive" in result.conditions
        assert "neutral" in result.conditions
