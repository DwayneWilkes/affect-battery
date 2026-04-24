"""Task 1.2 Red — base-model feasibility probe.

Per base-model-comparison spec "Week-1 go/no-go gate for base-model
feasibility": runs Llama-3-8B base on N GSM8K problems with no
conditioning. Pass threshold: baseline_accuracy >= 0.30.
"""

import json
from pathlib import Path

import pytest

from src.probes.base_model import (
    BaseModelProbeResult,
    run_base_model_probe,
)


class TestBaseModelProbeResultSchema:
    def test_result_carries_required_fields(self):
        r = BaseModelProbeResult(
            model="llama-3-8b-base",
            baseline_accuracy=0.4,
            n_problems=5,
            feasibility_verdict="pass",
            threshold=0.30,
        )
        assert r.feasibility_verdict == "pass"
        assert r.baseline_accuracy == 0.4


class TestRunBaseModelProbe:
    @pytest.mark.asyncio
    async def test_dry_run_produces_result(self, tmp_path):
        from src.models import DryRunClient
        client = DryRunClient(model="llama-3-8b-base")
        result = await run_base_model_probe(
            client=client,
            model_name="llama-3-8b-base",
            n=5,
            output_dir=tmp_path,
        )
        assert isinstance(result, BaseModelProbeResult)
        assert result.feasibility_verdict in ("pass", "fail")
        assert result.threshold == 0.30

    @pytest.mark.asyncio
    async def test_writes_json_to_output_dir(self, tmp_path):
        from src.models import DryRunClient
        client = DryRunClient(model="llama-3-8b-base")
        await run_base_model_probe(
            client=client,
            model_name="llama-3-8b-base",
            n=5,
            output_dir=tmp_path,
        )
        json_files = list(tmp_path.glob("base_model_probe_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert "feasibility_verdict" in data
        assert "baseline_accuracy" in data
        assert "threshold" in data

    def test_threshold_default_matches_spec(self):
        """0.30 default per base-model-comparison spec Week-1 gate."""
        from src.probes.base_model import DEFAULT_FEASIBILITY_THRESHOLD
        assert DEFAULT_FEASIBILITY_THRESHOLD == 0.30
