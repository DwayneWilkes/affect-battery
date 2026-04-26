"""Tier-2 results-layout fix:
- Result JSONs nest as `<output_dir>/<condition>/<run_NNNN>.json`, not flat
  underscore-encoded filenames.
- The leaf directory holds exactly one cell of (model, experiment, condition).
- Resume / cache-validation works across nesting depth (rglob).

Spec: affect-battery-proposal-realignment :: results-layout.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.conditioning.prompts import Condition
from src.models import ModelClient
from src.runner import (
    ExperimentConfig,
    ExperimentType,
    _cached_run_path,
    is_valid_cached_result,
    run_batch,
    run_single,
    save_result,
)


class _ScriptedClient(ModelClient):
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

    async def complete_text(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        raise NotImplementedError


def _cfg(condition: Condition, num_runs: int = 1) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="claude-haiku-4-5",
        condition=condition,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=num_runs,
        num_conditioning_turns=1,
        num_transfer_questions=1,
        seed=42,
    )


class TestNestedFilenames:
    def test_save_result_writes_under_condition_subdir(self, tmp_path):
        """save_result MUST place the JSON at
        <output_dir>/<condition>/<run_NNNN>.json — not flat."""
        cfg = _cfg(Condition.STRONG_NEGATIVE)
        client = _ScriptedClient(responses=["7", "Some answer."])
        result = asyncio.run(run_single(cfg, client, 3))

        path = save_result(result, tmp_path)

        # Path components: <tmp>/<condition>/<NNNN>.json
        assert path.parent.name == "strong_negative", (
            f"expected condition subdir, got {path.parent}"
        )
        assert path.name == "0003.json", f"expected run-numbered filename, got {path.name}"
        # No model/condition/exp prefix in the filename.
        assert "claude-haiku" not in path.name
        assert "exp1a" not in path.name
        assert "strong_negative" not in path.name

    def test_cached_run_path_matches_save_result_layout(self, tmp_path):
        """_cached_run_path must return the same path save_result would
        write to — otherwise the resume layer never finds cached files."""
        cfg = _cfg(Condition.NEUTRAL)
        client = _ScriptedClient(responses=["12", "Answer."])
        result = asyncio.run(run_single(cfg, client, 7))

        written = save_result(result, tmp_path)
        cached = _cached_run_path(tmp_path, cfg, 7)
        assert written == cached

    def test_is_valid_cached_result_works_under_nested_path(self, tmp_path):
        cfg = _cfg(Condition.MILD_NEGATIVE)
        client = _ScriptedClient(responses=["3", "Answer."])
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        assert is_valid_cached_result(path) is True

    def test_two_conditions_land_in_separate_subdirs(self, tmp_path):
        for cond in (Condition.STRONG_NEGATIVE, Condition.STRONG_POSITIVE):
            cfg = _cfg(cond)
            client = _ScriptedClient(responses=["5", "Answer."])
            result = asyncio.run(run_single(cfg, client, 0))
            save_result(result, tmp_path)

        children = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert children == ["strong_negative", "strong_positive"]

    def test_run_batch_writes_into_condition_subdir_and_resumes(self, tmp_path):
        cfg = _cfg(Condition.STRONG_NEGATIVE, num_runs=2)

        async def go(client):
            out = []
            async for r in run_batch(cfg, client, output_dir=tmp_path, max_concurrent=1):
                out.append(r)
            return out

        # First pass: write 2 results.
        client_a = _ScriptedClient(responses=["1", "A.", "2", "B."])
        results_a = asyncio.run(go(client_a))
        assert len(results_a) == 2
        # Both in the strong_negative/ subdir.
        cond_dir = tmp_path / "strong_negative"
        assert cond_dir.is_dir()
        files = sorted(cond_dir.glob("*.json"))
        assert [p.name for p in files] == ["0000.json", "0001.json"]

        # Second pass: same config, fresh client. Should hit cache, not call API.
        client_b = _ScriptedClient(responses=["NEVER", "CALLED"])
        results_b = asyncio.run(go(client_b))
        assert len(results_b) == 2
        assert client_b._i == 0, "expected zero API calls on resume"
