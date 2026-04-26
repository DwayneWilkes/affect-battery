"""exp3b and exp3c runners must skip API calls when results are cached.

Pre-fix the exp3b/exp3c runners had no cache layer at all — every
re-run made fresh API calls for every (run_num, axis_idx) cell, even
when valid result files already existed on disk. The user discovered
this during a real-API re-run where exp1a/1b cache-hit cleanly but
exp3b/exp3c made hundreds of unnecessary API calls.

This test pins the contract: when every cell for a given run_num is
already cached (transfer_bank_hash matches, schema valid, checksum
verifies), the runner skips conditioning + the per-cell API work and
yields the cached results from disk.

Spec: affect-battery-proposal-realignment :: experiment-dispatch.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.conditioning.prompts import Condition
from src.models import ModelClient
from src.runner import ExperimentConfig, ExperimentType


class _CountingClient(ModelClient):
    """Records how many times complete() was called so tests can
    assert "API was/wasn't hit"."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0
        self.complete_calls = 0

    @property
    def model_name(self) -> str:
        return "counting-client"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        self.complete_calls += 1
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

    async def complete_text(self, *args, **kwargs):
        raise NotImplementedError


class TestExp3cCache:
    @pytest.mark.asyncio
    async def test_exp3c_skips_api_when_all_items_cached(self, tmp_path):
        """First run writes fresh; second run with same config sees
        all items cached and makes ZERO API calls."""
        from src.runners.exp3c import run_exp3c

        cfg = ExperimentConfig(
            model_name="counting-client",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
            num_runs=1,
            num_conditioning_turns=2,
            seed=0,
            transfer_bank="",
            transfer_bank_hash="",
        )
        items = [
            {"difficulty": "easy", "question": "Q1?", "expected": "A1"},
            {"difficulty": "hard", "question": "Q2?", "expected": "A2"},
        ]

        # First run — populates cache.
        c1 = _CountingClient(responses=["10", "5", "answer1", "answer2"])
        out_dir = tmp_path / "exp3c_cache_test"
        results: list = []
        async for r in run_exp3c(
            cfg, c1, items=items, output_dir=out_dir,
        ):
            results.append(r)
        assert len(results) == 2  # 1 run × 2 items
        first_call_count = c1.complete_calls
        assert first_call_count > 0, "first run must call API"

        # Second run — same config, fresh client. Must hit cache.
        c2 = _CountingClient(responses=["NEVER_USED", "NEVER_USED2"])
        results2: list = []
        async for r in run_exp3c(
            cfg, c2, items=items, output_dir=out_dir,
        ):
            results2.append(r)
        assert len(results2) == 2
        assert c2.complete_calls == 0, (
            f"second run must hit cache; got {c2.complete_calls} API calls"
        )

    @pytest.mark.asyncio
    async def test_exp3c_re_runs_when_partial_cache(self, tmp_path):
        """If only SOME cells are cached, the run does the work fresh
        (run_num is the smallest cache unit; partial-run cache hit
        would require splitting the conditioning phase, which we don't
        support today)."""
        from src.runners.exp3c import run_exp3c

        cfg = ExperimentConfig(
            model_name="counting-client",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
            num_runs=1,
            num_conditioning_turns=2,
            seed=0,
        )
        items = [
            {"difficulty": "easy", "question": "Q1?", "expected": "A1"},
            {"difficulty": "hard", "question": "Q2?", "expected": "A2"},
        ]

        out_dir = tmp_path / "exp3c_partial"
        c1 = _CountingClient(responses=["10", "5", "ans1", "ans2"])
        async for _ in run_exp3c(cfg, c1, items=items, output_dir=out_dir):
            pass

        # Delete one of the cell files so the cache is partial.
        cell_to_remove = next((out_dir / "neutral").glob("*.json"))
        cell_to_remove.unlink()

        # Re-run — partial cache means full re-execution.
        c2 = _CountingClient(responses=["10", "5", "ans1", "ans2"])
        async for _ in run_exp3c(cfg, c2, items=items, output_dir=out_dir):
            pass
        assert c2.complete_calls > 0, "partial cache must trigger re-run"


class TestExp3bCache:
    @pytest.mark.asyncio
    async def test_exp3b_skips_api_when_all_prompts_cached(self, tmp_path):
        from src.runners.exp3b import run_exp3b

        cfg = ExperimentConfig(
            model_name="counting-client",
            condition=Condition.STRONG_NEGATIVE,
            experiment_type=ExperimentType.COGNITIVE_SCOPE,
            num_runs=1,
            num_conditioning_turns=2,
            seed=0,
        )
        prompts = [
            {"id": "p1", "text": "Tell a short story."},
            {"id": "p2", "text": "List uses for a paperclip."},
        ]

        out_dir = tmp_path / "exp3b_cache_test"
        c1 = _CountingClient(responses=["10", "5"] + ["gen"] * 100)
        results: list = []
        async for r in run_exp3b(
            cfg, c1, prompts=prompts, n_generations=2,
            output_dir=out_dir,
        ):
            results.append(r)
        assert len(results) == 2  # 1 run × 2 prompts
        assert c1.complete_calls > 0

        # Second run hits cache.
        c2 = _CountingClient(responses=["NEVER"] * 100)
        results2: list = []
        async for r in run_exp3b(
            cfg, c2, prompts=prompts, n_generations=2,
            output_dir=out_dir,
        ):
            results2.append(r)
        assert len(results2) == 2
        assert c2.complete_calls == 0, (
            f"second run must hit cache; got {c2.complete_calls} API calls"
        )
