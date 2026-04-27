"""exp3b / exp3c per-cell cache contract.

Each (run_num, axis_idx) cell is cached independently on disk. When a
re-run finds a cell whose stored transfer_bank_hash matches, schema
validates, and checksum verifies, the runner yields it without firing
the conditioning phase or any per-cell API call. A run_num with at
least one missing cell runs conditioning once and dispatches only the
missing cells.

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
    async def test_exp3c_partial_cache_only_re_runs_missing_cells(self, tmp_path):
        """Per-cell cache granularity: with N items where only one is
        missing from cache, the re-run must execute exactly:
        num_conditioning_turns conditioning calls + 1 missing-item call.
        Cached cells yield from disk with zero API calls."""
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
        results: list = []
        async for r in run_exp3c(cfg, c1, items=items, output_dir=out_dir):
            results.append(r)
        assert len(results) == 2

        # Delete one cell file so the cache is partial.
        cell_to_remove = next((out_dir / "neutral").glob("*.json"))
        cell_to_remove.unlink()

        # Re-run: should yield the cached cell with zero API calls,
        # then run conditioning (2 calls) + 1 missing item (1 call) = 3.
        c2 = _CountingClient(responses=["10", "5", "ans_missing"])
        results2: list = []
        async for r in run_exp3c(cfg, c2, items=items, output_dir=out_dir):
            results2.append(r)
        assert len(results2) == 2, "must yield both cached + freshly-run cells"
        assert c2.complete_calls == 3, (
            f"partial cache must re-run only missing cells "
            f"(2 conditioning + 1 missing item = 3); got {c2.complete_calls}"
        )

    @pytest.mark.asyncio
    async def test_exp3c_skips_conditioning_when_all_cells_cached(self, tmp_path):
        """All-cells-cached path must skip the conditioning phase
        entirely — no API calls at all, even though conditioning would
        otherwise fire once per run_num."""
        from src.runners.exp3c import run_exp3c

        cfg = ExperimentConfig(
            model_name="counting-client",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
            num_runs=2,
            num_conditioning_turns=2,
            seed=0,
        )
        items = [
            {"difficulty": "easy", "question": "Q1?", "expected": "A1"},
            {"difficulty": "hard", "question": "Q2?", "expected": "A2"},
        ]

        out_dir = tmp_path / "exp3c_skip_cond"
        # First run: 2 runs * (2 conditioning + 2 items) = 8 calls.
        c1 = _CountingClient(responses=["10", "5", "a", "b"] * 4)
        async for _ in run_exp3c(cfg, c1, items=items, output_dir=out_dir):
            pass

        # Second run: every cell cached → ZERO calls (no conditioning).
        c2 = _CountingClient(responses=["NEVER"] * 50)
        async for _ in run_exp3c(cfg, c2, items=items, output_dir=out_dir):
            pass
        assert c2.complete_calls == 0


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

    @pytest.mark.asyncio
    async def test_exp3b_partial_cache_only_re_runs_missing_prompts(self, tmp_path):
        """Per-cell cache granularity for exp3b: with one prompt cached
        and one missing, re-run must do conditioning + n_generations
        for the one missing prompt only — not both."""
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
        n_gen = 2

        out_dir = tmp_path / "exp3b_partial"
        c1 = _CountingClient(responses=["10", "5"] + ["gen"] * 100)
        async for _ in run_exp3b(
            cfg, c1, prompts=prompts, n_generations=n_gen, output_dir=out_dir,
        ):
            pass

        # Delete one cell so cache is partial.
        cell_to_remove = next((out_dir / "strong_negative").glob("*.json"))
        cell_to_remove.unlink()

        # Re-run: 1 cached cell yielded with no API, then 2 conditioning
        # + 1 missing prompt * 2 generations = 4 calls (not 6).
        c2 = _CountingClient(responses=["10", "5"] + ["gen"] * 100)
        results2: list = []
        async for r in run_exp3b(
            cfg, c2, prompts=prompts, n_generations=n_gen, output_dir=out_dir,
        ):
            results2.append(r)
        assert len(results2) == 2
        expected_calls = cfg.num_conditioning_turns + 1 * n_gen
        assert c2.complete_calls == expected_calls, (
            f"partial cache must re-run only missing prompt "
            f"({cfg.num_conditioning_turns} conditioning + {n_gen} generations "
            f"= {expected_calls}); got {c2.complete_calls}"
        )
