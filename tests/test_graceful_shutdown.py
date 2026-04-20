"""Tests for graceful shutdown via cooperative cancel event.

Spec: affect-battery-compute-guardrails::compute-guardrails Requirement 8.
"""

import asyncio
import json

from src.conditioning.prompts import Condition
from src.models import DryRunClient
from src.runner import (
    ExperimentConfig,
    ExperimentType,
    run_batch,
)


def _config(num_runs=20):
    return ExperimentConfig(
        model_name="test-model",
        condition=Condition.NO_CONDITIONING,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=num_runs,
        num_conditioning_turns=0,
        num_transfer_questions=1,
        seed=42,
    )


def test_cancel_event_stops_new_dispatch(tmp_path):
    """Setting cancel_event mid-batch stops new runs from starting."""
    cancel = asyncio.Event()
    config = _config(num_runs=20)
    client = DryRunClient(responses=["Paris"] * 30)

    async def go():
        results = []
        # Rate-limit to slow the batch so we have time to signal cancel.
        async for r in run_batch(
            config, client,
            max_concurrent=1,
            output_dir=tmp_path,
            rate_limit_rps=50.0,
            cancel_event=cancel,
        ):
            results.append(r)
            if len(results) == 3:
                cancel.set()
        return results

    results = asyncio.run(go())
    assert len(results) < 20, f"Cancel did not stop dispatch: {len(results)} runs completed"
    assert len(results) >= 3, "At least the 3 runs before cancel should yield"


def test_shutdown_event_emitted(tmp_path):
    """Signal cancel mid-batch, drain the rest, verify shutdown event fired."""
    cancel = asyncio.Event()
    config = _config(num_runs=10)
    client = DryRunClient(responses=["Paris"] * 20)

    async def go():
        results = []
        async for r in run_batch(
            config, client,
            max_concurrent=1,
            output_dir=tmp_path,
            rate_limit_rps=50.0,
            cancel_event=cancel,
        ):
            results.append(r)
            if len(results) == 1:
                cancel.set()
        return results

    results = asyncio.run(go())
    assert len(results) < 10, "Cancel did not stop dispatch"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    event_types = [e["event"] for e in events]
    assert "batch_shutdown_signal" in event_types


def test_cancel_does_not_affect_uncancelled_run(tmp_path):
    """No cancel_event set -> all 5 runs complete normally."""
    config = _config(num_runs=5)
    client = DryRunClient(responses=["Paris"] * 10)

    async def go():
        results = []
        async for r in run_batch(
            config, client,
            max_concurrent=2,
            output_dir=tmp_path,
        ):
            results.append(r)
        return results

    results = asyncio.run(go())
    assert len(results) == 5
