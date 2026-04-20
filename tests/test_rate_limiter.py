"""Tests for the token-bucket rate limiter.

Spec: affect-battery-compute-guardrails::compute-guardrails Requirement 3
(Rate limiting).
"""

import asyncio
import time

from src.conditioning.prompts import Condition
from src.models import DryRunClient
from src.runner import (
    ExperimentConfig,
    ExperimentType,
    _TokenBucket,
    run_batch,
)


# ---------------------------------------------------------------------------
# TokenBucket unit tests
# ---------------------------------------------------------------------------


def test_token_bucket_starts_full():
    bucket = _TokenBucket(rate_per_second=10, capacity=5)

    async def go():
        # Five rapid acquisitions should all succeed immediately.
        for _ in range(5):
            await bucket.acquire()

    asyncio.run(go())


def test_token_bucket_refills_at_rate():
    bucket = _TokenBucket(rate_per_second=10, capacity=1)

    async def go():
        start = time.time()
        await bucket.acquire()  # first token immediate
        await bucket.acquire()  # second token needs ~0.1s refill
        await bucket.acquire()  # third token needs ~0.1s more
        return time.time() - start

    elapsed = asyncio.run(go())
    # At 10 tokens/s, 2 refills is ~0.2s. Allow generous slack for CI.
    assert elapsed >= 0.15, f"Bucket drained too fast: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Integration with run_batch
# ---------------------------------------------------------------------------


def test_rate_limit_caps_throughput(tmp_path):
    """Spec scenario: rps=5, 20 concurrent workers, aggregate <= ~5/s.

    Scale down for test speed: we measure that a batch of 10 runs (each
    calling the API once) at rps=5 takes at least ~2 seconds (10 calls /
    5 per second = 2 seconds minimum).
    """
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NO_CONDITIONING,  # 0 conditioning calls
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=10,
        num_conditioning_turns=0,
        num_transfer_questions=1,  # 1 call per run
        seed=42,
    )
    client = DryRunClient(responses=["Paris"] * 20)

    async def go():
        start = time.time()
        async for _ in run_batch(
            config, client,
            max_concurrent=20,
            rate_limit_rps=5.0,
            output_dir=tmp_path,
        ):
            pass
        return time.time() - start

    elapsed = asyncio.run(go())
    # 10 calls at 5/s minimum bound, minus bucket capacity=5 starting full.
    # First 5 are instant, next 5 take 5 * 0.2s = 1.0s minimum.
    assert elapsed >= 0.8, f"Rate limit not enforced: {elapsed:.3f}s for 10 calls at rps=5"


def test_rate_limit_no_op_when_unset(tmp_path):
    """Default behaviour: no rate limiter means no throttling."""
    config = ExperimentConfig(
        model_name="test-model",
        condition=Condition.NO_CONDITIONING,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=5,
        num_conditioning_turns=0,
        num_transfer_questions=1,
        seed=42,
    )
    client = DryRunClient(responses=["Paris"] * 10)

    async def go():
        start = time.time()
        async for _ in run_batch(config, client, max_concurrent=5, output_dir=tmp_path):
            pass
        return time.time() - start

    elapsed = asyncio.run(go())
    # Should be basically instant.
    assert elapsed < 1.0, f"Default path too slow: {elapsed:.3f}s"
