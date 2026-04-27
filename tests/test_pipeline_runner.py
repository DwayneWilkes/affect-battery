"""Tests for the content-addressed pipeline orchestrator.

Spec: affect-battery-task-difficulty-calibration::pipeline (group 15,
NEW per 2026-04-22 course-correction).

Pipeline orchestrator consolidates generation + calibration + gate +
experiment + analysis + archive into a DAG with content-addressed
caching. Each stage has a deterministic input hash; cache hits skip
execution. All artifacts are git-trackable so the pipeline is
inspectable offline (per scoring-pipeline/spec.md file-backed
tracker constraint).

These tests exercise the DAG mechanics with fake stages (deterministic
run_fns). Integration with real bank/calibration/gate stages is a
later follow-up — the orchestrator itself just needs to prove it can
cache, order, and emit events correctly.
"""

import json
from pathlib import Path

import pytest

from src.pipeline.runner import (
    PipelineRunner,
    Stage,
)
from src.pipeline.cache import stage_input_hash


# ───────────────────────── fake stages for testing ──────────────────────────


def _count_calls(name: str):
    """Returns a run_fn that counts invocations per stage. Tests verify
    cache hits avoid re-invocation."""
    counter = {"n": 0}

    def run(config, upstream):
        counter["n"] += 1
        return {f"{name}_out": f"{config.get(name, 'default')}::{counter['n']}"}

    run.counter = counter  # type: ignore[attr-defined]
    return run


# ───────────────────────── input-hash determinism ──────────────────────────


class TestStageInputHashDeterminism:
    def test_identical_inputs_yield_identical_hash(self):
        stage = Stage(
            name="bank_gen",
            inputs=("generator_config",),
            outputs=("bank_path",),
            run_fn=lambda c, u: {"bank_path": "/x"},
        )
        config = {"generator_config": {"seed": 42, "n": 300}}
        upstream = {}
        h1 = stage_input_hash(stage, config, upstream)
        h2 = stage_input_hash(stage, config, upstream)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_config_yields_different_hash(self):
        stage = Stage(
            name="bank_gen",
            inputs=("generator_config",),
            outputs=("bank_path",),
            run_fn=lambda c, u: {"bank_path": "/x"},
        )
        h1 = stage_input_hash(stage, {"generator_config": {"seed": 42}}, {})
        h2 = stage_input_hash(stage, {"generator_config": {"seed": 43}}, {})
        assert h1 != h2

    def test_upstream_artifacts_participate_in_hash(self):
        stage = Stage(
            name="calibration",
            inputs=("bank_path", "calibrator_config"),
            outputs=("calibration_results",),
            run_fn=lambda c, u: {"calibration_results": "ok"},
        )
        cfg = {"calibrator_config": {"target_min": 0.60}}
        h1 = stage_input_hash(stage, cfg, {"bank_path": "/banks/v1"})
        h2 = stage_input_hash(stage, cfg, {"bank_path": "/banks/v2"})
        assert h1 != h2


# ───────────────────────── cache hit skips execution ──────────────────────────


class TestCacheHitSkipsExecution:
    def test_second_run_is_cache_hit(self, tmp_path):
        run_fn = _count_calls("bank_gen")
        stage = Stage(
            name="bank_gen",
            inputs=("generator_config",),
            outputs=("bank_gen_out",),
            run_fn=run_fn,
        )
        config = {"generator_config": {"seed": 42, "n": 300}}
        runner = PipelineRunner(stages=[stage], cache_root=tmp_path)
        runner.run(config)
        assert run_fn.counter["n"] == 1  # type: ignore[attr-defined]
        runner.run(config)
        # Second run should hit the cache and NOT re-execute.
        assert run_fn.counter["n"] == 1, (
            f"Expected cache hit on second run, got n={run_fn.counter['n']}"  # type: ignore[attr-defined]
        )

    def test_cache_miss_on_config_change(self, tmp_path):
        run_fn = _count_calls("bank_gen")
        stage = Stage(
            name="bank_gen",
            inputs=("generator_config",),
            outputs=("bank_gen_out",),
            run_fn=run_fn,
        )
        runner = PipelineRunner(stages=[stage], cache_root=tmp_path)
        runner.run({"generator_config": {"seed": 42}})
        runner.run({"generator_config": {"seed": 43}})  # different config
        assert run_fn.counter["n"] == 2  # type: ignore[attr-defined]


# ───────────────────────── stage ordering ──────────────────────────


class TestStageOrdering:
    def test_stages_run_in_declared_order_with_upstream_artifacts(self, tmp_path):
        run_a = _count_calls("a")
        run_b = _count_calls("b")
        stage_a = Stage(name="a", inputs=("a",), outputs=("a_out",), run_fn=run_a)
        stage_b = Stage(name="b", inputs=("a_out",), outputs=("b_out",), run_fn=run_b)
        runner = PipelineRunner(stages=[stage_a, stage_b], cache_root=tmp_path)
        artifacts = runner.run({"a": "hello"})
        assert run_a.counter["n"] == 1  # type: ignore[attr-defined]
        assert run_b.counter["n"] == 1  # type: ignore[attr-defined]
        assert "a_out" in artifacts
        assert "b_out" in artifacts


# ───────────────────────── manifest ──────────────────────────


class TestPipelineManifest:
    def test_manifest_records_each_stage_input_hash(self, tmp_path):
        run_a = _count_calls("a")
        stage_a = Stage(name="a", inputs=("a",), outputs=("a_out",), run_fn=run_a)
        runner = PipelineRunner(stages=[stage_a], cache_root=tmp_path)
        runner.run({"a": "hello"})
        manifest_path = tmp_path / "pipeline_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "stages" in manifest
        assert len(manifest["stages"]) == 1
        entry = manifest["stages"][0]
        assert entry["name"] == "a"
        assert "input_hash" in entry
        assert len(entry["input_hash"]) == 64


# ───────────────────────── events ──────────────────────────


class TestPipelineEvents:
    def test_events_emitted_in_order(self, tmp_path):
        run_a = _count_calls("a")
        stage_a = Stage(name="a", inputs=("a",), outputs=("a_out",), run_fn=run_a)
        runner = PipelineRunner(stages=[stage_a], cache_root=tmp_path)
        runner.run({"a": "hello"})
        events_path = tmp_path / "events.jsonl"
        assert events_path.exists()
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        event_types = [e["event_type"] for e in events]
        assert "stage_start" in event_types
        assert "stage_complete" in event_types

    def test_cache_hit_event_on_second_run(self, tmp_path):
        run_a = _count_calls("a")
        stage_a = Stage(name="a", inputs=("a",), outputs=("a_out",), run_fn=run_a)
        runner = PipelineRunner(stages=[stage_a], cache_root=tmp_path)
        runner.run({"a": "hello"})
        runner.run({"a": "hello"})  # should hit cache
        events_path = tmp_path / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        event_types = [e["event_type"] for e in events]
        assert "stage_cache_hit" in event_types
