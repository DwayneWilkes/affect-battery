"""Pipeline DAG executor with content-addressed caching.

Each Stage has a name, declared input keys (config keys and/or upstream
artifact names), declared output keys, and a run_fn. `PipelineRunner.run`
executes stages in declared order; cache hits skip re-execution.

All stage events are emitted to `events.jsonl` under cache_root. The
pipeline manifest (`pipeline_manifest.json`) records each stage's
input_hash so the whole pipeline's state is reconstructable from the
file-backed artifacts alone (no external tracker service required).

Spec: affect-battery-task-difficulty-calibration::pipeline. Group 15.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.pipeline.cache import cache_dir_for, stage_input_hash


# ───────────────────────── Stage ──────────────────────────


@dataclass(frozen=True)
class Stage:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    run_fn: Callable[[dict, dict], dict]


# ───────────────────────── PipelineRunner ──────────────────────────


@dataclass
class PipelineRunner:
    stages: list[Stage]
    cache_root: Path
    events_path: Path = field(init=False)
    manifest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.cache_root / "events.jsonl"
        self.manifest_path = self.cache_root / "pipeline_manifest.json"

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        """Execute all stages in order. Returns the accumulated upstream
        artifacts dict at the end."""
        upstream_artifacts: dict[str, Any] = {}
        manifest_entries: list[dict[str, Any]] = []

        for stage in self.stages:
            input_hash = stage_input_hash(stage, config, upstream_artifacts)
            stage_dir = cache_dir_for(stage, input_hash, self.cache_root)
            outputs_path = stage_dir / "outputs.json"

            if outputs_path.exists():
                self._emit(
                    "stage_cache_hit",
                    stage=stage.name,
                    input_hash=input_hash,
                )
                outputs = json.loads(outputs_path.read_text())
            else:
                self._emit(
                    "stage_start",
                    stage=stage.name,
                    input_hash=input_hash,
                )
                outputs = stage.run_fn(config, upstream_artifacts)
                stage_dir.mkdir(parents=True, exist_ok=True)
                outputs_path.write_text(json.dumps(outputs, indent=2, sort_keys=True))
                self._emit(
                    "stage_complete",
                    stage=stage.name,
                    input_hash=input_hash,
                )

            upstream_artifacts.update(outputs)
            manifest_entries.append(
                {
                    "name": stage.name,
                    "input_hash": input_hash,
                    "inputs": list(stage.inputs),
                    "outputs": list(stage.outputs),
                }
            )

        self.manifest_path.write_text(
            json.dumps({"stages": manifest_entries}, indent=2)
        )
        return upstream_artifacts

    def _emit(self, event_type: str, **fields: Any) -> None:
        event = {
            "event_type": event_type,
            "timestamp_epoch": time.time(),
            **fields,
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")
