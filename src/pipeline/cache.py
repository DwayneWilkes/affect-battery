"""Content-addressed caching for pipeline stages.

Stage inputs (config dict + upstream artifacts) are canonicalized and
SHA-256 hashed; the hash keys a directory under `cache_root` that holds
the stage's serialized outputs. Repeat runs with identical inputs hit
the cache and skip execution.

Spec: affect-battery-task-difficulty-calibration::pipeline. Group 15.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from src.pipeline.runner import Stage


def _canonicalize(value: Any) -> Any:
    """Turn a value into something json.dumps can handle deterministically.

    Dataclasses → dict; sets → sorted list; everything else passes through.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _canonicalize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, set):
        return sorted(_canonicalize(v) for v in value)
    return value


def stage_input_hash(
    stage: "Stage",
    config: dict[str, Any],
    upstream_artifacts: dict[str, Any],
) -> str:
    """SHA-256 of the canonicalized (stage_name, input_keys, inputs) tuple.

    `config` contains the full pipeline config; `stage.inputs` names the
    subset this stage depends on (either top-level config keys or upstream
    artifact names). Any change to those inputs changes the hash.
    """
    relevant: dict[str, Any] = {}
    for key in stage.inputs:
        if key in config:
            relevant[key] = config[key]
        elif key in upstream_artifacts:
            relevant[key] = upstream_artifacts[key]
        else:
            relevant[key] = None  # absent input — still hashed to stay deterministic
    payload = {
        "stage_name": stage.name,
        "inputs": _canonicalize(relevant),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_dir_for(
    stage: "Stage",
    input_hash: str,
    cache_root: Path,
) -> Path:
    """Return the cache directory for a (stage, input_hash) pair.

    Short-hash directory name (first 16 hex chars) keeps paths human-
    readable while preserving cache-identity strength.
    """
    return Path(cache_root) / f"{stage.name}-{input_hash[:16]}"
