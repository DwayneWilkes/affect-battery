"""File-backed experiment tracker.

Every run becomes a directory under `output_dir` with a `run_metadata.json`
index and an `artifacts/` folder. Re-starting with the same config resolves
to the same run (idempotent merge).
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .util import enum_value, model_slug


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_name_for(config: dict) -> str:
    """Deterministic run name.

    Required keys: model_name, condition, experiment_type, seed, run_number.
    Optional: intensity_level, intensity_set.
    """
    model = model_slug(config["model_name"])
    cond = enum_value(config["condition"])
    exp = enum_value(config["experiment_type"])
    seed = int(config["seed"])
    run_number = int(config["run_number"])

    name = f"{model}_{cond}_{exp}_seed{seed}_{run_number:04d}"
    if config.get("intensity_level") is not None:
        name += f"_lvl{int(config['intensity_level'])}"
    if config.get("intensity_set"):
        name += f"_{config['intensity_set']}"
    return name


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ExperimentTracker:
    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)
        self._current_run_name: str | None = None
        self._current_run_dir: Path | None = None
        self._meta: dict[str, Any] = {}
        self._dirty = False

    @property
    def current_run_name(self) -> str:
        if self._current_run_name is None:
            raise RuntimeError("No run has been started yet. Call start_run() first.")
        return self._current_run_name

    def start_run(self, config: dict) -> str:
        self._current_run_name = run_name_for(config)
        self._current_run_dir = self._output_dir / self._current_run_name
        self._current_run_dir.mkdir(parents=True, exist_ok=True)
        (self._current_run_dir / "artifacts").mkdir(exist_ok=True)

        meta_path = self._current_run_dir / "run_metadata.json"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text())
        else:
            self._meta = {
                "run_name": self._current_run_name,
                "config": {},
                "params": {},
                "metrics": {},
                "artifacts": {},
                "started_at": _utc_now(),
                "ended_at": None,
            }

        self._meta["config"] = dict(config)
        self._dirty = True
        self._flush()
        return self._current_run_name

    def log_params(self, **params: Any) -> None:
        self._ensure_run()
        for k, v in params.items():
            self._meta["params"][k] = v
        self._dirty = True

    def log_metrics(self, **metrics: Any) -> None:
        self._ensure_run()
        for k, v in metrics.items():
            self._meta["metrics"][k] = v
        self._dirty = True

    def log_artifact(self, path: Path) -> None:
        """Copy an artifact file into artifacts/ and record its SHA-256.

        Flushes immediately: artifacts are costly to recompute, so crash
        safety matters here more than for params/metrics.
        """
        self._ensure_run()
        path = Path(path)
        dest = self._current_run_dir / "artifacts" / path.name  # type: ignore[union-attr]
        shutil.copy2(path, dest)
        self._meta["artifacts"][path.name] = {
            "sha256": _sha256_of_file(path),
            "bytes": dest.stat().st_size,
            "logged_at": _utc_now(),
        }
        self._dirty = True
        self._flush()

    def end_run(self) -> None:
        self._ensure_run()
        self._meta["ended_at"] = _utc_now()
        self._dirty = True
        self._flush()
        # Keep _current_run_name set so current_run_name remains accessible
        # after end_run() for callers that inspect the run directory.
        self._current_run_dir = None
        self._meta = {}

    def _ensure_run(self) -> None:
        if self._current_run_dir is None:
            # _current_run_name may still be set post-end_run for inspection,
            # but the run is no longer active.
            raise RuntimeError("No active run. Call start_run() first.")

    def _flush(self) -> None:
        if not self._dirty or self._current_run_dir is None:
            return
        meta_path = self._current_run_dir / "run_metadata.json"
        meta_path.write_text(json.dumps(self._meta, indent=2, default=str))
        self._dirty = False
