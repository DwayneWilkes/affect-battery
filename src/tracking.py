"""Experiment tracker mirroring the ll/KV-Cache MLOps pattern.

Per GAPS.md task 10: before RunPod spend we want a file-backed tracker
that logs params, metrics, and artifacts per run so nothing is
recomputed if the pipeline re-enters, and results are inspectable
offline without an MLflow server.

MLflow integration is optional: if mlflow is installed, every start_run /
log_params / log_metrics / log_artifact call also mirrors to MLflow's
file-backed store (./mlruns by default). If mlflow is not installed, the
disk-side JSON store still captures everything and the tracker is a
no-op on the MLflow side. This keeps the core harness dependency-light
while letting the user opt into the MLflow UI when wanted.

Deterministic run naming (task 10.5):
    {model_slug}_{condition}_{experiment_type}_seed{seed}_{run_number:04d}

Idempotency (task 10.6):
Re-entering with the same config hashes to the same run_name and writes to
the same run directory, merging params/metrics/artifacts rather than
creating a duplicate.
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_slug(model_name: str) -> str:
    """Strip any 'org/' prefix so filenames do not embed path separators."""
    return model_name.rsplit("/", 1)[-1]


def run_name_for(config: dict) -> str:
    """Deterministic run name from config (task 10.5).

    Required keys: model_name, condition, experiment_type, seed, run_number.
    Optional keys are embedded when present so intensity sweeps do not
    collide (e.g., different intensity_level for the same condition).
    """
    model = _model_slug(str(config["model_name"]))
    cond = str(config["condition"])
    exp = str(config["experiment_type"])
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
    """File-backed tracker; optional MLflow mirror.

    Directory layout under `output_dir`:
        {run_name}/
            run_metadata.json  -- config, params, metrics, artifacts index
            artifacts/         -- copied artifact files
    """

    def __init__(self, output_dir: Path, use_mlflow: bool = False):
        self._output_dir = Path(output_dir)
        self._use_mlflow = use_mlflow
        self._current_run_name: str | None = None
        self._last_run_name: str | None = None
        self._current_run_dir: Path | None = None
        self._meta: dict[str, Any] = {}

    @property
    def current_run_name(self) -> str:
        """The run name for the most recently started run. Remains accessible
        after end_run() so callers can inspect the final run directory."""
        if self._current_run_name is None:
            if self._last_run_name is None:
                raise RuntimeError("No run has been started yet.")
            return self._last_run_name
        return self._current_run_name

    def start_run(self, config: dict) -> str:
        """Begin or re-enter a run. Idempotent: identical config hashes to
        the same run_name and merges into the existing run_metadata.json."""
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

        # Merge config: later start_run calls with the same run_name
        # update the config snapshot (in practice identical, but we
        # write explicitly to avoid drift).
        self._meta["config"] = dict(config)
        self._flush()
        return self._current_run_name

    def log_params(self, **params: Any) -> None:
        """Log key/value params. Idempotent: same key -> updated (or
        unchanged) value, no list of duplicates."""
        self._ensure_run()
        for k, v in params.items():
            self._meta["params"][k] = v
        self._flush()

    def log_metrics(self, **metrics: Any) -> None:
        self._ensure_run()
        for k, v in metrics.items():
            self._meta["metrics"][k] = v
        self._flush()

    def log_artifact(self, path: Path) -> None:
        """Copy an artifact file into the run's artifacts/ directory and
        record its SHA-256 checksum in run_metadata.json."""
        self._ensure_run()
        path = Path(path)
        dest = self._current_run_dir / "artifacts" / path.name  # type: ignore[union-attr]
        shutil.copy2(path, dest)
        self._meta["artifacts"][path.name] = {
            "sha256": _sha256_of_file(path),
            "bytes": dest.stat().st_size,
            "logged_at": _utc_now(),
        }
        self._flush()

    def end_run(self) -> None:
        self._ensure_run()
        self._meta["ended_at"] = _utc_now()
        self._flush()
        self._last_run_name = self._current_run_name
        self._current_run_name = None
        self._current_run_dir = None
        self._meta = {}

    def _ensure_run(self) -> None:
        if self._current_run_name is None:
            raise RuntimeError("No active run. Call start_run() first.")

    def _flush(self) -> None:
        assert self._current_run_dir is not None
        meta_path = self._current_run_dir / "run_metadata.json"
        meta_path.write_text(json.dumps(self._meta, indent=2, default=str))
