"""Experiment tracker with MLflow + disk caching + decorators.

General-purpose ML experiment tracking library. Provides:
- Structured logging (params, metrics, items, verdicts)
- Disk-based caching with checkpoint/resume
- Optional MLflow integration (works without mlflow installed)
- Per-item tracking with hash-based dedup
- Stage timing via context manager

Usage:
    tracker = ExperimentTracker(output_dir="./output/run_001", experiment_name="my_exp")
    tracker.log_params(model_id="qwen-7b", n_per_group=200)
    with tracker.stage("extraction"):
        for prompt in prompts:
            if tracker.is_cached(prompt.hash):
                continue
            features = extract(prompt)
            tracker.log_item(prompt.hash, features)
    tracker.log_metric("auroc", 0.85)

Not KV-cache-specific. Usable for any ML experiment pipeline.
"""

import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


_SAFE_KEY_DISALLOWED = re.compile(r"[^A-Za-z0-9._\-]")


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON to ``path`` via a tmp-and-rename. SIGTERM during the
    write leaves either the previous file or a stale ``.tmp`` sibling,
    never a truncated target. ``os.replace`` is atomic on POSIX."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _utc_now() -> str:
    """UTC timestamp in ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


class ExperimentTracker:
    """Experiment tracker with disk caching and optional MLflow.

    Creates a structured output directory:
        {output_dir}/
            run_metadata.json   — params, metrics, stages, verdicts, git hash
            cache/              — per-item cached results (JSON)
            run.log             — structured log (JSON lines)
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str,
        use_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.experiment_name = experiment_name

        # Initialize metadata
        self._metadata_path = self.output_dir / "run_metadata.json"
        if self._metadata_path.exists():
            with open(self._metadata_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {
                "experiment_name": experiment_name,
                "created_at": _utc_now(),
                "git_hash": _get_git_hash(),
                "params": {},
                "metrics": {},
                "stages": {},
                "verdicts": {},
            }
            self._save_metadata()

        # MLflow integration
        self._mlflow_run = None
        if use_mlflow:
            self._init_mlflow(mlflow_tracking_uri)

    def _init_mlflow(self, tracking_uri: Optional[str]) -> None:
        """Initialize MLflow run. Fails gracefully if mlflow not installed."""
        try:
            import mlflow
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_run = mlflow.start_run()
        except ImportError:
            pass
        except Exception:
            pass

    def _save_metadata(self) -> None:
        """Write metadata to disk atomically."""
        _atomic_write_json(self._metadata_path, self._metadata)

    # ================================================================
    # Parameters
    # ================================================================

    def log_params(self, **kwargs) -> None:
        """Log hyperparameters."""
        self._metadata["params"].update(kwargs)
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                for k, v in kwargs.items():
                    mlflow.log_param(k, v)
            except Exception:
                pass

    # ================================================================
    # Metrics
    # ================================================================

    def log_metric(self, name: str, value: float) -> None:
        """Log a scalar metric."""
        self._metadata["metrics"][name] = value
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                mlflow.log_metric(name, value)
            except Exception:
                pass

    def log_metrics(self, **kwargs: float) -> None:
        """Log multiple metrics with a single metadata write."""
        self._metadata["metrics"].update(kwargs)
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                for k, v in kwargs.items():
                    mlflow.log_metric(k, v)
            except Exception:
                pass

    # ================================================================
    # Per-Item Caching
    # ================================================================

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Sanitize a cache key to a filesystem-safe string.

        Allowlist: only [A-Za-z0-9._\\-] survive; anything else becomes
        underscore. Runs of two or more dots are collapsed to a single
        underscore so an attacker cannot reassemble `..` from individually
        allowed characters. Leading dots are stripped to prevent hidden
        files. Empty results coerce to '_'. Defense-in-depth boundary:
        callers are operator-curated today, but this is the natural
        enforcement point for any future bank source."""
        sanitized = _SAFE_KEY_DISALLOWED.sub("_", key)
        sanitized = re.sub(r"\.{2,}", "_", sanitized)
        sanitized = sanitized.lstrip(".")
        return sanitized or "_"

    def log_item(self, key: str, data: Dict[str, Any]) -> None:
        """Log a per-item result to disk cache atomically."""
        safe_key = self._sanitize_key(key)
        cache_path = self.cache_dir / f"{safe_key}.json"
        _atomic_write_json(cache_path, data)

    def is_cached(self, key: str) -> bool:
        """Check if an item is already cached."""
        safe_key = self._sanitize_key(key)
        return (self.cache_dir / f"{safe_key}.json").exists()

    def load_cached(self, key: str) -> Dict[str, Any]:
        """Load a cached item."""
        safe_key = self._sanitize_key(key)
        cache_path = self.cache_dir / f"{safe_key}.json"
        with open(cache_path) as f:
            return json.load(f)

    # ================================================================
    # Stage Timing
    # ================================================================

    @contextmanager
    def stage(self, name: str):
        """Context manager that times a pipeline stage."""
        start = _utc_now()
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            end = _utc_now()
            self._metadata["stages"][name] = {
                "start": start,
                "end": end,
                "duration_seconds": round(elapsed, 3),
            }
            self._save_metadata()

            if self._mlflow_run:
                try:
                    import mlflow
                    mlflow.log_metric(f"stage_{name}_duration_s", elapsed)
                except Exception:
                    pass

    # ================================================================
    # Verdicts
    # ================================================================

    def log_verdict(self, claim_id: str, verdict: str, evidence: str) -> None:
        """Log a structured verdict for a claim."""
        self._metadata["verdicts"][claim_id] = {
            "verdict": verdict,
            "evidence": evidence,
            "timestamp": _utc_now(),
        }
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                mlflow.set_tag(f"verdict_{claim_id}", verdict)
            except Exception:
                pass

    # ================================================================
    # Artifacts
    # ================================================================

    def log_artifact(self, local_path: str, artifact_subdir: Optional[str] = None) -> None:
        """Log a file as an artifact (stored with this run)."""
        if self._mlflow_run:
            try:
                import mlflow
                if artifact_subdir:
                    mlflow.log_artifact(local_path, artifact_subdir)
                else:
                    mlflow.log_artifact(local_path)
            except Exception:
                pass

    def log_dataset(self, path: str, name: str, context: str = "evaluation") -> None:
        """Log a dataset reference (MLflow dataset tracking)."""
        self._metadata.setdefault("datasets", {})[name] = {
            "path": str(path),
            "context": context,
            "logged_at": _utc_now(),
        }
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                from mlflow.data.sources import LocalArtifactDatasetSource
                ds = mlflow.data.from_json(
                    path=str(path),
                    source=LocalArtifactDatasetSource(path=str(path)),
                    name=name,
                )
                mlflow.log_input(ds, context=context)
            except Exception:
                pass

    def enable_sklearn_autolog(self) -> None:
        """Enable MLflow sklearn autologging for classifier runs."""
        if self._mlflow_run:
            try:
                import mlflow
                mlflow.sklearn.autolog(
                    log_models=False,  # don't persist LogisticRegression models
                    log_datasets=False,  # we log datasets manually
                    silent=True,
                )
            except Exception:
                pass

    # ================================================================
    # Tags
    # ================================================================

    def set_tag(self, key: str, value: str) -> None:
        """Set a run tag (for filtering in MLflow UI)."""
        self._metadata.setdefault("tags", {})[key] = value
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                mlflow.set_tag(key, value)
            except Exception:
                pass

    # ================================================================
    # Lifecycle
    # ================================================================

    def end(self) -> None:
        """End the tracking run. Finalize MLflow."""
        self._metadata["ended_at"] = _utc_now()
        self._save_metadata()

        if self._mlflow_run:
            try:
                import mlflow
                # Log the metadata file as artifact
                mlflow.log_artifact(str(self._metadata_path))
                mlflow.end_run()
            except Exception:
                pass
            self._mlflow_run = None
