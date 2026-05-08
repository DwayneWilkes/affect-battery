"""Experiment tracker with MLflow + disk caching + decorators.

General-purpose ML experiment tracking library. Provides:
- Structured logging (params, metrics, items, verdicts)
- Disk-based caching with checkpoint/resume
- Optional MLflow integration (works without mlflow installed)
- Per-item tracking with hash-based dedup
- Stage timing via context manager
- Decorator-based API for auto-caching, timing, and validation

Imperative API:
    tracker = ExperimentTracker(output_dir="./output/run_001", experiment_name="my_exp")
    tracker.log_params(model_id="qwen-7b", n_per_group=200)
    with tracker.stage("extraction"):
        for prompt in prompts:
            if tracker.is_cached(prompt.hash):
                continue
            features = extract(prompt)
            tracker.log_item(prompt.hash, features)
    tracker.log_metric("auroc", 0.85)

Decorator API:
    @tracked(tracker, cache_key=lambda prompt, **kw: hash(prompt))
    def extract_features(prompt, model, tokenizer):
        return do_extraction(prompt, model, tokenizer)

    @stage(tracker, "extraction", depends_on=["tokenization"])
    def run_extraction(config):
        ...

Not KV-cache-specific. Usable for any ML experiment pipeline.
"""

import functools
import hashlib
import json
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


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
        """Write metadata to disk."""
        with open(self._metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

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
        """Sanitize a cache key to prevent path traversal."""
        return key.replace("/", "_").replace("\\", "_").replace("..", "_")

    def log_item(self, key: str, data: Dict[str, Any]) -> None:
        """Log a per-item result to disk cache."""
        safe_key = self._sanitize_key(key)
        cache_path = self.cache_dir / f"{safe_key}.json"
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

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


# ================================================================
# DECORATOR API
# ================================================================

def tracked(
    tracker: ExperimentTracker,
    cache_key: Optional[Callable] = None,
    log_timing: bool = True,
    metric_name: Optional[str] = None,
):
    """Decorator that auto-caches function results, times execution, and logs to tracker.

    Args:
        tracker: ExperimentTracker instance to log to
        cache_key: function that takes the same args as the decorated function
                   and returns a string cache key. If None, no caching.
        log_timing: if True, logs wall-clock time as a metric
        metric_name: if set, logs the return value as this metric (must be numeric)

    Usage:
        @tracked(tracker, cache_key=lambda prompt, **kw: hash(prompt))
        def extract_features(prompt, model, tokenizer):
            return do_extraction(prompt, model, tokenizer)

        # First call: runs function, caches result
        # Second call with same args: returns cached result, skips function
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Check cache
            key = None
            if cache_key is not None:
                key = str(cache_key(*args, **kwargs))
                if tracker.is_cached(key):
                    return tracker.load_cached(key)

            # Run with timing
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            elapsed = time.monotonic() - t0

            # Log timing
            if log_timing:
                fn_name = fn.__name__
                tracker.log_metric(f"{fn_name}_seconds", round(elapsed, 3))

            # Cache result
            if key is not None and isinstance(result, dict):
                tracker.log_item(key, result)

            # Log as metric
            if metric_name is not None and isinstance(result, (int, float)):
                tracker.log_metric(metric_name, float(result))

            return result
        return wrapper
    return decorator


def stage_cache_key(name: str, config_hash: Optional[str] = None) -> str:
    """Build the cache key for a pipeline stage.

    Shared by the @stage decorator and anything that needs to write/read
    stage cache entries directly (e.g. tests that pre-populate cache).
    """
    suffix = f"_{config_hash}" if config_hash else ""
    return f"stage_{name}{suffix}"


def stage(
    tracker: ExperimentTracker,
    name: str,
    depends_on: Optional[List[str]] = None,
    skip_if_cached: bool = True,
    config_hash: Optional[str] = None,
):
    """Decorator that declares a pipeline stage with auto-timing and caching.

    Args:
        tracker: ExperimentTracker instance
        name: stage name (used for caching and logging)
        depends_on: list of stage names that must complete first
        skip_if_cached: if True, skip the stage if already completed
        config_hash: if provided, incorporated into the cache key so that
                     reruns with different config values invalidate the cache.
                     Note: old cache files from prior configs remain on disk.

    Usage:
        @stage(tracker, "extraction", depends_on=["tokenization"])
        def run_extraction(config):
            ...
            return {"n_items": 200}

        run_extraction(config)  # auto-timed, auto-cached, auto-skipped on restart
    """
    def decorator(fn: Callable) -> Callable:
        key = stage_cache_key(name, config_hash)
        dep_keys = {dep: stage_cache_key(dep, config_hash) for dep in (depends_on or [])}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Check cache
            if skip_if_cached and tracker.is_cached(key):
                cached = tracker.load_cached(key)
                if cached.get("status") == "complete":
                    return cached

            # Check dependencies
            for dep, dep_key in dep_keys.items():
                if not tracker.is_cached(dep_key):
                    raise RuntimeError(
                        f"Stage '{name}' depends on '{dep}' which has not completed"
                    )

            # Run with timing
            with tracker.stage(name):
                result = fn(*args, **kwargs)

            # Cache completion
            cache_data = {"status": "complete"}
            if isinstance(result, dict):
                cache_data.update(result)
            tracker.log_item(key, cache_data)

            return result
        return wrapper
    return decorator


def validated(
    tracker: ExperimentTracker,
    checks: Optional[List[Callable]] = None,
):
    """Decorator that runs validation checks before executing a function.

    Args:
        tracker: ExperimentTracker instance
        checks: list of callable validators. Each takes the same args as
                the decorated function. Must return True or raise ValueError.

    Usage:
        def check_prompts_exist(prompt_dir, **kw):
            if not prompt_dir.exists():
                raise ValueError(f"Prompt dir not found: {prompt_dir}")
            return True

        @validated(tracker, checks=[check_prompts_exist])
        def run_analysis(prompt_dir, config):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if checks:
                for check in checks:
                    try:
                        check(*args, **kwargs)
                    except (ValueError, AssertionError) as e:
                        tracker.log_metric(f"{fn.__name__}_validation_failed", 1)
                        raise
            return fn(*args, **kwargs)
        return wrapper
    return decorator
