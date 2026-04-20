"""Tests for the MLflow-compatible experiment tracker.

Spec (GAPS.md task 10): mirror the ll/KV-Cache MLOps pattern before RunPod
spend. Tracker must:
- Log params per run (10.2): model, condition, intensity_level, intensity_set,
  seed, temperature, num_runs, num_conditioning_turns.
- Log metrics per run (10.3): accuracy, hedging_per_100w per category,
  diversity, confidence_mean.
- Log artifacts per run (10.4): result JSON + SHA-256 checksum.
- Deterministic run naming (10.5): {model_slug}_{condition}_{experiment_type}_
  {seed}_{run_number}.
- Idempotent re-entry (10.6): same config -> same run name, no double-log.
- Offline loadability (10.7): tracker state is a standalone directory that can
  be read back without network / mlflow server.
"""

import hashlib
import json
from pathlib import Path

import pytest

from src.tracking import ExperimentTracker, run_name_for


def _config(**overrides) -> dict:
    defaults = dict(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        condition="strong_positive",
        experiment_type="transfer_within",
        seed=42,
        run_number=3,
        temperature=0.7,
        num_runs=50,
        num_conditioning_turns=5,
        intensity_set="primary_valence_axis",
        intensity_level=None,
    )
    defaults.update(overrides)
    return defaults


class TestRunNameForDeterminism:
    def test_same_config_same_name(self):
        assert run_name_for(_config()) == run_name_for(_config())

    def test_name_template_fields(self):
        name = run_name_for(_config())
        assert "Meta-Llama-3-8B-Instruct" in name
        assert "strong_positive" in name
        assert "transfer_within" in name
        assert "42" in name  # seed
        assert "0003" in name  # run_number zero-padded

    def test_different_runs_differ(self):
        a = run_name_for(_config(run_number=0))
        b = run_name_for(_config(run_number=1))
        assert a != b

    def test_model_slug_strips_provider_prefix(self):
        name = run_name_for(_config(model_name="meta-llama/Meta-Llama-3-8B-Instruct"))
        # Just the final segment of the path should appear, not 'meta-llama/'.
        assert "meta-llama/" not in name


class TestTrackerParamsAndMetrics:
    def test_start_run_creates_run_dir(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        cfg = _config()
        tracker.start_run(cfg)
        run_dir = tmp_path / "mlruns" / tracker.current_run_name
        assert run_dir.is_dir()
        tracker.end_run()

    def test_log_params_persists(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker.start_run(_config())
        tracker.log_params(model="test", temperature=0.5)
        tracker.end_run()
        meta = json.loads(
            (tmp_path / "mlruns" / tracker.current_run_name / "run_metadata.json").read_text()
        )
        assert meta["params"]["model"] == "test"
        assert meta["params"]["temperature"] == 0.5

    def test_log_metrics_persists(self, tmp_path):
        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker.start_run(_config())
        tracker.log_metrics(accuracy=0.82, hedging_per_100w_epistemic=1.4)
        tracker.end_run()
        meta = json.loads(
            (tmp_path / "mlruns" / tracker.current_run_name / "run_metadata.json").read_text()
        )
        assert meta["metrics"]["accuracy"] == 0.82
        assert meta["metrics"]["hedging_per_100w_epistemic"] == 1.4


class TestTrackerArtifacts:
    def test_log_artifact_copies_file_and_records_checksum(self, tmp_path):
        artifact = tmp_path / "result.json"
        artifact.write_text('{"run_number": 0}')
        expected_checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()

        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker.start_run(_config())
        tracker.log_artifact(artifact)
        tracker.end_run()

        run_dir = tmp_path / "mlruns" / tracker.current_run_name
        artifacts_dir = run_dir / "artifacts"
        copied = artifacts_dir / "result.json"
        assert copied.exists()
        # Checksums should be recorded in metadata.
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        assert "artifacts" in meta
        recorded = meta["artifacts"]["result.json"]
        assert recorded["sha256"] == expected_checksum


class TestIdempotency:
    def test_reentry_reuses_same_run_name(self, tmp_path):
        """Task 10.6: re-entry with identical config resolves to same run."""
        cfg = _config()
        tracker_a = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker_a.start_run(cfg)
        name_a = tracker_a.current_run_name
        tracker_a.end_run()

        tracker_b = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker_b.start_run(cfg)
        name_b = tracker_b.current_run_name
        tracker_b.end_run()

        assert name_a == name_b

    def test_reentry_does_not_double_log_params(self, tmp_path):
        """Same run, logging params twice with same values -> no duplicates."""
        cfg = _config()
        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker.start_run(cfg)
        tracker.log_params(model="test")
        tracker.log_params(model="test")  # idempotent
        tracker.end_run()

        tracker2 = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker2.start_run(cfg)
        tracker2.log_params(model="test")
        tracker2.end_run()

        meta = json.loads(
            (tmp_path / "mlruns" / tracker.current_run_name / "run_metadata.json").read_text()
        )
        # params dict remains a dict of single values, not a list.
        assert meta["params"]["model"] == "test"


class TestOfflineLoadability:
    def test_run_metadata_is_standalone(self, tmp_path):
        """Task 10.7 (functional slice): every run directory is a self-contained
        JSON store that can be read without MLflow / network."""
        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker.start_run(_config())
        tracker.log_params(model="dry")
        tracker.log_metrics(accuracy=0.9)
        tracker.end_run()

        run_dir = tmp_path / "mlruns" / tracker.current_run_name
        meta_path = run_dir / "run_metadata.json"
        assert meta_path.exists()
        # Fully loadable JSON with expected top-level keys.
        meta = json.loads(meta_path.read_text())
        for key in ("run_name", "config", "params", "metrics", "artifacts",
                    "started_at", "ended_at"):
            assert key in meta, f"Missing top-level key: {key}"


class TestConfigCapture:
    def test_start_run_captures_full_config(self, tmp_path):
        """Task 10.2: every param we care about is serialised on start."""
        cfg = _config()
        tracker = ExperimentTracker(output_dir=tmp_path / "mlruns")
        tracker.start_run(cfg)
        tracker.end_run()
        meta = json.loads(
            (tmp_path / "mlruns" / tracker.current_run_name / "run_metadata.json").read_text()
        )
        for field in (
            "model_name", "condition", "experiment_type", "seed", "temperature",
            "num_runs", "num_conditioning_turns", "intensity_set",
        ):
            assert meta["config"].get(field) == cfg[field], (
                f"Config field {field} not captured correctly"
            )
