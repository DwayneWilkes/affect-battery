"""Task 0.7 Red — reclassify tdc-era runs with is_pilot: true.

Per design.md migration plan + GAPS.md G12: tdc-era result files (Qwen
runs from task-difficulty-calibration) get is_pilot: true added to
config.run, original checksum preserved in `.orig_checksum`. Original
file backed up to <name>.orig.

Implements migration script that walks a results dir, identifies
tdc-era results (config.model contains 'qwen' AND no is_pilot flag
yet), updates them safely.
"""

import json
import shutil
from pathlib import Path

import pytest

from src.migrations.reclassify_tdc_runs import (
    is_tdc_era_result,
    migrate_result_file,
    walk_and_migrate,
)


def _write_result(path: Path, model: str, is_pilot: bool | None = None) -> None:
    config: dict = {
        "model_name": model,
        "condition": "NEUTRAL",
        "experiment_type": "exp1a",
        "stimulus_bank": "arithmetic_easy_v1",
        "stimulus_bank_hash": "abc123",
    }
    if is_pilot is not None:
        config["is_pilot"] = is_pilot
    payload = {
        "config": config,
        "run_number": 0,
        "experiment_type": "exp1a",
        "model": model,
        "condition": "NEUTRAL",
        "conditioning_responses": [],
        "conditioning_correct": [],
        "transfer_responses": [],
        "transfer_questions": [],
        "transfer_expected": [],
        "start_time": 1.0,
        "end_time": 2.0,
        "checksum": "original-checksum",
    }
    path.write_text(json.dumps(payload))


class TestIsTdcEraResult:
    def test_qwen_no_is_pilot_is_tdc_era(self, tmp_path):
        path = tmp_path / "qwen_run.json"
        _write_result(path, model="Qwen/Qwen2.5-7B")
        assert is_tdc_era_result(path) is True

    def test_qwen_already_flagged_pilot_not_tdc_era(self, tmp_path):
        path = tmp_path / "qwen_already_flagged.json"
        _write_result(path, model="Qwen/Qwen2.5-7B", is_pilot=True)
        assert is_tdc_era_result(path) is False

    def test_llama_run_not_tdc_era(self, tmp_path):
        path = tmp_path / "llama_run.json"
        _write_result(path, model="meta-llama/Llama-3-8B-Instruct")
        assert is_tdc_era_result(path) is False


class TestMigrateResultFile:
    def test_migration_adds_is_pilot_true(self, tmp_path):
        path = tmp_path / "qwen_run.json"
        _write_result(path, model="Qwen/Qwen2.5-7B")
        migrate_result_file(path)
        data = json.loads(path.read_text())
        assert data["config"]["is_pilot"] is True

    def test_migration_preserves_orig_checksum(self, tmp_path):
        path = tmp_path / "qwen_run.json"
        _write_result(path, model="Qwen/Qwen2.5-7B")
        migrate_result_file(path)
        data = json.loads(path.read_text())
        assert data.get("orig_checksum") == "original-checksum"

    def test_migration_creates_orig_backup(self, tmp_path):
        path = tmp_path / "qwen_run.json"
        _write_result(path, model="Qwen/Qwen2.5-7B")
        migrate_result_file(path)
        backup = path.with_suffix(".orig")
        assert backup.exists()
        backup_data = json.loads(backup.read_text())
        # Original file unchanged
        assert "is_pilot" not in backup_data["config"]

    def test_migration_idempotent_skips_already_flagged(self, tmp_path):
        path = tmp_path / "qwen_run.json"
        _write_result(path, model="Qwen/Qwen2.5-7B", is_pilot=True)
        migrate_result_file(path)  # should be a no-op
        # No backup created if no migration needed
        backup = path.with_suffix(".orig")
        assert not backup.exists()


class TestWalkAndMigrate:
    def test_walk_migrates_only_tdc_era_files(self, tmp_path):
        qwen_path = tmp_path / "qwen.json"
        llama_path = tmp_path / "llama.json"
        _write_result(qwen_path, model="Qwen/Qwen2.5-7B")
        _write_result(llama_path, model="meta-llama/Llama-3-8B-Instruct")

        migrated = walk_and_migrate(tmp_path)

        assert qwen_path.name in [p.name for p in migrated]
        assert llama_path.name not in [p.name for p in migrated]

        qwen_data = json.loads(qwen_path.read_text())
        llama_data = json.loads(llama_path.read_text())
        assert qwen_data["config"]["is_pilot"] is True
        assert "is_pilot" not in llama_data["config"]

    def test_walk_handles_empty_directory(self, tmp_path):
        migrated = walk_and_migrate(tmp_path)
        assert migrated == []
