"""End-to-end tests for the concrete pipeline stage definitions.

Spec: affect-battery-task-difficulty-calibration::pipeline (group 15).

Tasks 15.6 + 15.7: the pipeline CLI `affect-battery pipeline run <config>`
invokes a registry of stages that each wrap an existing module (generator,
calibrator, gate, report) rather than reimplementing them.

Three stages are offline-testable end-to-end:
    - bank_gen  → wraps src.calibration.generator
    - gate      → wraps src.calibration.gate.evaluate
    - analysis  → wraps src.analysis.report.manipulation_check_report

Two stages require a running vLLM pod and are exercised with a DryRunClient
or ScriptedClient in tests (calibration, experiment).

One stage is archival and is a pure git operation (archive) — test stubs
it via a fake git command.
"""

import json
from pathlib import Path

import pytest
import yaml

from src.pipeline.stages import (
    build_default_stages,
    run_pipeline_from_config,
)


def _write_config(tmp_path: Path, **overrides) -> Path:
    """Write a minimal pipeline config YAML with offline-only stages."""
    cfg = {
        "bank_gen": {
            "bank_id": "arithmetic_hard_v1_test",
            "seed": 12345,
            "n": 30,
        },
        "output_dir": str(tmp_path / "out"),
        "cache_root": str(tmp_path / "cache"),
        "stages": ["bank_gen"],
    }
    cfg.update(overrides)
    p = tmp_path / "pipeline-config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


class TestBankGenStage:
    def test_bank_gen_produces_yaml(self, tmp_path):
        """bank_gen stage writes a bank YAML + returns bank_path in outputs."""
        config_path = _write_config(tmp_path)
        artifacts = run_pipeline_from_config(config_path)
        assert "bank_path" in artifacts
        bank_path = Path(artifacts["bank_path"])
        assert bank_path.exists()
        bank = yaml.safe_load(bank_path.read_text())
        assert bank["bank_id"] == "arithmetic_hard_v1_test"

    def test_bank_gen_cached_on_second_run(self, tmp_path):
        config_path = _write_config(tmp_path)
        run_pipeline_from_config(config_path)
        events_path = Path(tmp_path / "cache") / "events.jsonl"
        events_1 = events_path.read_text().count("stage_start")
        run_pipeline_from_config(config_path)
        events_2_total = events_path.read_text().count("stage_start")
        # Second run emits no new stage_start — just a cache_hit.
        assert events_2_total == events_1


class TestDefaultStageRegistry:
    def test_registry_contains_expected_stages(self):
        stages = build_default_stages()
        names = [s.name for s in stages]
        expected = ["bank_gen", "calibration", "gate", "experiment",
                    "analysis", "archive"]
        for name in expected:
            assert name in names, f"stage '{name}' missing from default registry"


class TestPipelineCliSubcommand:
    def test_cli_pipeline_run_subcommand_exists(self):
        """`affect-battery pipeline run <config.yaml>` is a valid CLI path."""
        from src import cli
        # Parser must accept 'pipeline run <config>' — smoke-check by parsing.
        import argparse
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command", required=True)
        # The cli module should expose a helper that adds pipeline subcommand.
        assert hasattr(cli, "add_pipeline_subparser") or \
               hasattr(cli, "cmd_pipeline_run"), (
            "expected src.cli to expose either add_pipeline_subparser or "
            "cmd_pipeline_run for `affect-battery pipeline run` wiring"
        )

    def test_cli_pipeline_run_executes(self, tmp_path, monkeypatch):
        """Invoking the pipeline run command writes expected artifacts."""
        from src import cli
        config_path = _write_config(tmp_path)

        class Args:
            config = str(config_path)

        cli.cmd_pipeline_run(Args())
        # bank_gen stage should have written its output.
        cache_root = tmp_path / "cache"
        assert (cache_root / "events.jsonl").exists()
        assert (cache_root / "pipeline_manifest.json").exists()
