"""Concrete pipeline stage definitions + `run_pipeline_from_config` entry.

Each stage is a thin wrapper around an existing module rather than a
reimplementation. The domain-agnostic PipelineRunner (src.pipeline.runner)
composes them; this file only defines the stages and their run_fns.

Canonical stage set (order matters — DAG is linear):
    1. bank_gen     — wraps src.calibration.generator
    2. calibration  — wraps scripts/auto_calibrate_arithmetic.py (POD)
    3. gate         — wraps src.calibration.gate.evaluate
    4. experiment   — wraps src.runner.run_batch (POD)
    5. analysis     — wraps src.analysis.report.manipulation_check_report
    6. archive      — git commit + tag of the produced artifacts

For offline testing, stages 2 and 4 are stubs that expect the caller to
pass `dry_run: true` in config (exercised via DryRunClient).

Spec: affect-battery-task-difficulty-calibration::pipeline (group 15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.pipeline.runner import PipelineRunner, Stage


# ───────────────────────── stage run_fns ──────────────────────────


def _bank_gen_run(config: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Wrap src.calibration.generator.generate_items + build_bank_yaml + write_bank."""
    from src.calibration.generator import (
        build_bank_yaml,
        generate_items,
        write_bank,
    )

    gen_cfg = config.get("bank_gen", {})
    bank_id = gen_cfg.get("bank_id", "arithmetic_hard_v1")
    seed = gen_cfg.get("seed", 20260421)
    n = gen_cfg.get("n", 300)

    items = generate_items(total=n, rng_seed=seed, id_prefix=bank_id)
    bank = build_bank_yaml(items=items, bank_id=bank_id)

    output_dir = Path(config.get("output_dir", "./out"))
    bank_path = output_dir / "banks" / f"{bank_id}.yaml"
    write_bank(bank, bank_path)
    return {"bank_path": str(bank_path), "bank_id": bank_id}


def _calibration_run(config: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """POD-required in production; dry_run=True uses DryRunClient for offline smoke.

    Real calibration uses scripts/auto_calibrate_arithmetic.py. This stage
    just records the invocation intent; the actual calibration is driven by
    a separate pod-side script that writes to the artifacts directory.
    """
    cal_cfg = config.get("calibration", {})
    if not config.get("dry_run", True):
        raise NotImplementedError(
            "calibration stage requires pod integration. Run "
            "scripts/auto_calibrate_arithmetic.py manually and re-invoke "
            "the pipeline with dry_run=True to skip this stage."
        )
    return {
        "calibration_report_path": cal_cfg.get("report_path", ""),
        "calibration_dry_run": True,
    }


def _gate_run(config: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Wrap src.calibration.gate.load_config. Full evaluate() requires real
    calibration results; in dry_run mode we stop at config-load + hash.
    """
    from src.calibration.gate import load_config

    gate_cfg_path = config.get("gate_config_path")
    if not gate_cfg_path:
        return {"gate_verdict_status": "skipped", "gate_config_hash": ""}
    gc = load_config(Path(gate_cfg_path))
    return {
        "gate_verdict_status": "deferred_to_eval",
        "gate_config_hash": gc.config_hash,
    }


def _experiment_run(config: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """POD-required in production. dry_run=True records intent only."""
    if not config.get("dry_run", True):
        raise NotImplementedError(
            "experiment stage requires pod integration. Run `affect-battery "
            "run` or `affect-battery pilot` manually; then re-invoke the "
            "pipeline in dry_run mode to orchestrate analysis."
        )
    return {"experiment_dry_run": True, "results_dir": config.get("results_dir", "")}


def _analysis_run(config: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Produce a manipulation_check_report from results in the results_dir.

    In dry_run mode with no results, this is a no-op that emits an empty
    report so the pipeline completes end-to-end.
    """
    from src.analysis.report import manipulation_check_report

    results_dir = upstream.get("results_dir") or config.get("results_dir", "")
    if not results_dir:
        return {"analysis_skipped": True}
    # Real wiring: load every result JSON, group by model, compute manipulation
    # check per model, feed into manipulation_check_report. For skeleton-level
    # integration we just produce an empty report structure.
    report = manipulation_check_report(results=[])
    return {"analysis_row_count": len(report.rows)}


def _archive_run(config: dict[str, Any], upstream: dict[str, Any]) -> dict[str, Any]:
    """Stub: archive stage would git-tag the artifacts. In dry_run it just
    records the intended tag name."""
    arc_cfg = config.get("archive", {})
    tag = arc_cfg.get("tag", "")
    return {"archive_tag": tag}


# ───────────────────────── stage registry ──────────────────────────


def build_default_stages() -> list[Stage]:
    """Return the canonical 6-stage pipeline in execution order."""
    return [
        Stage(
            name="bank_gen",
            inputs=("bank_gen",),
            outputs=("bank_path", "bank_id"),
            run_fn=_bank_gen_run,
        ),
        Stage(
            name="calibration",
            inputs=("calibration", "bank_id", "dry_run"),
            outputs=("calibration_report_path", "calibration_dry_run"),
            run_fn=_calibration_run,
        ),
        Stage(
            name="gate",
            inputs=("gate_config_path", "calibration_report_path"),
            outputs=("gate_verdict_status", "gate_config_hash"),
            run_fn=_gate_run,
        ),
        Stage(
            name="experiment",
            inputs=("experiment", "bank_id", "gate_verdict_status", "dry_run"),
            outputs=("experiment_dry_run", "results_dir"),
            run_fn=_experiment_run,
        ),
        Stage(
            name="analysis",
            inputs=("analysis", "results_dir"),
            outputs=("analysis_row_count", "analysis_skipped"),
            run_fn=_analysis_run,
        ),
        Stage(
            name="archive",
            inputs=("archive", "analysis_row_count"),
            outputs=("archive_tag",),
            run_fn=_archive_run,
        ),
    ]


def _filter_stages_to_requested(
    stages: list[Stage],
    requested_names: list[str] | None,
) -> list[Stage]:
    """If the config lists specific stages to run, filter the registry to
    that subset (preserving registry order). None/empty list = run all."""
    if not requested_names:
        return stages
    requested = set(requested_names)
    return [s for s in stages if s.name in requested]


def run_pipeline_from_config(config_path: Path | str) -> dict[str, Any]:
    """Top-level entry: load pipeline config, build default stages (filtered
    by config.stages if provided), run, return accumulated artifacts."""
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())

    cache_root = Path(config.get("cache_root", "./cache"))
    all_stages = build_default_stages()
    stages_to_run = _filter_stages_to_requested(all_stages, config.get("stages"))

    runner = PipelineRunner(stages=stages_to_run, cache_root=cache_root)
    return runner.run(config)
