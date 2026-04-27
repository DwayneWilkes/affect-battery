"""cmd_run constructs ExperimentConfig for exp3a without --condition.

The exp3a CLI guardrail rejects --condition (per pre-reg §3.4.1, the
runner ignores it). cmd_run must therefore not pass None to the
Condition() typed enum constructor.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


def _exp3a_run_args(tmp_path):
    """Minimal Namespace for cmd_run with --experiment exp3a."""
    return SimpleNamespace(
        experiment="exp3a",
        condition=None,
        neutral_turns=0,
        provider="openai",
        model="gpt-5.4-nano",
        base_model=False,
        base_url="http://localhost:8000/v1",
        num_runs=1,
        temperature=0.7,
        seed=42,
        bank=None,
        transfer_bank=str(tmp_path / "bank.yaml"),
        runner_config=str(tmp_path / "runner.yaml"),
        output_dir=str(tmp_path / "out"),
        dry_run=True,
        estimate=True,
        overwrite=False,
        max_concurrent=1,
        budget_max_calls=None,
        cost_per_call=None,
        rate_limit_rps=None,
        circuit_breaker_threshold=5,
        pre_registration_osf_url=None,
        pre_registration_github_commit=None,
        power_report_path=None,
        power_report_sha=None,
        skip_prereg_gate=True,
        skip_power_gate=True,
    )


def test_cmd_run_handles_none_condition_for_exp3a(tmp_path):
    """cmd_run must not pass None to Condition() when --experiment exp3a
    and --condition is omitted (the guardrail rejects --condition for
    exp3a, so this is the expected operator path)."""
    import yaml
    from src import cli

    bank_path = tmp_path / "bank.yaml"
    bank_path.write_text(yaml.safe_dump({"items": [
        {"id": f"item_{i:03d}", "question": f"q{i}", "expected": str(i)}
        for i in range(50)
    ]}))
    runner_cfg = tmp_path / "runner.yaml"
    runner_cfg.write_text(yaml.safe_dump({
        "intensity_levels": [1, 2, 3, 4, 5, 6, 7],
        "pilot_seed_path": "configs/intensity_pilot_seed.json",
    }))

    args = _exp3a_run_args(tmp_path)

    # --estimate path returns before any client construction or runner
    # dispatch; we just need cmd_run to not raise on the Condition()
    # call.
    with mock.patch("src.cli._print_estimate"):
        cli.cmd_run(args)
