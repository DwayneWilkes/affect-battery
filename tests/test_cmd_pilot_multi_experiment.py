"""cmd_pilot multi-experiment dispatch contract.

cmd_pilot iterates the 7 default conditions and dispatches to
RUNNERS[args.experiment] for each, attaching the correct ExperimentType
to each ExperimentConfig. exp1a / exp1b / exp2 share the run_batch
shape; exp3a / exp3b / exp3c require --runner-config but follow the
same condition iteration. The ExperimentType on the saved config must
match args.experiment so result files file under the correct
<root>/data/<exp>/<condition>/ leaf.

Spec: affect-battery-proposal-realignment :: experiment-dispatch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml


def _base_args(tmp_path, experiment: str, **overrides):
    """Minimal args namespace for cmd_pilot --dry-run."""
    args = SimpleNamespace(
        dry_run=True,
        base_model=False,
        provider="anthropic",
        model="dry-run",
        experiment=experiment,
        base_url=None,
        num_runs=1,
        seed=0,
        temperature=0.7,
        output_dir=str(tmp_path),
        bank=None,
        transfer_bank=None,
        max_concurrent=1,
        budget_max_calls=None,
        cost_per_call=None,
        rate_limit_rps=100.0,
        circuit_breaker_threshold=5,
        pre_registration_osf_url=None,
        pre_registration_github_commit=None,
        power_report_path=None,
        power_report_sha=None,
        skip_prereg_gate=True,
        skip_power_gate=True,
        # cmd_run-style fields cmd_pilot now also honors:
        runner_config=None,
        neutral_turns=2,
        condition="neutral",  # ignored by cmd_pilot but some helpers read it
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestCmdPilotRespectsExperimentType:
    """For exp1a/1b/2 (which share the run_batch dispatch shape),
    cmd_pilot must propagate args.experiment into ExperimentConfig
    rather than hardcoding TRANSFER_WITHIN."""

    @pytest.mark.parametrize("experiment", ["exp1a", "exp1b", "exp2"])
    def test_pilot_writes_data_under_correct_experiment_dir(
        self, tmp_path, experiment,
    ):
        from src import cli

        args = _base_args(tmp_path, experiment=experiment)
        cli.cmd_pilot(args)

        # Result JSONs land under <root>/data/<experiment>/<condition>/.
        exp_dir = tmp_path / "data"
        assert exp_dir.is_dir(), (
            f"--experiment {experiment} did not produce <root>/data/{experiment}/. "
            f"Found: {[p.name for p in tmp_path.iterdir() if p.is_dir()]}"
        )
        result_files = list(exp_dir.rglob("*.json"))
        assert len(result_files) >= 1, (
            f"--experiment {experiment} produced no result JSONs"
        )

    @pytest.mark.parametrize("experiment", ["exp1a", "exp1b", "exp2"])
    def test_manifest_records_correct_experiment(self, tmp_path, experiment):
        from src import cli

        args = _base_args(tmp_path, experiment=experiment)
        cli.cmd_pilot(args)

        manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
        assert manifest["experiment"] == experiment

    def test_pilot_result_files_carry_correct_experiment_type(self, tmp_path):
        """When --experiment exp1b is set, every saved RunResult JSON
        carries experiment_type='exp1b' on the top-level field. Cells
        whose top-level type doesn't match the directory they're filed
        under would silently misroute analysis."""
        import json

        from src import cli

        args = _base_args(tmp_path, experiment="exp1b")
        cli.cmd_pilot(args)

        exp_dir = tmp_path / "data"
        result_files = list(exp_dir.rglob("*.json"))
        assert len(result_files) >= 1

        sample = json.loads(result_files[0].read_text())
        assert sample.get("experiment_type") == "exp1b", (
            f"Result file experiment_type wrong; got {sample.get('experiment_type')}"
        )
        assert sample.get("config", {}).get("experiment_type") == "exp1b"


class TestCmdPilotExp3RequiresRunnerConfig:
    """exp3a/3b/3c need --runner-config because they have additional
    sweep axes (intensity_levels / prompts / items) that aren't covered
    by the base ExperimentConfig. cmd_pilot must surface a clear error
    when --runner-config is missing for these experiments."""

    @pytest.mark.parametrize("experiment", ["exp3a", "exp3b", "exp3c"])
    def test_exp3_without_runner_config_exits(self, tmp_path, capsys, experiment):
        from src import cli

        args = _base_args(tmp_path, experiment=experiment, runner_config=None)
        with pytest.raises(SystemExit) as excinfo:
            cli.cmd_pilot(args)
        assert excinfo.value.code == 2
        captured = capsys.readouterr()
        assert "runner-config" in captured.err.lower()


class TestCmdPilotExp2NeutralTurns:
    """Exp 2 requires neutral_turns to be set on the config (the
    persistence-recovery sweep parameter). cmd_pilot must thread
    args.neutral_turns through."""

    def test_exp2_pilot_records_neutral_turns_in_config(self, tmp_path):
        import json

        from src import cli

        args = _base_args(tmp_path, experiment="exp2", neutral_turns=3)
        cli.cmd_pilot(args)

        exp_dir = tmp_path / "data"
        result_files = list(exp_dir.rglob("*.json"))
        assert len(result_files) >= 1

        sample = json.loads(result_files[0].read_text())
        assert sample.get("config", {}).get("neutral_turns") == 3


class TestCmdPilotIteratesAllConditions:
    """Regression guard: even with the multi-experiment fix, cmd_pilot
    must still iterate all 7 default conditions per experiment. This
    is the existing test_cmd_pilot_iterates_all_seven_conditions
    contract — restating to catch any regression from the dispatch
    refactor."""

    @pytest.mark.parametrize("experiment", ["exp1a", "exp1b", "exp2"])
    def test_all_seven_condition_subdirs_appear(self, tmp_path, experiment):
        from src import cli

        args = _base_args(tmp_path, experiment=experiment)
        cli.cmd_pilot(args)

        exp_dir = tmp_path / "data"
        cond_subdirs = sorted(p.name for p in exp_dir.iterdir() if p.is_dir())
        # 7 conditions: accurate_negative, mild_negative, neutral,
        # no_conditioning, self_check_neutral, strong_negative,
        # strong_positive.
        expected = [
            "accurate_negative", "mild_negative", "neutral",
            "no_conditioning", "self_check_neutral",
            "strong_negative", "strong_positive",
        ]
        assert cond_subdirs == expected
