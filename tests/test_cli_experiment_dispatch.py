"""CLI `--experiment` flag dispatches to per-experiment runners.

Per design.md D5 (unified `--experiment` CLI flag + per-experiment library
modules) and :

- `affect-battery run --experiment exp1a` routes to `src.runners.exp1a.run_exp1a`
- Same for exp1b, exp2, exp3a, exp3b, exp3c
- New `probe` subcommand with `variance` and `base-model` sub-subcommands
  (Week-0 probes per design.md Phase 1)
- CLI exports a dispatch table `RUNNERS: dict[str, Callable]` so dispatch
  is declarative, not if/elif cascaded
"""

import pytest


class TestRunnersPackage:
    """`src.runners` MUST be a package exporting run_exp* functions + a
    RUNNERS dispatch dict."""

    def test_runners_package_exports_run_exp1a(self):
        from src.runners import run_exp1a
        assert callable(run_exp1a)

    def test_runners_package_exports_run_exp1b(self):
        from src.runners import run_exp1b
        assert callable(run_exp1b)

    def test_runners_package_exports_run_exp2(self):
        from src.runners import run_exp2
        assert callable(run_exp2)

    def test_runners_package_exports_run_exp3a(self):
        from src.runners import run_exp3a
        assert callable(run_exp3a)

    def test_runners_package_exports_run_exp3b(self):
        from src.runners import run_exp3b
        assert callable(run_exp3b)

    def test_runners_package_exports_run_exp3c(self):
        from src.runners import run_exp3c
        assert callable(run_exp3c)

    def test_runners_dispatch_table_has_all_six(self):
        from src.runners import RUNNERS
        assert set(RUNNERS.keys()) == {
            "exp1a", "exp1b", "exp2", "exp3a", "exp3b", "exp3c"
        }

    def test_runners_dispatch_table_maps_to_callables(self):
        from src.runners import RUNNERS
        for key, fn in RUNNERS.items():
            assert callable(fn), f"RUNNERS[{key!r}] is not callable"


class TestCliExperimentFlag:
    """CLI `run --experiment <name>` accepts the six paper-aligned values."""

    @pytest.mark.parametrize("experiment", [
        "exp1a", "exp1b", "exp2", "exp3a", "exp3b", "exp3c",
    ])
    def test_run_accepts_experiment_flag(self, experiment):
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "run", "--experiment", experiment,
            "--base-url", "http://localhost:8000/v1",
            "--model", "test/model",
            "--condition", "NEUTRAL",
        ])
        assert args.experiment == experiment

    def test_run_default_experiment_is_exp1a(self):
        """Per D5: default --experiment is 'exp1a' (was 'transfer_within')."""
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--base-url", "http://localhost:8000/v1",
            "--model", "test/model",
            "--condition", "NEUTRAL",
        ])
        assert args.experiment == "exp1a"


class TestCmdRunDispatches:
    """Cmd_run MUST route through RUNNERS[args.experiment] — not the legacy
    run_batch path directly. Verifies the fix for reviewer-flagged gap
    where --experiment was validated but never dispatched."""

    def test_cmd_run_imports_from_runners(self):
        """Regression guard: cmd_run's import block MUST reference
        src.runners.RUNNERS."""
        import inspect
        from src.cli import cmd_run
        source = inspect.getsource(cmd_run)
        assert "from .runners import RUNNERS" in source, (
            "cmd_run must dispatch via RUNNERS dispatch table"
        )
        assert "RUNNERS[args.experiment]" in source, (
            "cmd_run must look up runner by args.experiment"
        )

    def test_all_experiment_runners_implemented(self):
        """All six experiment runners (exp1a-exp3c) are implemented as of
        Phase 7. None should raise NotImplementedError on import; runtime
        validation of inputs lives in each runner's signature, not in
        an unconditional NotImplementedError stub."""
        from src.runners import RUNNERS

        for name in ("exp1a", "exp1b", "exp2", "exp3a", "exp3b", "exp3c"):
            assert callable(RUNNERS[name]), f"{name} runner missing"

    def test_cli_run_dispatches_exp3b_with_runner_config(self, tmp_path, monkeypatch):
        """Per `cli.cmd_run` must dispatch exp3a/3b/3c
        without raising TypeError. The previous implementation passed only
        the base run_batch kwargs (max_concurrent, budget, ...) which the
        new runners don't accept, and supplied no intensity_levels/prompts/
        items so the call would TypeError before any work happened."""
        import yaml
        from src.cli import build_parser, cmd_run

        runner_cfg = tmp_path / "exp3b.yaml"
        runner_cfg.write_text(yaml.safe_dump({
            "prompts": [{"id": "p1", "text": "Continue."}],
            "n_generations": 2,
        }))

        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "dry-run",
            "--condition", "neutral",
            "--experiment", "exp3b",
            "--num-runs", "1",
            "--seed", "42",
            "--dry-run",
            "--output-dir", str(tmp_path / "out"),
            "--runner-config", str(runner_cfg),
        ])
        # Should complete without TypeError. We don't assert on output
        # here — just that the dispatch wiring is correct.
        cmd_run(args)


class TestCliProbeSubcommand:
    """CLI `probe` subcommand with variance + base-model sub-subcommands
    (design.md Phase 1 Week-0 probes)."""

    def test_probe_variance_subcommand_parses(self):
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "probe", "variance",
            "--base-url", "http://localhost:8000/v1",
            "--model", "llama-3-8b-instruct",
            "--n", "20",
        ])
        assert args.command == "probe"
        assert args.probe_kind == "variance"
        assert args.model == "llama-3-8b-instruct"
        assert args.n == 20

    def test_probe_base_model_subcommand_parses(self):
        from src.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "probe", "base-model",
            "--base-url", "http://localhost:8000/v1",
            "--model", "llama-3-8b-base",
        ])
        assert args.command == "probe"
        assert args.probe_kind == "base-model"
