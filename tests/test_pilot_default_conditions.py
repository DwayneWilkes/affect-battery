"""Tests for the default pilot conditions list.

Spec: affect-battery-task-difficulty-calibration::conditioning-protocol::
"Default pilot conditions list includes SELF_CHECK_NEUTRAL".

Tasks 7.1 + 7.2 from
`specs`:
the pilot subcommand's default conditions list MUST include all seven
conditions (the existing six + SELF_CHECK_NEUTRAL), so the length-matched
control runs alongside STRONG_NEGATIVE without requiring a per-invocation
override.
"""

import pytest

from src.conditioning.prompts import Condition


EXPECTED_DEFAULT_PILOT_CONDITIONS = {
    Condition.STRONG_POSITIVE,
    Condition.MILD_NEGATIVE,
    Condition.STRONG_NEGATIVE,
    Condition.NEUTRAL,
    Condition.NO_CONDITIONING,
    Condition.ACCURATE_NEGATIVE,
    Condition.SELF_CHECK_NEUTRAL,
}


class TestDefaultPilotConditionsConstant:
    def test_default_pilot_conditions_exposed_as_module_constant(self):
        """The default conditions list MUST be exposed as `DEFAULT_PILOT_CONDITIONS`
        in src.cli so tests (and the pipeline orchestrator) can read it without
        invoking the CLI."""
        from src import cli
        assert hasattr(cli, "DEFAULT_PILOT_CONDITIONS"), (
            "Expected src.cli.DEFAULT_PILOT_CONDITIONS to be exposed as "
            "a module-level constant"
        )

    def test_default_pilot_conditions_has_all_seven(self):
        from src import cli
        assert set(cli.DEFAULT_PILOT_CONDITIONS) == EXPECTED_DEFAULT_PILOT_CONDITIONS

    def test_default_pilot_conditions_has_exactly_seven(self):
        from src import cli
        assert len(cli.DEFAULT_PILOT_CONDITIONS) == 7


class TestCmdPilotUsesSevenConditionsByDefault:
    """Cmd_pilot invoked with no --conditions override iterates the 7-condition
    default. Exercised by monkey-patching the batch runner and capturing what
    conditions the CLI iterates over."""

    def test_cmd_pilot_iterates_all_seven_conditions(self, monkeypatch, tmp_path):
        from src import cli

        captured_conditions: list = []

        class _NoOpClient:
            def __init__(self, *a, **kw):
                pass

            async def close(self):
                pass

        # Stub out the real VLLM clients so we don't try to talk to a pod.
        monkeypatch.setattr("src.cli.VLLMClient", _NoOpClient, raising=False)
        monkeypatch.setattr("src.cli.VLLMCompletionClient", _NoOpClient, raising=False)

        # Intercept the per-experiment runner dispatch so we capture each
        # condition cmd_pilot constructs an ExperimentConfig for. cmd_pilot
        # resolves the runner via `RUNNERS[args.experiment]` (post the
        # multi-experiment dispatch refactor), so patching run_batch on
        # src.runner alone misses the dispatch path. Patching the entry
        # in the RUNNERS dict is the principled hook.
        async def fake_runner(config, *args, **kwargs):
            captured_conditions.append(config.condition)
            return
            yield  # makes this an async generator

        from src.runners import RUNNERS as _RUNNERS
        # Patch every experiment slot so the test works regardless of
        # which --experiment was passed.
        for k in list(_RUNNERS):
            monkeypatch.setitem(_RUNNERS, k, fake_runner)

        # Drive asyncio.run without actually scheduling. We want the body to
        # run to completion on our fake coroutines.
        import asyncio

        orig_run = asyncio.run

        def _fake_asyncio_run(coro):
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(coro)
                finally:
                    loop.close()
            except Exception:
                coro.close()

        monkeypatch.setattr("asyncio.run", _fake_asyncio_run)

        class Args:
            dry_run = True
            base_model = False
            provider = "vllm"
            model = "test-model"
            experiment = "exp1a"
            base_url = "http://unused"
            num_runs = 5
            seed = 42
            temperature = 0.7
            output_dir = str(tmp_path)
            bank = None
            max_concurrent = 1
            budget_max_calls = None
            cost_per_call = None
            rate_limit_rps = None
            circuit_breaker_threshold = 5
            # Pre-reg gates bypassed by --dry-run; values irrelevant.
            pre_registration_osf_url = None
            pre_registration_github_commit = None
            power_report_path = None
            power_report_sha = None
            skip_prereg_gate = False
            skip_power_gate = False

        cli.cmd_pilot(Args())

        # Every condition the pilot runs should be in the expected 7-set, and
        # the set of conditions actually run should equal the expected 7-set.
        assert set(captured_conditions) == EXPECTED_DEFAULT_PILOT_CONDITIONS, (
            f"cmd_pilot iterated over {set(captured_conditions)}, "
            f"expected {EXPECTED_DEFAULT_PILOT_CONDITIONS}"
        )
