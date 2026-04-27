"""CLI guardrails for --experiment exp3a.

Per pre-reg §3.4.1 the H3a runner is single-turn: intensity stimuli
deliver as the system message, no multi-turn conditioning, no neutral
buffer turns. The CLI rejects --condition and --neutral-turns for
exp3a so operators don't pass flags that look like they configure the
experiment but provably have no effect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _args_for_exp3a(**overrides):
    """Build a minimal argparse-like Namespace for the exp3a CLI guardrail check."""
    base = dict(
        experiment="exp3a",
        condition=None,
        neutral_turns=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_condition_rejected_for_exp3a():
    from src.cli import _check_exp3a_cli_compat

    args = _args_for_exp3a(condition="strong_positive")
    with pytest.raises(SystemExit) as exc:
        _check_exp3a_cli_compat(args)
    assert exc.value.code == 2


def test_neutral_turns_rejected_for_exp3a():
    from src.cli import _check_exp3a_cli_compat

    args = _args_for_exp3a(neutral_turns=5)
    with pytest.raises(SystemExit) as exc:
        _check_exp3a_cli_compat(args)
    assert exc.value.code == 2


def test_exp3a_with_no_conflicting_flags_passes():
    from src.cli import _check_exp3a_cli_compat

    args = _args_for_exp3a()
    _check_exp3a_cli_compat(args)


def test_other_experiments_not_constrained():
    from src.cli import _check_exp3a_cli_compat

    args = SimpleNamespace(
        experiment="exp1a",
        condition="strong_positive",
        neutral_turns=3,
    )
    _check_exp3a_cli_compat(args)
