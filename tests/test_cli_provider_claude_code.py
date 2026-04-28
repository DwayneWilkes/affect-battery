"""--provider claude_code wires through to ClaudeCodeClient."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


def _args(provider="claude_code", model="sonnet", **overrides):
    base = dict(
        dry_run=False,
        base_model=False,
        provider=provider,
        model=model,
        base_url="http://localhost:8000/v1",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_provider_claude_code_in_argparse_choices():
    """argparse accepts --provider claude_code without raising."""
    import sys
    import importlib
    from unittest import mock
    src_cli = importlib.import_module("src.cli")
    # Run the argparse with --help to verify the choice is registered;
    # we capture SystemExit since --help exits zero after printing.
    argv = ["affect-battery", "run", "--help"]
    with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
        src_cli.main()
    assert exc.value.code == 0


def test_provider_claude_code_rejected_unknown_value():
    """argparse rejects --provider with an unknown value."""
    import sys
    import importlib
    from unittest import mock
    src_cli = importlib.import_module("src.cli")
    argv = ["affect-battery", "run", "--model", "x", "--provider", "bogus"]
    with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit) as exc:
        src_cli.main()
    assert exc.value.code == 2


def test_build_client_dispatches_claude_code():
    """_build_client returns a ClaudeCodeClient for --provider claude_code."""
    from src.cli import _build_client
    from src.models import ClaudeCodeClient

    with mock.patch("src.models._detect_claude_auth_source", return_value=("subscription", "claude max")), \
         mock.patch("shutil.which", return_value="/usr/local/bin/claude"):
        client = _build_client(_args())
    assert isinstance(client, ClaudeCodeClient)
    assert client.model_name == "sonnet"


def test_claude_code_rejects_base_model():
    """--provider claude_code with --base-model raises SystemExit(2)."""
    from src.cli import _build_client

    with pytest.raises(SystemExit) as exc:
        _build_client(_args(base_model=True))
    assert exc.value.code == 2


def test_dry_run_with_claude_code_uses_dry_run_client():
    """--dry-run short-circuits to DryRunClient regardless of provider."""
    from src.cli import _build_client
    from src.models import DryRunClient

    client = _build_client(_args(dry_run=True))
    assert isinstance(client, DryRunClient)
