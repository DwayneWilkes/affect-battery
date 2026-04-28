"""_write_pilot_manifest records inference-backend metadata when present.

Specifically, when the client carries auth_source, total_cost_usd, or
params_unhonored attributes, the manifest writer records them under
inference_auth_source, inference_total_cost_usd, and
inference_params_unhonored. ClaudeCodeClient is the current source of
these attributes; OpenAIClient / AnthropicClient / VLLMClient do not
expose them.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml


def _args(**overrides):
    base = dict(
        experiment="exp1a",
        model="dry-run",
        provider="claude_code",
        num_runs=1,
        seed=42,
        temperature=0.7,
        base_model=False,
        transfer_bank=None,
        pre_registration_osf_url=None,
        pre_registration_github_commit=None,
        power_report_path=None,
        power_report_sha=None,
        skip_prereg_gate=True,
        skip_power_gate=True,
        dry_run=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_manifest_records_claude_code_metadata(tmp_path):
    from src.cli import _write_pilot_manifest

    fake_client = SimpleNamespace(
        auth_source="subscription",
        total_cost_usd=0.42,
        params_unhonored=False,
    )
    _write_pilot_manifest(
        pilot_root=tmp_path,
        args=_args(),
        conditions=[],
        bank_id="b1",
        bank_hash="deadbeef",
        started_utc="2026-04-27T20:00:00+00:00",
        completed_utc="2026-04-27T20:01:00+00:00",
        per_cond_elapsed={},
        per_cond_count={},
        client=fake_client,
    )
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["inference_auth_source"] == "subscription"
    assert manifest["inference_total_cost_usd"] == 0.42


def test_manifest_records_params_unhonored_when_set(tmp_path):
    from src.cli import _write_pilot_manifest

    fake_client = SimpleNamespace(
        auth_source="api",
        total_cost_usd=0.0,
        params_unhonored=True,
    )
    _write_pilot_manifest(
        pilot_root=tmp_path, args=_args(),
        conditions=[], bank_id="b", bank_hash="h",
        started_utc="x", completed_utc="y",
        per_cond_elapsed={}, per_cond_count={},
        client=fake_client,
    )
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert manifest["inference_params_unhonored"] is True


def test_manifest_omits_inference_fields_when_client_lacks_them(tmp_path):
    """OpenAI / Anthropic / VLLM clients don't carry these attributes;
    the manifest must not write the fields when absent."""
    from src.cli import _write_pilot_manifest

    fake_client = SimpleNamespace(model_name="gpt-5.4-nano")  # no auth_source, etc.
    _write_pilot_manifest(
        pilot_root=tmp_path, args=_args(provider="openai"),
        conditions=[], bank_id="b", bank_hash="h",
        started_utc="x", completed_utc="y",
        per_cond_elapsed={}, per_cond_count={},
        client=fake_client,
    )
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert "inference_auth_source" not in manifest
    assert "inference_total_cost_usd" not in manifest
    assert "inference_params_unhonored" not in manifest


def test_manifest_omits_inference_fields_when_client_is_none(tmp_path):
    """cmd_pilot path passes client=None (per-condition clients constructed inline)."""
    from src.cli import _write_pilot_manifest

    _write_pilot_manifest(
        pilot_root=tmp_path, args=_args(),
        conditions=[], bank_id="b", bank_hash="h",
        started_utc="x", completed_utc="y",
        per_cond_elapsed={}, per_cond_count={},
        client=None,
    )
    manifest = yaml.safe_load((tmp_path / "manifest.yaml").read_text())
    assert "inference_auth_source" not in manifest
