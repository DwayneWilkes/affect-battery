"""Stimulus-injection contract for Exp 3a (paper §3.4.1).

Each (level, run) cell SHALL deliver INTENSITY_LEVELS[level-1].feedback_text
as the system message and one bank item as the user message. No multi-turn
conditioning prefix and no neutral buffer turns are sent. Per-cell binary
correctness is recorded on Exp3aBody.

Tests cover the spec's six requirements from
specs/changes/affect-battery-h3a-runner-stimulus-injection/specs/
arousal-performance-runner/spec.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

from src.conditioning.prompts import Condition, FEEDBACK_SETS, INTENSITY_LEVELS
from src.models import ModelClient
from src.runner import Exp3aBody, ExperimentConfig, ExperimentType
from src.runners import run_exp3a


class RecordingClient(ModelClient):
    """DryRunClient analog that captures every messages array passed to complete()."""

    def __init__(self, response: str = "42", model: str = "recording"):
        self._response = response
        self._model = model
        self.captured: list[list[dict]] = []

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(self, messages, temperature=0.7, max_tokens=512) -> str:
        self.captured.append([dict(m) for m in messages])
        return self._response

    async def complete_text(self, prompt, temperature=0.7, max_tokens=512, stop=None) -> str:
        self.captured.append([{"role": "user", "content": prompt}])
        return self._response


def _bank_yaml(tmp_path: Path, n: int) -> Path:
    """Write a synthetic gsm8k-shaped bank with n items."""
    items = [
        {"id": f"item_{i:03d}", "question": f"What is {i} + {i}?", "expected": str(2 * i)}
        for i in range(n)
    ]
    bank = tmp_path / "bank.yaml"
    bank.write_text(yaml.safe_dump({"items": items}))
    return bank


def _write_seed(tmp_path: Path) -> Path:
    """Write a valid intensity-pilot seed (single-rater path)."""
    from src.probes.intensity_pilot import emit_solo_seed

    seed_path = tmp_path / "intensity_pilot_seed.json"
    emit_solo_seed(
        rater_id="rater_test",
        ratings=[1, 2, 3, 4, 5, 6, 7],
        axis_id="primary_valence_axis",
        n_levels=7,
        pilot_date="2026-04-27",
        output_path=seed_path,
    )
    return seed_path


def _config_for_exp3a(bank_path: Path, num_runs: int = 1, seed: int = 42, condition: Condition = Condition.STRONG_POSITIVE) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="recording",
        condition=condition,
        experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
        num_runs=num_runs,
        seed=seed,
        temperature=0.7,
        transfer_bank=str(bank_path),
    )


async def _collect(runner_kwargs):
    """Drain the async generator into a list."""
    return [r async for r in run_exp3a(**runner_kwargs)]


@pytest.mark.asyncio
async def test_messages_contain_level_feedback_text_as_system_role(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=1)

    await _collect(dict(
        config=config, client=client,
        intensity_levels=[4],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
    ))

    assert len(client.captured) == 1
    msgs = client.captured[0]
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == INTENSITY_LEVELS[3].feedback_text
    assert msgs[1]["role"] == "user"


@pytest.mark.asyncio
async def test_no_conditioning_phase_called(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=1)

    with mock.patch("src.runner.run_conditioning_phase") as mock_phase:
        await _collect(dict(
            config=config, client=client,
            intensity_levels=[1, 2, 3, 4, 5, 6, 7],
            pilot_seed_path=seed_path,
            output_dir=tmp_path / "out",
        ))
        mock_phase.assert_not_called()

    for msgs in client.captured:
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


@pytest.mark.asyncio
async def test_per_level_samples_are_disjoint(tmp_path):
    bank = _bank_yaml(tmp_path, n=200)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=10)

    results = await _collect(dict(
        config=config, client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
    ))
    items_per_level: dict[int, list[str]] = {}
    for r in results:
        items_per_level.setdefault(r.body.intensity_level, []).append(
            r.body.expected_answer
        )
    for i in items_per_level:
        for j in items_per_level:
            if i >= j:
                continue
            assert set(items_per_level[i]).isdisjoint(set(items_per_level[j]))


@pytest.mark.asyncio
async def test_same_seed_same_assignment(tmp_path):
    bank = _bank_yaml(tmp_path, n=100)
    seed_path = _write_seed(tmp_path)
    config_a = _config_for_exp3a(bank, num_runs=5, seed=42)
    config_b = _config_for_exp3a(bank, num_runs=5, seed=42)

    a = await _collect(dict(
        config=config_a, client=RecordingClient(),
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "a",
    ))
    b = await _collect(dict(
        config=config_b, client=RecordingClient(),
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "b",
    ))
    a_assignments = [(r.body.intensity_level, r.run_number, r.body.expected_answer) for r in a]
    b_assignments = [(r.body.intensity_level, r.run_number, r.body.expected_answer) for r in b]
    assert a_assignments == b_assignments


@pytest.mark.asyncio
async def test_insufficient_bank_raises(tmp_path):
    bank = _bank_yaml(tmp_path, n=5)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=10)

    with pytest.raises(ValueError) as exc:
        await _collect(dict(
            config=config, client=client,
            intensity_levels=[1, 2, 3, 4, 5, 6, 7],
            pilot_seed_path=seed_path,
            output_dir=tmp_path / "out",
        ))
    msg = str(exc.value)
    assert "5" in msg
    assert "70" in msg


@pytest.mark.asyncio
async def test_body_records_per_cell_correctness(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="0")
    config = _config_for_exp3a(bank, num_runs=1)

    results = await _collect(dict(
        config=config, client=client,
        intensity_levels=[1],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
    ))
    r = results[0]
    assert r.body.model_response == "0"
    assert r.body.expected_answer == r.body.expected_answer
    if r.body.expected_answer == "0":
        assert r.body.binary_correct == 1
    else:
        assert r.body.binary_correct == 0


@pytest.mark.asyncio
async def test_unparseable_response_scores_zero(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="I cannot answer that")
    config = _config_for_exp3a(bank, num_runs=1)

    results = await _collect(dict(
        config=config, client=client,
        intensity_levels=[4],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
    ))
    assert results[0].body.binary_correct == 0


@pytest.mark.asyncio
async def test_tolerance_boundary(tmp_path):
    """Every item in the bank has expected=42, so every cell is a boundary test."""
    bank = tmp_path / "bank.yaml"
    bank.write_text(yaml.safe_dump({"items": [
        {"id": f"i_{i:03d}", "question": f"q{i}", "expected": "42"}
        for i in range(70)
    ]}))
    seed_path = _write_seed(tmp_path)
    config = _config_for_exp3a(bank, num_runs=1)

    near = RecordingClient(response="42.005")
    far = RecordingClient(response="42.02")

    near_results = await _collect(dict(
        config=config, client=near,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "near",
    ))
    far_results = await _collect(dict(
        config=config, client=far,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "far",
    ))
    assert all(r.body.binary_correct == 1 for r in near_results)
    assert all(r.body.binary_correct == 0 for r in far_results)


@pytest.mark.asyncio
async def test_condition_does_not_affect_messages(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)

    client_a = RecordingClient(response="42")
    client_b = RecordingClient(response="42")
    cfg_a = _config_for_exp3a(bank, num_runs=2, condition=Condition.STRONG_POSITIVE)
    cfg_b = _config_for_exp3a(bank, num_runs=2, condition=Condition.NEUTRAL)

    await _collect(dict(
        config=cfg_a, client=client_a,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "a",
    ))
    await _collect(dict(
        config=cfg_b, client=client_b,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "b",
    ))
    assert client_a.captured == client_b.captured


@pytest.mark.asyncio
async def test_pilot_seed_gate_runs_first(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    payload = json.loads(seed_path.read_text())
    payload["axis_id"] = "tampered"
    seed_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=1)

    with pytest.raises(ValueError, match="SHA"):
        await _collect(dict(
            config=config, client=client,
            intensity_levels=[1, 2, 3, 4, 5, 6, 7],
            pilot_seed_path=seed_path,
            output_dir=tmp_path / "out",
        ))
    assert client.captured == []


@pytest.mark.asyncio
async def test_cell_count_contract_small(tmp_path):
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=3)

    results = await _collect(dict(
        config=config, client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
    ))
    assert len(results) == 21
    counts: dict[int, int] = {}
    for r in results:
        counts[r.body.intensity_level] = counts.get(r.body.intensity_level, 0) + 1
    for level in range(1, 8):
        assert counts[level] == 3


@pytest.mark.asyncio
async def test_results_persisted_under_level_dir(tmp_path):
    """Each yielded RunResult is written as JSON under output_dir/level_<N>/."""
    bank = _bank_yaml(tmp_path, n=50)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=2)

    out = tmp_path / "out"
    await _collect(dict(
        config=config, client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=out,
    ))

    written = sorted(out.rglob("*.json"))
    assert len(written) == 14, f"expected 14 result files, found {len(written)}"
    for level in range(1, 8):
        level_dir = out / f"level_{level}"
        json_files = sorted(level_dir.rglob("*.json"))
        assert len(json_files) == 2, f"level {level} has {len(json_files)} files, expected 2"
        first = json.loads(json_files[0].read_text())
        assert first["body"]["intensity_level"] == level
        assert first["body"]["model_response"] == "42"


@pytest.mark.asyncio
async def test_cell_count_at_pre_reg_n(tmp_path):
    bank = _bank_yaml(tmp_path, n=854)
    seed_path = _write_seed(tmp_path)
    client = RecordingClient(response="42")
    config = _config_for_exp3a(bank, num_runs=122)

    results = await _collect(dict(
        config=config, client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
    ))
    assert len(results) == 854
    counts: dict[int, int] = {}
    for r in results:
        counts[r.body.intensity_level] = counts.get(r.body.intensity_level, 0) + 1
    for level in range(1, 8):
        assert counts[level] == 122
