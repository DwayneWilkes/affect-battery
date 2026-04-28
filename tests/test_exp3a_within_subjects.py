"""run_exp3a within-subjects mode (amendment 002)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.conditioning.prompts import Condition
from src.runner import ExperimentConfig, ExperimentType
from src.runners import run_exp3a


class _RecordingClient:
    def __init__(self, response="42"):
        self._response = response
        self.captured = []

    @property
    def model_name(self):
        return "test"

    async def complete(self, messages, temperature=1.0, max_tokens=512):
        self.captured.append([dict(m) for m in messages])
        return self._response

    async def complete_text(self, *_a, **_kw):
        raise NotImplementedError


def _bank(tmp_path: Path, n: int) -> Path:
    bank = tmp_path / "bank.yaml"
    bank.write_text(yaml.safe_dump({"items": [
        {"id": f"item_{i:03d}", "question": f"q{i}", "expected": str(i)}
        for i in range(n)
    ]}))
    return bank


def _seed(tmp_path: Path) -> Path:
    from src.probes.intensity_pilot import emit_solo_seed
    seed_path = tmp_path / "seed.json"
    emit_solo_seed(
        rater_id="rater_test", ratings=[1, 2, 3, 4, 5, 6, 7],
        axis_id="primary_valence_axis", n_levels=7,
        pilot_date="2026-04-27", output_path=seed_path,
    )
    return seed_path


def _config(bank, num_runs=10):
    return ExperimentConfig(
        model_name="test", condition=Condition.NEUTRAL,
        experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
        num_runs=num_runs, seed=42, temperature=0.7,
        transfer_bank=str(bank),
    )


async def _drain(gen):
    return [r async for r in gen]


@pytest.mark.asyncio
async def test_within_subjects_mode_uses_shared_items(tmp_path):
    bank = _bank(tmp_path, 50)
    seed_path = _seed(tmp_path)
    client = _RecordingClient()
    results = await _drain(run_exp3a(
        config=_config(bank, num_runs=10), client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
        sampling_mode="within_subjects",
    ))

    assert len(results) == 70  # 7 levels × 10 runs
    by_level: dict[int, list[str]] = {}
    for r in results:
        by_level.setdefault(r.body.intensity_level, []).append(r.body.item_id)
    expected_items = by_level[1]
    for level, items in by_level.items():
        assert items == expected_items, (
            f"level {level} items differ from level 1; within-subjects "
            f"mode requires identical item lists across levels"
        )


@pytest.mark.asyncio
async def test_default_mode_is_cross_level_disjoint(tmp_path):
    bank = _bank(tmp_path, 100)
    seed_path = _seed(tmp_path)
    client = _RecordingClient()
    results = await _drain(run_exp3a(
        config=_config(bank, num_runs=10), client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
        # sampling_mode omitted; default applies
    ))
    by_level: dict[int, set[str]] = {}
    for r in results:
        by_level.setdefault(r.body.intensity_level, set()).add(r.body.item_id)
    for i in by_level:
        for j in by_level:
            if i >= j:
                continue
            assert by_level[i].isdisjoint(by_level[j])


@pytest.mark.asyncio
async def test_within_subjects_records_mode_on_body(tmp_path):
    bank = _bank(tmp_path, 50)
    seed_path = _seed(tmp_path)
    client = _RecordingClient()
    results = await _drain(run_exp3a(
        config=_config(bank, num_runs=5), client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
        sampling_mode="within_subjects",
    ))
    assert all(r.body.sampling_mode == "within_subjects" for r in results)
    assert all(r.body.item_id != "" for r in results)


@pytest.mark.asyncio
async def test_unknown_sampling_mode_raises(tmp_path):
    bank = _bank(tmp_path, 50)
    seed_path = _seed(tmp_path)
    client = _RecordingClient()
    with pytest.raises(ValueError, match="sampling_mode"):
        await _drain(run_exp3a(
            config=_config(bank, num_runs=5), client=client,
            intensity_levels=[1, 2, 3, 4, 5, 6, 7],
            pilot_seed_path=seed_path,
            output_dir=tmp_path / "out",
            sampling_mode="bogus",
        ))


@pytest.mark.asyncio
async def test_within_subjects_works_with_small_bank(tmp_path):
    """Within-subjects mode requires only n_per_level items, not n_per_level × n_levels."""
    bank = _bank(tmp_path, 10)  # exactly n_per_level
    seed_path = _seed(tmp_path)
    client = _RecordingClient()
    results = await _drain(run_exp3a(
        config=_config(bank, num_runs=10), client=client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=seed_path,
        output_dir=tmp_path / "out",
        sampling_mode="within_subjects",
    ))
    assert len(results) == 70
