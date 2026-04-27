"""Tests for the discriminated-union RunResult schema ().

Per design.md D6: RunResult carries base fields (config, checksum,
experiment_type, model, condition, run_number, start_time, end_time)
and a discriminated body keyed on experiment_type.

- Exp1aBody / Exp1bBody wrap conditioning/transfer responses.
- Exp2Body has per-turn accuracies + N-value.
- Exp3aBody has intensity-level data.
- Exp3bBody has open-ended generations + embedding variance.
- Exp3cBody has factual QA responses + hedging/confidence/refusal.

Round-trips through JSON preserve body type via experiment_type tag.
"""

import json
from dataclasses import asdict

import pytest

from src.runner import (
    RunResult,
    Exp1aBody,
    Exp1bBody,
    Exp2Body,
    Exp3aBody,
    Exp3bBody,
    Exp3cBody,
)


def _base_kwargs(experiment_type: str) -> dict:
    return {
        "config": {"model": "m", "condition": "NEUTRAL", "seed": 0},
        "run_number": 1,
        "experiment_type": experiment_type,
        "model": "m",
        "condition": "NEUTRAL",
        "start_time": 1.0,
        "end_time": 2.0,
    }


def test_result_has_experiment_type_field():
    r = RunResult(
        **_base_kwargs("exp1a"),
        body=Exp1aBody(
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=[],
            transfer_questions=[],
            transfer_expected=[],
        ),
    )
    assert r.experiment_type == "exp1a"


def test_exp1a_body_wraps_conditioning_and_transfer():
    r = RunResult(
        **_base_kwargs("exp1a"),
        body=Exp1aBody(
            conditioning_responses=["a"],
            conditioning_correct=[True],
            transfer_responses=["b"],
            transfer_questions=["q"],
            transfer_expected=["b"],
        ),
    )
    assert r.body.conditioning_responses == ["a"]
    assert r.body.transfer_responses == ["b"]


def test_exp1b_body_has_session_seeds():
    r = RunResult(
        **_base_kwargs("exp1b"),
        body=Exp1bBody(
            conditioning_responses=[],
            conditioning_correct=[],
            transfer_responses=[],
            transfer_questions=[],
            transfer_expected=[],
            session_1_seed=1,
            session_2_seed=2,
        ),
    )
    assert r.body.session_1_seed == 1
    assert r.body.session_2_seed == 2


def test_exp2_body_has_per_turn_accuracy_and_n():
    r = RunResult(
        **_base_kwargs("exp2"),
        body=Exp2Body(
            n_value=5,
            turn_accuracies=[0.8, 0.7, 0.9, 0.85, 0.95],
        ),
    )
    assert r.body.n_value == 5
    assert len(r.body.turn_accuracies) == 5


def test_exp3a_body_has_intensity_level():
    r = RunResult(
        **_base_kwargs("exp3a"),
        body=Exp3aBody(
            intensity_level=4,
            transfer_responses=["x"],
            transfer_expected=["x"],
        ),
    )
    assert r.body.intensity_level == 4


def test_exp3b_body_has_generations():
    r = RunResult(
        **_base_kwargs("exp3b"),
        body=Exp3bBody(
            prompt_id="p1",
            generations=["g1", "g2", "g3"],
            per_generation_seeds=[10, 20, 30],
        ),
    )
    assert len(r.body.generations) == 3
    assert r.body.per_generation_seeds == [10, 20, 30]


def test_exp3c_body_has_difficulty_and_confidence():
    r = RunResult(
        **_base_kwargs("exp3c"),
        body=Exp3cBody(
            difficulty="hard",
            question="q1",
            response="r1",
            expected="r1",
            stated_confidence=7,
            refused=False,
        ),
    )
    assert r.body.difficulty == "hard"
    assert r.body.stated_confidence == 7


def test_round_trip_via_asdict_preserves_experiment_type():
    r = RunResult(
        **_base_kwargs("exp2"),
        body=Exp2Body(n_value=3, turn_accuracies=[0.5, 0.6, 0.7]),
    )
    d = asdict(r)
    assert d["experiment_type"] == "exp2"
    assert d["body"]["n_value"] == 3


def test_json_round_trip_preserves_body_fields():
    r = RunResult(
        **_base_kwargs("exp1a"),
        body=Exp1aBody(
            conditioning_responses=["r1"],
            conditioning_correct=[True],
            transfer_responses=["t1"],
            transfer_questions=["q1"],
            transfer_expected=["t1"],
        ),
    )
    encoded = json.dumps(asdict(r))
    decoded = json.loads(encoded)
    assert decoded["experiment_type"] == "exp1a"
    assert decoded["body"]["conditioning_responses"] == ["r1"]


def test_invalid_experiment_type_rejected():
    with pytest.raises((ValueError, TypeError)):
        RunResult(
            **{**_base_kwargs("exp99"), "experiment_type": "exp99"},
            body=Exp1aBody(
                conditioning_responses=[],
                conditioning_correct=[],
                transfer_responses=[],
                transfer_questions=[],
                transfer_expected=[],
            ),
        )
