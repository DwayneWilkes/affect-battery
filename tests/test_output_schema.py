"""Tests for result JSON schema, save_result validation, and load_results.

Spec (scoring-pipeline, Requirement: Result JSON schema):
- Every experiment run produces a JSON file conforming to configs/result_schema.json.
- save_result rejects results missing required fields.
- load_results recomputes and verifies checksum; warns on tampered files.
"""

import json
from pathlib import Path

import pytest

from src.runner import (
    ExperimentType,
    RunResult,
    load_result,
    load_results,
    save_result,
)


SCHEMA_PATH = Path(__file__).parent.parent / "configs" / "result_schema.json"


class TestSchemaFile:
    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), f"Missing schema file: {SCHEMA_PATH}"

    def test_schema_is_valid_json(self):
        data = json.loads(SCHEMA_PATH.read_text())
        assert "properties" in data or "required" in data

    def test_schema_declares_required_fields(self):
        schema = json.loads(SCHEMA_PATH.read_text())
        required_fields = {
            "config", "run_number", "conditioning_responses", "conditioning_correct",
            "transfer_responses", "transfer_questions", "transfer_expected",
            "start_time", "end_time", "checksum",
        }
        declared = set(schema.get("required", []))
        missing = required_fields - declared
        assert not missing, f"Schema missing required fields: {missing}"


def _full_result(**overrides) -> RunResult:
    defaults = dict(
        config={"model_name": "test", "condition": "neutral", "experiment_type": "transfer_within"},
        run_number=0,
        conditioning_responses=["42"],
        conditioning_correct=[True],
        transfer_responses=["Paris"],
        transfer_questions=["Q"],
        transfer_expected=["Paris"],
        start_time=1000.0,
        end_time=1001.0,
    )
    defaults.update(overrides)
    r = RunResult(**defaults)
    r.compute_checksum()
    return r


class TestSaveValidation:
    def test_save_rejects_empty_config(self, tmp_path):
        r = _full_result(config={})
        with pytest.raises(ValueError, match="config"):
            save_result(r, tmp_path)

    def test_save_rejects_mismatched_lengths(self, tmp_path):
        """transfer_questions, transfer_responses, transfer_expected should
        have equal length per run."""
        r = _full_result(
            transfer_questions=["Q1", "Q2"],
            transfer_responses=["Paris"],
            transfer_expected=["Paris", "Berlin"],
        )
        with pytest.raises(ValueError, match="length"):
            save_result(r, tmp_path)

    def test_save_accepts_valid_result(self, tmp_path):
        r = _full_result()
        path = save_result(r, tmp_path)
        assert path.exists()


class TestLoadResult:
    def test_load_returns_dict(self, tmp_path):
        r = _full_result()
        path = save_result(r, tmp_path)
        loaded = load_result(path)
        assert loaded["run_number"] == 0
        assert loaded["checksum"] == r.checksum

    def test_load_warns_on_tamper(self, tmp_path, caplog):
        import logging
        r = _full_result()
        path = save_result(r, tmp_path)

        # Tamper by hand-editing the transfer_responses.
        data = json.loads(path.read_text())
        data["transfer_responses"] = ["London"]  # was ["Paris"]
        path.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING):
            loaded = load_result(path, verify=True)
        assert any(
            "checksum" in record.message.lower() or "tamper" in record.message.lower()
            for record in caplog.records
        ), f"Expected checksum/tamper warning. Records: {[r.message for r in caplog.records]}"


class TestLoadResults:
    def test_load_directory(self, tmp_path):
        for i in range(3):
            r = _full_result(run_number=i)
            save_result(r, tmp_path)
        loaded = load_results(tmp_path)
        assert len(loaded) == 3
        assert {d["run_number"] for d in loaded} == {0, 1, 2}

    def test_load_empty_directory(self, tmp_path):
        assert load_results(tmp_path) == []


class TestProtocolFeedbackSetUsage:
    """Task 8.5: protocol.py uses FEEDBACK_SETS (per-turn feedback).
    Verify that different turns yield different feedback text for
    conditions where FEEDBACK_SETS defines unique-per-turn texts."""

    def test_per_turn_feedback_differs(self):
        """STRONG_POSITIVE has 5 unique per-turn texts. Messages built for
        turn 0 and turn 2 must contain different feedback text."""
        from src.conditioning.prompts import Condition
        from src.conditioning.protocol import (
            ConditioningProtocol,
            build_conditioning_messages,
        )
        from src.conditioning.tasks import get_arithmetic_problems

        protocol = ConditioningProtocol(
            condition=Condition.STRONG_POSITIVE, num_conditioning_turns=5
        )
        problems = get_arithmetic_problems(5, seed=42)
        model_answers = [str(int(p.answer)) for p in problems]
        actual_correct = [True] * 5

        messages = build_conditioning_messages(
            protocol, problems, model_answers=model_answers, actual_correct=actual_correct
        )
        # Feedback turns are Human-role messages that appear after each
        # assistant response. Extract them.
        feedbacks = []
        for i, m in enumerate(messages):
            if i > 0 and messages[i - 1].role == "assistant" and m.role == "user":
                feedbacks.append(m.content)
        assert len(feedbacks) == 5
        # At least turn 0 and turn 2 differ -- per-turn variation in action.
        assert feedbacks[0] != feedbacks[2], (
            f"Turn 0 and turn 2 should differ under per-turn FEEDBACK_SETS. "
            f"Got: {feedbacks[0]!r} and {feedbacks[2]!r}"
        )
