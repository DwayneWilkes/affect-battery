"""Transfer scoring contract.

`run_single` populates `transfer_correct` at write time by scoring each
transfer response against its expected answer. Result files that lack
`transfer_correct` (legacy schema) are rescued at load time: the
analyzer's `_load_corpus` backfills the field on the fly via the same
`score_factual_qa` scorer so downstream `run_accuracy` sees a populated
list rather than collapsing to 0.0.

Spec: affect-battery-proposal-realignment :: scoring-pipeline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.analysis.pipeline import _load_corpus
from src.conditioning.prompts import Condition
from src.models import ModelClient
from src.runner import ExperimentConfig, ExperimentType, run_single, save_result


class _ScriptedChatClient(ModelClient):
    """Returns scripted answers in order. Records messages it was called with."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0
        self.calls: list[list[dict]] = []

    @property
    def model_name(self) -> str:
        return "scripted-chat"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        self.calls.append(list(messages))
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

    async def complete_text(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        raise NotImplementedError("chat path; complete_text not used here")


def _cfg(num_transfer: int = 2) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="claude-haiku-4-5",
        condition=Condition.STRONG_NEGATIVE,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=1,
        num_conditioning_turns=2,
        num_transfer_questions=num_transfer,
        seed=42,
    )


class TestRunnerWriteTimeScoring:
    """run_single MUST populate transfer_correct from transfer_responses
    and transfer_expected using score_factual_qa."""

    def test_run_single_grades_transfer_responses(self):
        cfg = _cfg(num_transfer=2)
        # Conditioning answers (numeric) come first; transfer answers
        # (factual_qa) follow. The transfer responses below intentionally
        # match the expected substrings the seed=42 transfer task batch
        # produces, but the test asserts a property — that
        # transfer_correct has the right length and per-item type — that
        # holds regardless of which expected strings show up.
        client = _ScriptedChatClient(
            responses=[
                "115", "86",  # 2 conditioning turns
                "The first person to circumnavigate the globe was Ferdinand Magellan.",
                "Canberra is the capital of Australia.",
            ],
        )
        result = asyncio.run(run_single(cfg, client, 0))

        # Length-3 contract: questions == responses == expected == correct
        assert len(result.transfer_correct) == cfg.num_transfer_questions
        assert len(result.transfer_correct) == len(result.transfer_responses)
        # Each correctness entry is 0.0 or 1.0 (float, not None / not bool)
        for c in result.transfer_correct:
            assert isinstance(c, float)
            assert c in (0.0, 1.0)

    def test_obviously_wrong_response_scores_zero(self):
        """Response with no overlap with expected scores 0.0."""
        cfg = _cfg(num_transfer=1)
        client = _ScriptedChatClient(
            responses=["115", "86", "I have no idea what you're asking."],
        )
        result = asyncio.run(run_single(cfg, client, 0))
        # Whatever the seed=42 expected answer is, "I have no idea..."
        # cannot contain it as a substring, so this MUST be 0.0.
        assert result.transfer_correct == [0.0]

    def test_save_roundtrip_preserves_transfer_correct(self, tmp_path):
        cfg = _cfg(num_transfer=1)
        client = _ScriptedChatClient(responses=["115", "86", "Some answer."])
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        data = json.loads(path.read_text())
        assert "transfer_correct" in data
        assert len(data["transfer_correct"]) == 1


class TestAnalyzerLoadTimeScoringFallback:
    """_load_corpus MUST fill transfer_correct on the fly when a legacy
    result file lacks it (or has it as None / empty list). This rescues
    pilot data already saved without re-running the API."""

    def _write_legacy_result(self, path: Path, response: str, expected: str) -> None:
        """Write a result JSON in the legacy shape: transfer_correct missing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "model_name": "claude-haiku-4-5",
                "condition": "strong_negative",
                "experiment_type": "exp1a",
            },
            "run_number": 0,
            "experiment_type": "exp1a",
            "model": "claude-haiku-4-5",
            "condition": "strong_negative",
            "conditioning_responses": [],
            "conditioning_correct": [],
            "transfer_responses": [response],
            "transfer_questions": ["dummy?"],
            "transfer_expected": [expected],
            # transfer_correct intentionally MISSING (legacy shape).
            "start_time": 0.0,
            "end_time": 0.0,
            "checksum": "",
        }
        path.write_text(json.dumps(payload))

    def test_load_corpus_rescues_legacy_correct_response(self, tmp_path):
        results_dir = tmp_path / "results"
        out = results_dir / "exp1a" / "run_0001.json"
        self._write_legacy_result(
            out,
            response="Ferdinand Magellan circumnavigated the globe.",
            expected="Ferdinand Magellan",
        )
        corpus = _load_corpus(results_dir, "exp1a")
        assert len(corpus) == 1
        # Substring match → 1.0
        assert corpus[0]["transfer_correct"] == [1.0]

    def test_load_corpus_rescues_legacy_wrong_response(self, tmp_path):
        results_dir = tmp_path / "results"
        out = results_dir / "exp1a" / "run_0002.json"
        self._write_legacy_result(
            out,
            response="I don't know.",
            expected="Ferdinand Magellan",
        )
        corpus = _load_corpus(results_dir, "exp1a")
        assert corpus[0]["transfer_correct"] == [0.0]

    def test_load_corpus_preserves_existing_transfer_correct(self, tmp_path):
        """When a result file already has transfer_correct populated (new
        runner output), the loader MUST NOT overwrite it."""
        results_dir = tmp_path / "results"
        out = results_dir / "exp1a" / "run_0003.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {"model_name": "m", "condition": "neutral",
                       "experiment_type": "exp1a"},
            "run_number": 0,
            "experiment_type": "exp1a",
            "model": "m",
            "condition": "neutral",
            "conditioning_responses": [],
            "conditioning_correct": [],
            # Pre-graded: response says "wrong" but transfer_correct
            # claims [1.0]. Loader must trust the stored value.
            "transfer_responses": ["totally off-base"],
            "transfer_questions": ["q?"],
            "transfer_expected": ["right answer"],
            "transfer_correct": [1.0],
            "start_time": 0.0,
            "end_time": 0.0,
            "checksum": "",
        }
        out.write_text(json.dumps(payload))
        corpus = _load_corpus(results_dir, "exp1a")
        assert corpus[0]["transfer_correct"] == [1.0]
