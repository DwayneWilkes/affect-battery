"""Tier-1 fix for the exp1a Haiku ceiling effect:
- Source transfer questions from a real benchmark bank (TriviaQA hard)
  rather than 6 hardcoded items.
- Score using answer aliases so 'U.S.' matches 'United States'.

Spec: affect-battery-proposal-realignment :: scoring-pipeline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.conditioning.prompts import Condition
from src.conditioning.tasks import TransferQuestion, get_transfer_tasks
from src.models import ModelClient
from src.runner import ExperimentConfig, ExperimentType, run_single
from src.scoring.accuracy import score_factual_qa


# --- 1. Alias-aware scorer --------------------------------------------------

class TestScoreFactualQAWithAliases:
    def test_aliases_default_to_none_backward_compat(self):
        # Existing 2-arg signature still works.
        assert score_factual_qa("The capital is Canberra.", "Canberra") == 1.0
        assert score_factual_qa("Sydney.", "Canberra") == 0.0

    def test_alias_match_scores_one(self):
        """'U.S.' in response, expected='United States', aliases=['U.S.', 'USA']
        → must score 1.0. Without aliases this would score 0.0 because
        'united states' is not a substring of 'the answer is u.s.'."""
        score = score_factual_qa(
            response="The answer is U.S.",
            expected_answer="United States",
            aliases=["U.S.", "USA", "America"],
        )
        assert score == 1.0

    def test_canonical_match_still_works_when_aliases_present(self):
        score = score_factual_qa(
            response="It's the United States of America.",
            expected_answer="United States",
            aliases=["U.S.", "USA"],
        )
        assert score == 1.0

    def test_no_alias_match_scores_zero(self):
        score = score_factual_qa(
            response="I think it might be Canada.",
            expected_answer="United States",
            aliases=["U.S.", "USA", "America"],
        )
        assert score == 0.0

    def test_empty_aliases_list_equivalent_to_no_aliases(self):
        assert score_factual_qa("Canberra is the capital.", "Canberra", aliases=[]) == 1.0
        assert score_factual_qa("Sydney.", "Canberra", aliases=[]) == 0.0

    def test_case_insensitive_alias_match(self):
        score = score_factual_qa(
            response="the response mentions usa specifically",
            expected_answer="United States",
            aliases=["U.S.", "USA"],
        )
        assert score == 1.0


# --- 2. Bank loader ---------------------------------------------------------

class TestGetTransferTasksFromBank:
    def test_default_pool_still_works_without_bank_path(self):
        """Backward-compat: existing callers that don't pass bank_path keep
        getting the hardcoded factual_qa pool."""
        tasks = get_transfer_tasks("factual_qa", 3, seed=42)
        assert len(tasks) == 3
        assert all(isinstance(t, TransferQuestion) for t in tasks)

    def test_bank_path_loads_yaml_items(self, tmp_path):
        """Passing a bank_path loads items from a YAML bank file. The bank's
        items become the sampling pool."""
        import yaml

        bank_file = tmp_path / "test_bank.yaml"
        bank_file.write_text(yaml.safe_dump({
            "bank_id": "test_bank_v1",
            "items": [
                {
                    "id": "t1",
                    "question": "Capital of Australia?",
                    "expected": "Canberra",
                    "answer_aliases": ["Canberra, ACT"],
                    "difficulty": "hard",
                },
                {
                    "id": "t2",
                    "question": "Symbol for gold?",
                    "expected": "Au",
                    "answer_aliases": ["AU", "gold (Au)"],
                    "difficulty": "hard",
                },
            ],
        }))

        tasks = get_transfer_tasks("factual_qa", 2, seed=0, bank_path=str(bank_file))
        assert len(tasks) == 2
        # Aliases flowed through into TransferQuestion.
        questions_by_expected = {t.expected_answer: t for t in tasks}
        assert "Canberra" in questions_by_expected
        assert questions_by_expected["Canberra"].expected_aliases == ["Canberra, ACT"]
        assert questions_by_expected["Au"].expected_aliases == ["AU", "gold (Au)"]


# --- 3. TransferQuestion dataclass ------------------------------------------

class TestTransferQuestionAliases:
    def test_expected_aliases_defaults_empty(self):
        q = TransferQuestion(
            question="Q?", expected_answer="A", task_type="factual_qa",
        )
        assert q.expected_aliases == []

    def test_expected_aliases_settable(self):
        q = TransferQuestion(
            question="Q?", expected_answer="A", task_type="factual_qa",
            expected_aliases=["a", "A."],
        )
        assert q.expected_aliases == ["a", "A."]


# --- 4. Runner persists aliases + uses them when scoring --------------------

class _ScriptedChatClient(ModelClient):
    def __init__(self, responses):
        self._responses = responses
        self._i = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

    async def complete_text(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        raise NotImplementedError


class TestRunnerAliasScoring:
    def test_runner_uses_aliases_from_bank(self, tmp_path):
        """When transfer_bank is set on the config, the runner loads from
        the bank, scores with aliases, and persists alias lists on the
        result so the analyzer can rescore later if needed."""
        import yaml

        bank_file = tmp_path / "transfer_bank.yaml"
        bank_file.write_text(yaml.safe_dump({
            "bank_id": "transfer_test_v1",
            "items": [
                {
                    "id": "t1",
                    "question": "What does USA stand for?",
                    "expected": "United States of America",
                    "answer_aliases": ["U.S.A.", "USA", "United States"],
                    "difficulty": "hard",
                },
            ],
        }))

        cfg = ExperimentConfig(
            model_name="m",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=1,
            num_conditioning_turns=1,
            num_transfer_questions=1,
            seed=0,
            transfer_bank=str(bank_file),
        )
        # Conditioning answer + transfer answer that matches an ALIAS only.
        client = _ScriptedChatClient(responses=["10", "USA"])
        result = asyncio.run(run_single(cfg, client, 0))

        # Alias match → 1.0 (would have been 0.0 without alias-aware scoring,
        # because 'united states of america' isn't a substring of 'usa').
        assert result.transfer_correct == [1.0]
        # Aliases persisted alongside expected so re-scoring is possible.
        assert result.transfer_expected_aliases == [
            ["U.S.A.", "USA", "United States"]
        ]


# --- 5. Smoke test: real bank file exists and has hard subset ---------------

class TestExp1aHardBankFile:
    def test_hard_bank_file_exists_and_has_hard_items(self):
        """The Tier-1 fix ships a new bank YAML at
        configs/banks/exp1a_factual_qa_hard_v1.yaml containing only the
        TriviaQA HARD subset (the existing exp3c bank's hard tier)."""
        import yaml

        path = Path("configs/banks/exp1a_factual_qa_hard_v1.yaml")
        assert path.exists(), (
            f"Tier-1 hard transfer bank not found at {path}. "
            "Subset exp3c_triviaqa.yaml's hard items into this file."
        )
        bank = yaml.safe_load(path.read_text())
        items = bank["items"]
        assert len(items) >= 15, (
            f"Hard bank should have >=15 items, found {len(items)}"
        )
        assert all(it["difficulty"] == "hard" for it in items), (
            "All items in the hard bank must be tagged difficulty=hard"
        )
        # Items must carry alias lists for alias-aware scoring.
        assert all(
            isinstance(it.get("answer_aliases"), list) and it["answer_aliases"]
            for it in items
        ), "Every hard-bank item must have a non-empty answer_aliases list"
