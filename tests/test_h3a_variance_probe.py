"""Tests for the H3a variance probe."""
from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

import pytest

# Add scripts/probes to path so we can import the probe module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

import h3a_variance_probe as probe  # noqa: E402

from src.models import ModelClient  # noqa: E402


class _ScriptedClient(ModelClient):
    """Returns responses from a pre-supplied list, in order."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0

    @property
    def model_name(self) -> str:
        return "scripted"

    async def complete(self, messages, temperature=0.7, max_tokens=512):
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

    async def complete_text(self, *args, **kwargs):
        raise NotImplementedError


class TestScoring:
    def test_correct_numeric_answer(self):
        assert probe.score_response("The answer is 42.", "42") == 1

    def test_correct_with_chain_of_reasoning(self):
        text = "First I add 7 + 8 = 15. So the answer is 15."
        assert probe.score_response(text, "15") == 1

    def test_incorrect_answer(self):
        assert probe.score_response("The answer is 42.", "100") == 0

    def test_unparseable_response(self):
        assert probe.score_response("I don't know.", "42") == 0

    def test_float_expected_with_int_response(self):
        """Expected '2796088.0' should match response containing 2796088."""
        assert probe.score_response("The answer is 2796088.", "2796088.0") == 1


class TestSampling:
    def test_disjoint_per_level_samples(self):
        items = [{"id": f"item_{i}", "question": "q", "expected": "0"} for i in range(100)]
        per_level = probe.sample_items(items, n_per_level=5, n_levels=7, seed=42)
        assert len(per_level) == 7
        # Each level gets exactly n_per_level items
        assert all(len(level) == 5 for level in per_level)
        # No item appears in two levels (disjoint)
        all_ids = [item["id"] for level in per_level for item in level]
        assert len(set(all_ids)) == len(all_ids)

    def test_too_few_items_raises(self):
        items = [{"id": "x", "question": "q", "expected": "0"}] * 5
        with pytest.raises(ValueError, match="need at least"):
            probe.sample_items(items, n_per_level=5, n_levels=7, seed=42)

    def test_seed_reproducibility(self):
        items = [{"id": f"item_{i}", "question": "q", "expected": str(i)} for i in range(100)]
        a = probe.sample_items(items, n_per_level=5, n_levels=7, seed=42)
        b = probe.sample_items(items, n_per_level=5, n_levels=7, seed=42)
        assert a == b


class TestProbeOneLevel:
    def test_perfect_accuracy(self):
        """Client always returns the right answer -> all scores are 1."""
        items = [{"id": "a", "question": "q1", "expected": "10"},
                 {"id": "b", "question": "q2", "expected": "20"}]
        client = _ScriptedClient(["The answer is 10.", "The answer is 20."])
        scores = asyncio.run(probe.probe_one_level(
            client=client, intensity_text="some stimulus",
            items=items, temperature=0.7, max_tokens=512,
        ))
        assert scores == [1, 1]

    def test_partial_accuracy(self):
        """Client returns half right -> mean = 0.5, std nonzero."""
        items = [
            {"id": "a", "question": "q", "expected": "10"},
            {"id": "b", "question": "q", "expected": "20"},
            {"id": "c", "question": "q", "expected": "30"},
            {"id": "d", "question": "q", "expected": "40"},
        ]
        client = _ScriptedClient([
            "The answer is 10.", "The answer is 999.",  # 1, 0
            "The answer is 30.", "The answer is 999.",  # 1, 0
        ])
        scores = asyncio.run(probe.probe_one_level(
            client=client, intensity_text="x",
            items=items, temperature=0.7, max_tokens=512,
        ))
        assert scores == [1, 0, 1, 0]
        assert statistics.mean(scores) == 0.5
        # Sample std for [1, 0, 1, 0] = sqrt(((0.5)^2 + (0.5)^2 + (0.5)^2 + (0.5)^2) / 3)
        #                              = sqrt(1/3) ≈ 0.577
        assert statistics.stdev(scores) == pytest.approx(0.577, abs=0.01)


class TestBankLoading:
    def test_loads_well_formed_bank(self, tmp_path):
        import yaml
        bank_path = tmp_path / "test_bank.yaml"
        bank_path.write_text(yaml.safe_dump({
            "bank_id": "test_v1",
            "items": [
                {"id": "a1", "question": "What is 2+2?", "expected": "4",
                 "difficulty": "easy"},
                {"id": "a2", "question": "What is 3+5?", "expected": "8",
                 "difficulty": "easy"},
            ],
        }))
        items = probe.load_bank_items(bank_path)
        assert len(items) == 2
        assert items[0]["id"] == "a1"
        assert items[0]["expected"] == "4"
        assert items[0]["difficulty"] == "easy"

    def test_empty_bank_raises(self, tmp_path):
        import yaml
        bank_path = tmp_path / "empty.yaml"
        bank_path.write_text(yaml.safe_dump({"bank_id": "empty", "items": []}))
        with pytest.raises(ValueError, match="no `items`"):
            probe.load_bank_items(bank_path)
