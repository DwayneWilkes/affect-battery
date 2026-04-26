"""Tests for the canonical data banks committed to configs/banks/.

These tests load each YAML and validate its schema against the spec
contract. They DO NOT re-run the ingestion scripts (those require
HuggingFace network access). The fixture YAML files are checked into
the repo so the schema contract is testable without network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "banks"


class TestFolioBank:
    def test_folio_bank_loads_with_active_status(self):
        path = CONFIGS / "folio_active.yaml"
        if not path.exists():
            pytest.skip("folio_active.yaml not yet ingested")
        bank = yaml.safe_load(path.read_text())
        assert bank["bank_id"] == "folio_active_v1"
        assert bank["status"] == "active"
        assert bank["bank_type"] == "transfer"
        assert bank["alignment_review"]["verdict"] == "pass"
        # Citation present (cross-domain-transfer-tasks spec requirement)
        assert "citation" in bank["difficulty_profile"]
        # Items have the schema run_exp3c expects
        for item in bank["items"]:
            assert item["task_type"] == "logic-puzzle"
            assert item["difficulty_class"] in {"easy", "medium", "hard"}
            assert item["prompt"]
            assert item["expected_answer"]


class TestTriviaQABank:
    def test_triviaqa_bank_balanced_buckets(self):
        path = CONFIGS / "exp3c_triviaqa.yaml"
        if not path.exists():
            pytest.skip("exp3c_triviaqa.yaml not yet ingested")
        bank = yaml.safe_load(path.read_text())
        assert bank["status"] == "active"
        assert bank["alignment_review"]["verdict"] == "pass"
        assert "citation" in bank["difficulty_profile"]

        # Items in the runner-config schema (difficulty / question / expected)
        difficulties = [it["difficulty"] for it in bank["items"]]
        # Each bucket has at least 5 items so per-bucket aggregation is meaningful
        for bucket in ("easy", "medium", "hard"):
            assert difficulties.count(bucket) >= 5, (
                f"bucket {bucket!r} has only {difficulties.count(bucket)} items"
            )
        # Aliases preserved for accuracy scoring
        assert "answer_aliases" in bank["items"][0]


class TestExp3bPromptsBank:
    def test_prompts_have_three_families(self):
        path = CONFIGS / "exp3b_prompts.yaml"
        bank = yaml.safe_load(path.read_text())
        families = {p["family"] for p in bank["prompts"]}
        assert families == {
            "story_completion",
            "brainstorming",
            "creative_problem_solving",
        }
        # At least 3 prompts per family so cross-condition aggregation
        # has variety
        for fam in families:
            count = sum(1 for p in bank["prompts"] if p["family"] == fam)
            assert count >= 3, f"family {fam} has {count} prompts"

    def test_prompts_runner_config_schema(self):
        """Each prompt has the {id, text} keys required by
        `affect-battery run --experiment exp3b --runner-config <yaml>`."""
        path = CONFIGS / "exp3b_prompts.yaml"
        bank = yaml.safe_load(path.read_text())
        for prompt in bank["prompts"]:
            assert "id" in prompt
            assert "text" in prompt
            assert prompt["text"].strip()


class TestExp3aIntensityLevels:
    def test_two_axes_present(self):
        path = CONFIGS / "exp3a_intensity_levels.yaml"
        bank = yaml.safe_load(path.read_text())
        assert "primary_valence_axis" in bank
        assert "replication_arousal_axis" in bank

    def test_primary_axis_has_seven_levels(self):
        path = CONFIGS / "exp3a_intensity_levels.yaml"
        bank = yaml.safe_load(path.read_text())
        primary = bank["primary_valence_axis"]
        assert len(primary["levels"]) == 7
        levels = [lvl["level"] for lvl in primary["levels"]]
        assert levels == [1, 2, 3, 4, 5, 6, 7]
        # Pre-registered peak prediction recorded per
        # conditioning-protocol spec scenario
        assert primary["expected_peak_level"] == 4

    def test_replication_axis_uses_li_2024_ordering(self):
        """Per the conditioning-protocol spec scenario 'Replication axis
        uses authors' rank order verbatim': order MUST be EA22, EA17,
        EA20, EA18, EA21, EA16, EA19 (Li et al. 2024 Table 19)."""
        path = CONFIGS / "exp3a_intensity_levels.yaml"
        bank = yaml.safe_load(path.read_text())
        replication = bank["replication_arousal_axis"]
        assert len(replication["levels"]) == 7
        order = [lvl["stimulus_id"] for lvl in replication["levels"]]
        assert order == ["EA22", "EA17", "EA20", "EA18", "EA21", "EA16", "EA19"]

    def test_status_starts_candidate_until_pilot_passes(self):
        """The intensity-axis bank waits on the Krippendorff pilot before
        promotion. Status MUST be 'candidate' in the committed YAML."""
        path = CONFIGS / "exp3a_intensity_levels.yaml"
        bank = yaml.safe_load(path.read_text())
        assert bank["status"] == "candidate"


# ---------------------------------------------------------------------------
# Wiring: budget / rate-limit / cancel kwargs reach exp3a/3b/3c
# ---------------------------------------------------------------------------


class TestExp3CancellationWiring:
    @pytest.mark.asyncio
    async def test_exp3b_respects_cancel_event(self, tmp_path):
        """When cancel_event is set before iteration, run_exp3b yields
        zero results (the outer loop checks cancel_event.is_set())."""
        import asyncio
        from src.conditioning.prompts import Condition
        from src.models import DryRunClient
        from src.runner import ExperimentConfig, ExperimentType
        from src.runners.exp3b import run_exp3b

        cancel = asyncio.Event()
        cancel.set()  # pre-cancel before any work

        client = DryRunClient(model="dry-run", responses=["hello"] * 50)
        config = ExperimentConfig(
            model_name="dry-run",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.COGNITIVE_SCOPE,
            num_runs=2,
            seed=0,
        )
        results = []
        async for r in run_exp3b(
            config, client,
            prompts=[{"id": "p1", "text": "hi"}],
            n_generations=2,
            output_dir=tmp_path,
            cancel_event=cancel,
        ):
            results.append(r)
        assert results == []

    @pytest.mark.asyncio
    async def test_exp3c_respects_cancel_event(self, tmp_path):
        import asyncio
        from src.conditioning.prompts import Condition
        from src.models import DryRunClient
        from src.runner import ExperimentConfig, ExperimentType
        from src.runners.exp3c import run_exp3c

        cancel = asyncio.Event()
        cancel.set()

        client = DryRunClient(model="dry-run", responses=["x"] * 50)
        config = ExperimentConfig(
            model_name="dry-run",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.CONSERVATIVE_SHIFT,
            num_runs=2,
            seed=0,
        )
        results = []
        async for r in run_exp3c(
            config, client,
            items=[{"difficulty": "easy", "question": "Q?", "expected": "A"}],
            output_dir=tmp_path,
            cancel_event=cancel,
        ):
            results.append(r)
        assert results == []

    @pytest.mark.asyncio
    async def test_exp3b_budget_caps_calls(self, tmp_path):
        """A BatchBudget with max_api_calls=4 halts run_exp3b before
        the matrix completes when client calls exceed 4."""
        from src.conditioning.prompts import Condition
        from src.models import DryRunClient
        from src.runner import (
            BatchBudget,
            BatchBudgetExceeded,
            ExperimentConfig,
            ExperimentType,
        )
        from src.runners.exp3b import run_exp3b

        client = DryRunClient(model="dry-run", responses=["x"] * 100)
        config = ExperimentConfig(
            model_name="dry-run",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.COGNITIVE_SCOPE,
            num_runs=1,
            num_conditioning_turns=5,
            seed=0,
        )
        # 5 conditioning calls + at least 1 generation call > budget of 4.
        budget = BatchBudget(max_api_calls=4)

        with pytest.raises(BatchBudgetExceeded):
            async for _ in run_exp3b(
                config, client,
                prompts=[{"id": "p1", "text": "hi"}],
                n_generations=3,
                output_dir=tmp_path,
                budget=budget,
            ):
                pass
