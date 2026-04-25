"""Task 3.3 Red — Exp 1a batch executor across paper §3.1 models.

Per tasks.md Task 3.3 + base-model-comparison spec: driver iterates
{Llama-3-8B-Instruct, Mistral-7B-Instruct, Gemma-2-9B-IT, Llama-3-8B
base} × 6 conditions × n. Base-model branch uses few-shot scaffold
inference path; instruct branches use chat path.
"""

from pathlib import Path

import pytest


PAPER_3_1_MODELS_INSTRUCT = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/Gemma-2-9B-IT",
]
PAPER_3_1_MODELS_BASE = [
    "meta-llama/Meta-Llama-3-8B",
]
ALL_PAPER_3_1_MODELS = PAPER_3_1_MODELS_INSTRUCT + PAPER_3_1_MODELS_BASE

SIX_ARMS = [
    "strong_positive",
    "mild_negative",
    "strong_negative",
    "neutral",
    "no_conditioning",
    "accurate_negative",
]


class TestRunExp1aBatch:
    @pytest.mark.asyncio
    async def test_iterates_all_paper_3_1_models(self, tmp_path):
        from src.runners.batch_exp1a import run_exp1a_batch
        from src.models import DryRunClient

        clients = {m: DryRunClient(model=m, responses=["42"]) for m in ALL_PAPER_3_1_MODELS}
        run_count = await run_exp1a_batch(
            models=ALL_PAPER_3_1_MODELS,
            conditions=SIX_ARMS,
            client_factory=lambda m: clients[m],
            n_per_condition=1,
            output_dir=tmp_path,
        )
        # 4 models × 6 conditions × n=1 = 24 runs
        assert run_count == len(ALL_PAPER_3_1_MODELS) * len(SIX_ARMS) * 1

    @pytest.mark.asyncio
    async def test_base_model_branch_uses_completion_path(self, tmp_path):
        """Llama-3-8B base model triggers is_base_model=True in config."""
        from src.runners.batch_exp1a import run_exp1a_batch
        from src.models import DryRunClient
        import json

        client = DryRunClient(model="meta-llama/Meta-Llama-3-8B", responses=["42"])
        await run_exp1a_batch(
            models=["meta-llama/Meta-Llama-3-8B"],
            conditions=["neutral"],
            client_factory=lambda m: client,
            n_per_condition=1,
            output_dir=tmp_path,
        )
        # Find the result file + check is_base_model config
        result_files = list(tmp_path.rglob("*.json"))
        assert len(result_files) >= 1
        sample = json.loads(result_files[0].read_text())
        assert sample["config"]["is_base_model"] is True

    @pytest.mark.asyncio
    async def test_writes_per_model_subdirs(self, tmp_path):
        from src.runners.batch_exp1a import run_exp1a_batch
        from src.models import DryRunClient

        clients = {m: DryRunClient(model=m, responses=["42"]) for m in PAPER_3_1_MODELS_INSTRUCT}
        await run_exp1a_batch(
            models=PAPER_3_1_MODELS_INSTRUCT,
            conditions=["neutral"],
            client_factory=lambda m: clients[m],
            n_per_condition=1,
            output_dir=tmp_path,
        )
        # Subdir-per-model layout for clean grouping
        subdirs = {p.name for p in tmp_path.iterdir() if p.is_dir()}
        assert any("Llama" in d or "llama" in d for d in subdirs)
