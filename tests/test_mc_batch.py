"""Manipulation-check batch + exclusion filter.

Per scoring-pipeline spec "Manipulation check gate": every paper §3.1
model runs the manipulation check before primary data collection.
Failing MC excludes the model from primary analysis.
"""

import pytest

from src.analysis.stats import ManipulationVerdict


PAPER_3_1_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/Gemma-2-9B-IT",
    "meta-llama/Meta-Llama-3-8B",
]


class TestRunBatchManipulationCheck:
    @pytest.mark.asyncio
    async def test_runs_all_paper_models(self, tmp_path):
        from src.analysis.batch_mc import run_batch_manipulation_check
        from src.models import DryRunClient

        clients = {m: DryRunClient(model=m) for m in PAPER_3_1_MODELS}
        verdicts = await run_batch_manipulation_check(
            models=PAPER_3_1_MODELS,
            client_factory=lambda m: clients[m],
            n_per_condition=3,
            output_dir=tmp_path,
        )
        assert set(verdicts.keys()) == set(PAPER_3_1_MODELS)


class TestPrimaryAnalysisExclusion:
    """ — exclude MC-failed models from primary corpus."""

    def test_passing_model_is_eligible(self):
        from src.analysis.exclusion import is_model_eligible_for_primary

        assert is_model_eligible_for_primary(
            model="m",
            mc_verdict=ManipulationVerdict.PASS,
        ) is True

    def test_failing_model_is_excluded(self):
        from src.analysis.exclusion import is_model_eligible_for_primary

        assert is_model_eligible_for_primary(
            model="m",
            mc_verdict=ManipulationVerdict.FAIL,
        ) is False

    def test_unavailable_verdict_eligible(self):
        """UNAVAILABLE is a measurement gap (no NO_CONDITIONING data),
        not a failed-conditioning verdict, so the model still participates
        in transfer analysis."""
        from src.analysis.exclusion import is_model_eligible_for_primary

        assert is_model_eligible_for_primary(
            model="m",
            mc_verdict=ManipulationVerdict.UNAVAILABLE,
        ) is True

    def test_filter_results_by_mc_verdicts(self):
        """Given a list of result-tuples + a verdict map, return only the
        results from MC-passing models."""
        from src.analysis.exclusion import filter_results_by_mc

        results = [
            {"model": "llama", "data": "a"},
            {"model": "mistral", "data": "b"},
            {"model": "gemma", "data": "c"},
        ]
        verdicts = {
            "llama": ManipulationVerdict.PASS,
            "mistral": ManipulationVerdict.FAIL,
            "gemma": ManipulationVerdict.PASS,
        }
        kept = filter_results_by_mc(results, verdicts)
        assert [r["model"] for r in kept] == ["llama", "gemma"]
