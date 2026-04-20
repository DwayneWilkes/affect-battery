"""Tests for the base-model inference path in run_single / run_batch.

Spec: affect-battery-base-model-comparison::base-model-comparison.
"""

import asyncio
import json

from src.conditioning.prompts import Condition, FEEDBACK_SETS
from src.models import ModelClient
from src.runner import (
    ExperimentConfig,
    ExperimentType,
    is_valid_cached_result,
    run_batch,
    run_single,
    save_result,
)


class _ScriptedCompletionClient(ModelClient):
    """Fake VLLMCompletionClient: returns scripted text responses, records
    the prompt it was called with on each turn."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._i = 0
        self.prompts_seen: list[str] = []
        self.stops_seen: list[list[str] | None] = []

    @property
    def model_name(self) -> str:
        return "scripted-base"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        raise NotImplementedError("base path should use complete_text")

    async def complete_text(
        self, prompt: str, temperature: float = 0.7,
        max_tokens: int = 1024, stop=None,
    ) -> str:
        self.prompts_seen.append(prompt)
        self.stops_seen.append(stop)
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r


def _config(is_base_model: bool = True) -> ExperimentConfig:
    return ExperimentConfig(
        model_name="Qwen/Qwen2.5-7B",
        condition=Condition.STRONG_NEGATIVE,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=1,
        num_conditioning_turns=3,
        num_transfer_questions=2,
        seed=42,
        is_base_model=is_base_model,
    )


class TestExperimentConfigFlag:
    def test_is_base_model_defaults_false(self):
        c = ExperimentConfig(
            model_name="m", condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
        )
        assert c.is_base_model is False

    def test_is_base_model_settable(self):
        c = _config(is_base_model=True)
        assert c.is_base_model is True


class TestBaseModelPath:
    def test_produces_valid_run_result(self):
        cfg = _config()
        client = _ScriptedCompletionClient(
            responses=[" 115", " 86", " 621",  # conditioning answers (correct per seed=42)
                       " Paris", " 1989"],  # transfer answers
        )
        result = asyncio.run(run_single(cfg, client, 0))

        assert len(result.conditioning_responses) == 3
        assert len(result.transfer_responses) == 2
        assert result.checksum != ""

    def test_config_records_is_base_model(self):
        cfg = _config()
        client = _ScriptedCompletionClient(responses=[" 0"] * 10)
        result = asyncio.run(run_single(cfg, client, 0))
        assert result.config.get("is_base_model") is True

    def test_scaffold_includes_feedback_text(self):
        """The prompt on turn 2 must include turn-0 and turn-1 feedback
        from the condition's FeedbackSet, so the model sees the conditioning
        history as it generates the next answer."""
        cfg = _config()
        client = _ScriptedCompletionClient(
            responses=[" 115", " 86", " 621", " Paris", " 1989"],
        )
        asyncio.run(run_single(cfg, client, 0))

        # Look at the prompt submitted for turn 2 (index 2 in prompts_seen).
        prompt_turn_2 = client.prompts_seen[2]
        fs = FEEDBACK_SETS[Condition.STRONG_NEGATIVE]
        # Turn 0 feedback must appear (model saw it after its turn-0 answer).
        assert fs.turns[0].correct in prompt_turn_2, (
            f"Turn 0 feedback missing from turn 2 scaffold. "
            f"Expected: {fs.turns[0].correct!r}"
        )
        # Turn 1 feedback must also appear.
        assert fs.turns[1].correct in prompt_turn_2

    def test_stop_tokens_used(self):
        """complete_text should be called with a stop list containing
        'Human:' so the base model doesn't hallucinate the next turn."""
        cfg = _config()
        client = _ScriptedCompletionClient(responses=[" 0"] * 10)
        asyncio.run(run_single(cfg, client, 0))
        # Every call should include a stop token for 'Human:' or similar.
        for stop in client.stops_seen:
            assert stop is not None, "stop tokens must be passed"
            assert any("Human" in s for s in stop), f"expected Human stop, got {stop}"


class TestSaveRoundtrip:
    def test_base_model_result_saves_and_loads(self, tmp_path):
        cfg = _config()
        client = _ScriptedCompletionClient(responses=[" 0"] * 10)
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["config"].get("is_base_model") is True
        assert data["checksum"] == result.checksum

    def test_base_model_result_is_valid_cached(self, tmp_path):
        """Spec scenario: base-model output JSONs must pass
        is_valid_cached_result so the resume layer treats them as cacheable."""
        cfg = _config()
        client = _ScriptedCompletionClient(responses=[" 0"] * 10)
        result = asyncio.run(run_single(cfg, client, 0))
        path = save_result(result, tmp_path)
        assert is_valid_cached_result(path) is True


class TestCLISelectsCompletionPath:
    """Spec scenario: --base-model flag selects VLLMCompletionClient."""

    def test_cli_pilot_base_model_flag_uses_completion_client(self, monkeypatch):
        """When --base-model is set, cmd_pilot must construct a
        VLLMCompletionClient and set is_base_model on the config."""
        from src import cli
        from src.models import VLLMCompletionClient

        captured = {}

        def fake_vllm_completion_client(*args, **kwargs):
            captured["completion_client"] = True
            return VLLMCompletionClient(*args, **kwargs)

        def fake_vllm_client(*args, **kwargs):
            captured["chat_client"] = True

        monkeypatch.setattr(
            "src.models.VLLMCompletionClient", fake_vllm_completion_client
        )
        monkeypatch.setattr("src.models.VLLMClient", fake_vllm_client)

        # Intercept asyncio.run so we don't actually hit the API. Close the
        # coroutine explicitly to avoid "coroutine never awaited" warnings.
        def _fake_run(coro):
            coro.close()

        monkeypatch.setattr("asyncio.run", _fake_run)

        class Args:
            dry_run = False
            base_model = True
            model = "Qwen/Qwen2.5-7B"
            base_url = "http://localhost:8000/v1"
            temperature = 0.7
            output_dir = "/tmp/test"
            max_concurrent = 1
            budget_max_calls = None
            cost_per_call = None
            rate_limit_rps = None
            circuit_breaker_threshold = 5

        cli.cmd_pilot(Args())
        assert captured.get("completion_client") is True
        assert "chat_client" not in captured


class TestRunBatchBaseModel:
    """run_batch should dispatch to the base path when config.is_base_model
    is True, and its cached-skip / events behaviour should be unchanged."""

    def test_run_batch_base_produces_results(self, tmp_path):
        cfg = ExperimentConfig(
            model_name="Qwen/Qwen2.5-7B",
            condition=Condition.NEUTRAL,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=2,
            num_conditioning_turns=1,
            num_transfer_questions=1,
            seed=42,
            is_base_model=True,
        )
        client = _ScriptedCompletionClient(responses=[" 0"] * 20)

        async def go():
            out = []
            async for r in run_batch(cfg, client, output_dir=tmp_path, max_concurrent=1):
                out.append(r)
            return out

        results = asyncio.run(go())
        assert len(results) == 2
        for r in results:
            assert r.config.get("is_base_model") is True
