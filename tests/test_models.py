"""Tests for model clients."""

import asyncio
from src.models import DryRunClient


class TestDryRunClient:
    def test_cycles_responses(self):
        client = DryRunClient(responses=["a", "b", "c"])
        results = [asyncio.run(client.complete([])) for _ in range(5)]
        assert results == ["a", "b", "c", "a", "b"]

    def test_model_name(self):
        client = DryRunClient(model="test-model")
        assert client.model_name == "test-model"
