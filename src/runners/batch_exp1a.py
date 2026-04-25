"""Task 3.3 — Exp 1a batch executor across paper §3.1 models.

Iterates {Llama-3-8B-Instruct, Mistral-7B-Instruct, Gemma-2-9B-IT,
Llama-3-8B base} × paper §3.2.1 6-arm condition set. Llama-3-8B base
uses few-shot scaffold (is_base_model=True); instruct families use
chat path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.runner import ExperimentConfig, ExperimentType, run_batch
from src.runners.exp1a import run_exp1a
from src.conditioning.prompts import Condition


# Heuristic: a model is a "base" (non-instruct) model if its name does
# NOT include any chat/instruct marker. Llama-3-8B base is the
# canonical example for paper §3.1.
INSTRUCT_MARKERS = ("instruct", "-it", "chat")


def is_base_model_name(model_name: str) -> bool:
    lowered = model_name.lower()
    return not any(marker in lowered for marker in INSTRUCT_MARKERS)


def _slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


async def run_exp1a_batch(
    models: list[str],
    conditions: list[str],
    client_factory: Callable[[str], object],
    n_per_condition: int = 50,
    output_dir: Path | None = None,
    seed: int = 42,
) -> int:
    """Run Exp 1a across `models` × `conditions` × `n_per_condition`.

    Per-model subdir layout: <output_dir>/<model-slug>/<condition>/<run-N>.json

    Returns total run count completed.
    """
    output_dir = Path(output_dir) if output_dir else Path("results/exp1a")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for model_name in models:
        is_base = is_base_model_name(model_name)
        client = client_factory(model_name)

        model_dir = output_dir / _slug(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        for cond_str in conditions:
            cond = Condition(cond_str)
            cond_dir = model_dir / cond_str
            cond_dir.mkdir(parents=True, exist_ok=True)

            config = ExperimentConfig(
                model_name=model_name,
                condition=cond,
                experiment_type=ExperimentType.TRANSFER_WITHIN,
                num_runs=n_per_condition,
                temperature=0.7,
                seed=seed,
                is_base_model=is_base,
            )

            async for _ in run_exp1a(config, client, output_dir=cond_dir):
                total += 1

    return total
