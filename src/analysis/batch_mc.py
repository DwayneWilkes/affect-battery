"""batch manipulation-check across paper §3.1 models.

Iterates strong-positive / strong-negative / no-conditioning conditions
on each model in the paper §3.1 set, computes per-model MC verdict via
the existing src.analysis.stats.manipulation_check function.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Awaitable, Callable

from src.analysis.stats import (
    ManipulationVerdict,
    manipulation_check,
)
from src.conditioning.prompts import Condition
from src.runner import ExperimentConfig, ExperimentType, run_batch


PAPER_3_1_MC_CONDITIONS = (
    Condition.STRONG_POSITIVE,
    Condition.STRONG_NEGATIVE,
    Condition.NO_CONDITIONING,
)


def _accuracy(correct_list: list[bool]) -> float:
    if not correct_list:
        return 0.0
    return sum(1 for c in correct_list if c) / len(correct_list)


async def run_batch_manipulation_check(
    models: list[str],
    client_factory: Callable[[str], object],
    n_per_condition: int = 10,
    output_dir: Path | None = None,
) -> dict[str, ManipulationVerdict]:
    """Run MC across all paper-§3.1 models, return verdict map.

    Args:
        models: list of model names (typically PAPER_3_1_MODELS).
        client_factory: callable(model_name) -> ModelClient.
        n_per_condition: runs per condition per model.
        output_dir: where to write per-model MC JSON results.

    Returns:
        dict mapping model name -> ManipulationVerdict.
    """
    output_dir = Path(output_dir) if output_dir else Path("results/mc_batch")
    output_dir.mkdir(parents=True, exist_ok=True)

    verdicts: dict[str, ManipulationVerdict] = {}

    for model_name in models:
        client = client_factory(model_name)
        accuracy_by_condition: dict[str, list[float]] = {}

        for cond in PAPER_3_1_MC_CONDITIONS:
            config = ExperimentConfig(
                model_name=model_name,
                condition=cond,
                experiment_type=ExperimentType.TRANSFER_WITHIN,
                num_runs=n_per_condition,
                temperature=0.7,
                seed=42,
            )
            run_accs: list[float] = []
            async for r in run_batch(
                config, client, output_dir=output_dir / "_runs"
            ):
                cond_correct = (
                    r.body.conditioning_correct
                    if r.body and hasattr(r.body, "conditioning_correct")
                    else r.conditioning_correct
                )
                run_accs.append(_accuracy(cond_correct))
            accuracy_by_condition[cond.value] = run_accs

        try:
            mc = manipulation_check(
                accuracy_by_condition=accuracy_by_condition,
                model=model_name,
            )
            verdicts[model_name] = mc.verdict
            (output_dir / f"mc_{model_name.replace('/', '__')}.json").write_text(
                json.dumps({
                    "model": model_name,
                    "verdict": mc.verdict.value if hasattr(mc.verdict, "value") else str(mc.verdict),
                    "accuracy_by_condition": mc.accuracy_by_condition,
                    "max_delta_pp": mc.max_delta_pp,
                    "annotation": mc.annotation,
                }, indent=2)
            )
        except (ValueError, KeyError) as e:
            verdicts[model_name] = ManipulationVerdict.UNAVAILABLE
            (output_dir / f"mc_{model_name.replace('/', '__')}.json").write_text(
                json.dumps({
                    "model": model_name,
                    "verdict": "unavailable",
                    "error": str(e),
                }, indent=2)
            )

    return verdicts
