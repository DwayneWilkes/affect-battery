"""Variance probe — Week-0 small-n manipulation-check for MDE grounding.

Per power-analysis spec "Week-1 pilot feeds a simulation-based power
analysis" + design.md D3 variance-probe-override:

Runs strong-positive vs neutral conditions on arithmetic. Computes
variance + observed effect size to inform MDE table updates.
Output JSON consumed by `src.power.mde.update_from_probe`.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runner import ExperimentConfig, ExperimentType, run_batch
from src.conditioning.prompts import Condition


@dataclass
class VarianceProbeResult:
    model: str
    variance_estimate: float
    std_err: float
    observed_effect_size: float
    n_per_condition: int
    conditions: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""


def _accuracy(correct_list: list[bool]) -> float:
    if not correct_list:
        return 0.0
    return sum(1 for c in correct_list if c) / len(correct_list)


async def run_variance_probe(
    client: Any,
    model_name: str,
    n_per_condition: int = 20,
    output_dir: Path | None = None,
    base_url: str | None = None,
) -> VarianceProbeResult:
    """Run a variance probe and write the result JSON.

    Args:
        client: ModelClient (live or DryRun).
        model_name: Model identifier (recorded in result).
        n_per_condition: Sample size per condition (default 20 per spec).
        output_dir: Where to write `variance_probe_<timestamp>.json`.
        base_url: Optional, recorded in result for reproducibility.

    Returns:
        VarianceProbeResult with variance_estimate, std_err, observed effect.
    """
    output_dir = Path(output_dir) if output_dir else Path("results/probes")
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracies_by_condition: dict[str, list[float]] = {}

    for condition in (Condition.STRONG_POSITIVE, Condition.NEUTRAL):
        config = ExperimentConfig(
            model_name=model_name,
            condition=condition,
            experiment_type=ExperimentType.TRANSFER_WITHIN,
            num_runs=n_per_condition,
            temperature=0.7,
            seed=42,
        )
        condition_accs: list[float] = []
        async for result in run_batch(config, client, output_dir=output_dir / "_probe_runs"):
            acc = _accuracy(result.body.transfer_responses if result.body and hasattr(result.body, "transfer_responses") else result.transfer_responses)
            # Use conditioning correctness as the variance signal for the probe
            cond_acc = _accuracy(
                result.body.conditioning_correct
                if result.body and hasattr(result.body, "conditioning_correct")
                else result.conditioning_correct
            )
            condition_accs.append(cond_acc)
        accuracies_by_condition[condition.value] = condition_accs

    pos_accs = accuracies_by_condition.get(Condition.STRONG_POSITIVE.value, [])
    neutral_accs = accuracies_by_condition.get(Condition.NEUTRAL.value, [])

    pooled = pos_accs + neutral_accs
    variance_estimate = statistics.pvariance(pooled) if len(pooled) >= 2 else 0.0

    pos_mean = statistics.mean(pos_accs) if pos_accs else 0.0
    neutral_mean = statistics.mean(neutral_accs) if neutral_accs else 0.0
    pooled_sd = (variance_estimate ** 0.5) if variance_estimate > 0 else 1e-6
    observed_effect_size = abs(pos_mean - neutral_mean) / pooled_sd if pooled_sd > 0 else 0.0

    se = (
        (pooled_sd / (len(pooled) ** 0.5))
        if pooled
        else 0.0
    )

    result = VarianceProbeResult(
        model=model_name,
        variance_estimate=variance_estimate,
        std_err=se,
        observed_effect_size=observed_effect_size,
        n_per_condition=n_per_condition,
        conditions=[Condition.STRONG_POSITIVE.value, Condition.NEUTRAL.value],
        notes=(
            "Variance probe over conditioning accuracy across "
            f"{n_per_condition} runs × 2 conditions on {model_name}. "
            "Output consumed by src.power.mde.update_from_probe."
        ),
    )

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out_path = output_dir / f"variance_probe_{timestamp}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))

    return result
