"""Base-model feasibility probe — Week-0 go/no-go gate.

Per base-model-comparison spec "Week-1 go/no-go gate for base-model
feasibility": runs N GSM8K problems on the base model with no
conditioning. If baseline_accuracy < threshold (default 0.30), the
H4 base-vs-instruct test is demoted to exploratory and primary family
shrinks to 4 hypotheses.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runner import ExperimentConfig, ExperimentType, run_batch
from src.conditioning.prompts import Condition


DEFAULT_FEASIBILITY_THRESHOLD = 0.30


@dataclass
class BaseModelProbeResult:
    model: str
    baseline_accuracy: float
    n_problems: int
    feasibility_verdict: str  # "pass" | "fail"
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""


def _accuracy(correct_list: list[bool]) -> float:
    if not correct_list:
        return 0.0
    return sum(1 for c in correct_list if c) / len(correct_list)


async def run_base_model_probe(
    client: Any,
    model_name: str,
    n: int = 5,
    output_dir: Path | None = None,
    threshold: float = DEFAULT_FEASIBILITY_THRESHOLD,
) -> BaseModelProbeResult:
    """Run N GSM8K problems via NO_CONDITIONING + few-shot scaffold;
    decide pass/fail against threshold."""
    output_dir = Path(output_dir) if output_dir else Path("results/probes")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        model_name=model_name,
        condition=Condition.NO_CONDITIONING,
        experiment_type=ExperimentType.TRANSFER_WITHIN,
        num_runs=n,
        temperature=0.0,
        seed=42,
        is_base_model=True,
    )

    accs: list[float] = []
    async for r in run_batch(config, client, output_dir=output_dir / "_probe_runs"):
        # Use transfer-task accuracy as the baseline-arithmetic proxy
        # (NO_CONDITIONING runs have no conditioning phase; transfer phase
        # exercises the same arithmetic-style scoring).
        responses = (
            r.body.transfer_responses
            if r.body and hasattr(r.body, "transfer_responses")
            else r.transfer_responses
        )
        expected = (
            r.body.transfer_expected
            if r.body and hasattr(r.body, "transfer_expected")
            else r.transfer_expected
        )
        per_run_correct = [
            (resp or "").strip() == (exp or "").strip()
            for resp, exp in zip(responses, expected)
        ]
        if per_run_correct:
            accs.append(_accuracy(per_run_correct))

    baseline_accuracy = sum(accs) / len(accs) if accs else 0.0
    verdict = "pass" if baseline_accuracy >= threshold else "fail"

    result = BaseModelProbeResult(
        model=model_name,
        baseline_accuracy=baseline_accuracy,
        n_problems=n,
        feasibility_verdict=verdict,
        threshold=threshold,
        notes=(
            f"Base-model feasibility probe on {model_name}: ran {len(accs)} "
            f"NO_CONDITIONING runs. Threshold {threshold} = paper §3.1 "
            f"+ base-model-comparison spec Week-1 gate. Verdict: {verdict}."
        ),
    )

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out_path = output_dir / f"base_model_probe_{timestamp}.json"
    out_path.write_text(json.dumps(asdict(result), indent=2))

    return result
