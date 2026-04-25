"""Exp 3c — conservative shift (paper §3.4.3).

Per conservative-shift-measurement spec "Conservative-shift protocol" +
"Question-difficulty stratification" + tasks.md Task 7.3: Exp 3c runs
3 conditions x factual-QA items x N runs, stratified by difficulty
(easy/medium/hard). Each yielded RunResult carries an Exp3cBody recording
(difficulty, question, response, expected, stated_confidence, refused).

DRY check: factual-QA loader and conditioning protocol are reused;
new logic is per-item dispatch + Exp3cBody construction.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from src.runner import (
    Exp3cBody,
    ExperimentType,
    RunResult,
    save_result,
)


VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})


async def run_exp3c(
    config,
    client,
    items: list[dict],
    output_dir: Path | None = None,
    **kwargs,
):
    """Run Exp 3c across `items` (each with difficulty/question/expected)."""
    if config.experiment_type != ExperimentType.CONSERVATIVE_SHIFT:
        raise ValueError(
            f"run_exp3c requires config.experiment_type=CONSERVATIVE_SHIFT; "
            f"got {config.experiment_type!r}"
        )
    if not items:
        raise ValueError("items must be non-empty")

    for item in items:
        if item.get("difficulty") not in VALID_DIFFICULTIES:
            raise ValueError(
                f"invalid difficulty {item.get('difficulty')!r}; "
                f"must be one of {sorted(VALID_DIFFICULTIES)}"
            )

    base_dir = Path(output_dir) if output_dir else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for run_num in range(config.num_runs):
        for item in items:
            response = await client.complete(
                [{"role": "user", "content": item["question"]}],
                temperature=config.temperature,
            )
            # Refusal heuristic: response begins with classic refusal cues
            response_lower = response.lower().strip()
            refused = any(
                response_lower.startswith(prefix)
                for prefix in (
                    "i cannot", "i can't", "i'm not able", "i don't",
                    "i am not able", "i am unable", "i refuse",
                )
            )
            body = Exp3cBody(
                difficulty=item["difficulty"],
                question=item["question"],
                response=response,
                expected=item["expected"],
                stated_confidence=None,
                refused=refused,
            )
            result = RunResult(
                config=asdict(config),
                run_number=run_num,
                experiment_type=config.experiment_type.value,
                model=config.model_name,
                condition=config.condition.value,
                start_time=time.time(),
                end_time=time.time(),
                body=body,
            )
            result.compute_checksum()
            if base_dir is not None:
                save_result(result, base_dir)
            yield result
