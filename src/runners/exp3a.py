"""Exp 3a — nonlinear arousal-performance (paper §3.4.1).

Single-turn intensity-stimulus paradigm. Each (level, run) cell delivers
INTENSITY_LEVELS[level-1].feedback_text as the system message and one
disjoint sample item from the configured task bank as the user message.
Per-cell binary correctness (0.01 tolerance against the bank's expected
field) is recorded on Exp3aBody.

Pilot-seed gate: before any cells dispatch, run_exp3a validates that
the SHA-256 in the seed JSON matches the canonicalized payload digest.
A tampered seed raises ValueError. This enforces the pre-registration
chain: pilot pass -> seed emitted -> Exp 3a runs gated on seed integrity.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from src.banks.loader import load_bank_items
from src.banks.sampling import sample_items
from src.conditioning.prompts import INTENSITY_LEVELS
from src.runner import Exp3aBody, ExperimentType, RunResult, save_result
from src.scoring.accuracy import extract_numeric_answer
from src.util import canonical_json_bytes


def _validate_pilot_seed(seed_path: Path) -> dict:
    payload = json.loads(Path(seed_path).read_text())
    sha = payload.pop("sha256", None)
    if sha is None:
        raise ValueError(f"pilot-seed {seed_path} missing 'sha256' field")
    expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if expected != sha:
        raise ValueError(
            f"pilot-seed SHA mismatch: file says {sha}, recomputed {expected}; "
            f"seed has been modified since emission"
        )
    return payload


def _score(response: str, expected: str) -> int:
    """Binary correctness with tolerance 0.01 against the bank's expected."""
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return 0
    try:
        target = float(expected)
    except (TypeError, ValueError):
        return 0
    return int(abs(extracted - target) < 0.01)


async def run_exp3a(
    config,
    client,
    intensity_levels: list[int],
    pilot_seed_path: Path,
    output_dir: Path | None = None,
    **_kwargs,
):
    """Run Exp 3a across the configured intensity levels."""
    if config.experiment_type != ExperimentType.AROUSAL_PERFORMANCE:
        raise ValueError(
            f"run_exp3a requires config.experiment_type=AROUSAL_PERFORMANCE; "
            f"got {config.experiment_type!r}"
        )
    if not intensity_levels:
        raise ValueError("intensity_levels must be non-empty")

    _validate_pilot_seed(pilot_seed_path)

    if not config.transfer_bank:
        raise ValueError("run_exp3a requires config.transfer_bank to be set")
    items = load_bank_items(config.transfer_bank)
    per_level = sample_items(
        items,
        n_per_level=config.num_runs,
        n_levels=len(intensity_levels),
        seed=config.seed or 0,
    )

    base_dir = Path(output_dir) if output_dir else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for level_idx, level in enumerate(intensity_levels):
        level_dir = (base_dir / f"level_{level}") if base_dir else None
        if level_dir is not None:
            level_dir.mkdir(parents=True, exist_ok=True)
        intensity_text = INTENSITY_LEVELS[level - 1].feedback_text
        for run_number, item in enumerate(per_level[level_idx]):
            messages = [
                {"role": "system", "content": intensity_text},
                {"role": "user", "content": item["question"]},
            ]
            start = time.time()
            response = await client.complete(
                messages, temperature=config.temperature, max_tokens=512,
            )
            end = time.time()

            expected = str(item["expected"])
            binary = _score(response, expected)

            result = RunResult(
                config=asdict(config),
                run_number=run_number,
                experiment_type="exp3a",
                model=config.model_name,
                condition=config.condition.value,
                start_time=start,
                end_time=end,
                body=Exp3aBody(
                    intensity_level=level,
                    model_response=response,
                    expected_answer=expected,
                    binary_correct=binary,
                ),
            )
            result.compute_checksum()
            if level_dir is not None:
                save_result(result, level_dir)
            yield result
