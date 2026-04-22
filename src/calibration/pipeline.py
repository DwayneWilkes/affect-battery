"""End-to-end calibration pipeline: probe + calibrator + bank writer.

The `ModelBackedProbe` wraps a `ModelClient` so the calibrator can measure
accuracy at a given (operator, digit_level). `run_calibration` orchestrates
per-operator calibration and writes the resulting bank YAML.

Spec: affect-battery-task-difficulty-calibration. Task 1.1e.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.calibration.auto_calibrator import (
    AutoCalibrator,
    CalibrationResult,
    CalibratorConfig,
    SweetSpotResult,
)
from src.calibration.generator import (
    GenSpec,
    OPERATOR_MIX,
    build_bank_yaml,
    generate_items_for_operator,
    write_bank,
)
from src.models import ModelClient


# ───────────────────────── prompt + answer extraction ──────────────────────────


_OPERATOR_SYMBOLS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


def default_prompt_fn(item: dict) -> str:
    """Bare arithmetic prompt used for probing. Intentionally minimal so
    the measurement captures the model's raw arithmetic ability without
    contamination from instruction-following quirks."""
    a, b = item["operands"]
    symbol = _OPERATOR_SYMBOLS[item["operator"]]
    return f"What is {a} {symbol} {b}?\nAnswer:"


_NUMERIC_RE = re.compile(r"-?\d[\d,]*")


def extract_numeric_answer(response: str) -> int | None:
    """Extract the first integer from the response. Returns None if no
    numeric token is present. Tolerates comma-separated thousands
    (e.g. "1,234") by stripping commas before int() conversion."""
    match = _NUMERIC_RE.search(response)
    if match is None:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


# ───────────────────────── model-backed probe ──────────────────────────


@dataclass
class ModelBackedProbe:
    """Probe backed by a real ModelClient: generates `n_items_per_probe`
    items per call, queries the model, extracts numeric answers, returns
    observed accuracy."""
    client: ModelClient
    n_items_per_probe: int = 20

    def probe(self, operator: str, digit_level: int, seed: int = 0) -> float:
        spec = GenSpec(digit_level=digit_level)
        rng = random.Random((seed, operator, digit_level).__hash__())
        items = generate_items_for_operator(
            operator=operator,
            spec=spec,
            count=self.n_items_per_probe,
            rng=rng,
            id_prefix=f"probe_{operator}_d{digit_level}",
        )
        correct = 0
        for item in items:
            prompt = default_prompt_fn(item)
            response = asyncio.run(self.client.complete_text(prompt))
            answer = extract_numeric_answer(response)
            if answer is not None and answer == item["answer"]:
                correct += 1
        return correct / self.n_items_per_probe


# ───────────────────────── end-to-end runner ──────────────────────────


def run_calibration(
    client: ModelClient,
    calibrator_config: CalibratorConfig,
    output_path: Path,
    n_items_per_probe: int = 20,
    total_bank_items: int = 300,
    bank_id: str = "arithmetic_hard_v1",
    bank_version: int = 1,
    calibration_source_label: str | None = None,
) -> dict[str, CalibrationResult]:
    """Run the calibrator per operator, then write a bank YAML whose
    per-operator specs are derived from the calibration results.

    Returns a dict mapping operator -> CalibrationResult for downstream
    reporting.

    For operators where the calibrator returns a BracketResult (no integer
    digit_level hits the target window), falls back to the easier_side
    (accuracy above target_max) digit_level so the bank is still generated.
    This is a deliberate conservative choice: better to under-shoot the
    window than to produce items the model can't solve at all. The bank's
    difficulty_profile flags the fallback so the reviewer knows.
    """
    probe = ModelBackedProbe(client=client, n_items_per_probe=n_items_per_probe)
    calibrator = AutoCalibrator(config=calibrator_config, probe=probe)

    results: dict[str, CalibrationResult] = {}
    operator_specs: dict[str, GenSpec] = {}
    fallback_operators: list[str] = []

    for operator in OPERATOR_MIX:
        result = calibrator.calibrate_operator(operator)
        results[operator] = result
        if isinstance(result, SweetSpotResult):
            chosen_digit_level = result.digit_level
        else:
            # Bracket fallback: prefer easier side (above target_max) over
            # harder side. If neither exists (range fully below window),
            # use the digit_range lower bound.
            if result.easier_side is not None:
                chosen_digit_level = result.easier_side.digit_level
            elif result.harder_side is not None:
                chosen_digit_level = result.harder_side.digit_level
            else:
                chosen_digit_level = calibrator_config.digit_range[0]
            fallback_operators.append(operator)
        operator_specs[operator] = GenSpec(digit_level=chosen_digit_level)

    # Generate the calibrated bank.
    rng = random.Random((bank_id, bank_version).__hash__())
    items: list[dict] = []
    idx = 1
    target_counts = _target_counts_for_mix(OPERATOR_MIX, total_bank_items)
    for operator in OPERATOR_MIX:
        target = target_counts[operator]
        bucket = generate_items_for_operator(
            operator=operator,
            spec=operator_specs[operator],
            count=target,
            rng=rng,
            id_prefix=bank_id,
            start_idx=idx,
        )
        items.extend(bucket)
        idx += len(bucket)

    # Annotate the calibration source so readers know whether the bank was
    # hand-tuned or empirically calibrated, and under what model.
    if calibration_source_label is None:
        label = f"auto-calibrated via {client.model_name}"
        if fallback_operators:
            label += (
                f" (bracket fallback for operators: "
                f"{', '.join(fallback_operators)})"
            )
        calibration_source_label = label

    bank = build_bank_yaml(
        items=items,
        bank_id=bank_id,
        bank_version=bank_version,
        operator_specs=operator_specs,
        calibration_source=calibration_source_label,
    )
    write_bank(bank, output_path)
    return results


def _target_counts_for_mix(mix: dict[str, float], total: int) -> dict[str, int]:
    counts = {op: round(total * frac) for op, frac in mix.items()}
    delta = total - sum(counts.values())
    first_op = next(iter(mix))
    counts[first_op] += delta
    return counts
