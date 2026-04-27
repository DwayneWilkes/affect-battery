"""End-to-end test for the auto-calibration pipeline:
ModelBackedProbe + AutoCalibrator + calibrated-bank writer.

 d.
"""

import asyncio
from pathlib import Path

import yaml

from src.calibration.pipeline import (
    ModelBackedProbe,
    run_calibration,
)
from src.calibration.auto_calibrator import (
    CalibratorConfig,
    SweetSpotResult,
)
from src.models import ModelClient


class ArithmeticDifficultyClient(ModelClient):
    """A mock completion client that answers arithmetic correctly iff the
    item's max digit_count is <= a threshold. Emulates "model gets easier
    items right, harder items wrong."

    per_operator_threshold: {operator: max_digit_count_model_can_solve}
    """

    def __init__(self, per_operator_threshold: dict[str, int]):
        self.thresholds = per_operator_threshold
        self.prompts_seen: list[str] = []

    @property
    def model_name(self) -> str:
        return "arithmetic-difficulty-mock"

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        raise NotImplementedError("calibration uses complete_text")

    async def complete_text(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        self.prompts_seen.append(prompt)
        # Parse the prompt to infer operator + operands + answer.
        # Prompt format is expected to be: "What is A OP B?\nAnswer:"
        parsed = self._parse_prompt(prompt)
        if parsed is None:
            return " 0"
        a, op, b, answer = parsed
        max_digits = max(len(str(a)), len(str(b)))
        threshold = self.thresholds.get(op, 0)
        if max_digits <= threshold:
            return f" {answer}"
        else:
            # Return a wrong answer (off by 1).
            return f" {answer + 1}"

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[int, str, int, int] | None:
        # Best-effort parser for prompts like "What is 1234 + 5678?\nAnswer: "
        # or "What is 42 - 17?\nAnswer: ". Case-insensitive, tolerant.
        body = prompt.lower().strip()
        if "what is" not in body:
            return None
        try:
            expr = body.split("what is", 1)[1].split("?", 1)[0].strip()
        except (IndexError, ValueError):
            return None
        for symbol, op_name in [("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div")]:
            if symbol in expr:
                parts = [p.strip() for p in expr.split(symbol, 1)]
                if len(parts) != 2:
                    return None
                try:
                    a = int(parts[0])
                    b = int(parts[1])
                except ValueError:
                    return None
                answer = {
                    "add": a + b, "sub": a - b,
                    "mul": a * b, "div": a // b,
                }[op_name]
                return a, op_name, b, answer
        return None


class TestCalibrationPipeline:
    def test_probe_reports_accuracy_matching_model_thresholds(self):
        """ModelBackedProbe generates items, queries the model, and returns
        the fraction correctly answered."""
        client = ArithmeticDifficultyClient({"add": 3})
        probe = ModelBackedProbe(client=client, n_items_per_probe=10)
        # Model gets 3-digit adds right, 4-digit adds wrong.
        acc_easy = probe.probe(operator="add", digit_level=3, seed=0)
        acc_hard = probe.probe(operator="add", digit_level=4, seed=0)
        assert acc_easy == 1.0, f"Expected 1.0 at digit_level=3, got {acc_easy}"
        assert acc_hard == 0.0, f"Expected 0.0 at digit_level=4, got {acc_hard}"

    def test_end_to_end_calibration_writes_bank_with_calibrated_specs(self, tmp_path):
        """Run_calibration orchestrates probe + calibrator + bank generator
        and writes a YAML whose operator_specs reflect the search results."""
        client = ArithmeticDifficultyClient({
            "add": 3,  # model solves 3-digit add, fails 4-digit
            "sub": 3,
            "mul": 2,  # model solves 2-digit mul, fails 3-digit
            "div": 2,
        })
        config = CalibratorConfig(
            target_min=0.40,
            target_max=1.00,
            digit_range=(2, 5),
            max_iter=5,
        )
        output_path = tmp_path / "bank.yaml"
        results = run_calibration(
            client=client,
            calibrator_config=config,
            output_path=output_path,
            n_items_per_probe=10,
            total_bank_items=100,
            bank_id="arithmetic_hard_v1",
        )
        # Verify the bank was written.
        assert output_path.exists()
        with output_path.open() as f:
            bank = yaml.safe_load(f)
        assert bank["bank_id"] == "arithmetic_hard_v1"
        assert bank["difficulty_profile"]["calibration_source"].startswith("auto-calibrated")
        specs = bank["difficulty_profile"]["operator_specs"]
        # All four operators should have been calibrated.
        assert set(specs.keys()) == {"add", "sub", "mul", "div"}
        # For this mock, add/sub sweet spot is digit_level=3 (100% correct).
        # mul/div sweet spot is digit_level=2.
        assert specs["add"]["digit_level"] == 3
        assert specs["sub"]["digit_level"] == 3
        assert specs["mul"]["digit_level"] == 2
        assert specs["div"]["digit_level"] == 2
        # Results dict mirrors the per-operator calibration result objects.
        assert "add" in results
        assert isinstance(results["add"], SweetSpotResult)
