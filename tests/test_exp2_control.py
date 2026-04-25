"""Task 5.2 Red — neutral-conditioning control runs.

Per persistence-dynamics spec "Neutral control runs alongside": for each
strong-positive and strong-negative run at a given N, a paired neutral-
conditioning control run exists at the same N. The scheduler emits the
control alongside the treatment so analyses can pair them by (model, N).
"""

from __future__ import annotations

import pytest


class TestSchedulePaired:
    def test_each_treatment_has_paired_neutral_control(self):
        from src.runner.schedule import schedule_exp2_with_controls
        from src.conditioning.prompts import Condition

        plan = schedule_exp2_with_controls(
            conditions=[Condition.STRONG_POSITIVE, Condition.STRONG_NEGATIVE],
            n_values=[1, 5, 10],
            num_runs_per_cell=2,
        )
        # For each (treatment, N) cell, there must be a paired (NEUTRAL, N) cell
        treatment_cells = {
            (cell["condition"], cell["n_value"]) for cell in plan
            if cell["condition"] != Condition.NEUTRAL.value
        }
        control_cells = {
            (cell["condition"], cell["n_value"]) for cell in plan
            if cell["condition"] == Condition.NEUTRAL.value
        }
        # Every (treatment, N) implies a (NEUTRAL, N) entry exists
        for _, n in treatment_cells:
            assert (Condition.NEUTRAL.value, n) in control_cells, (
                f"Missing paired neutral control for N={n}"
            )

    def test_neutral_control_count_per_n_matches_treatment_count(self):
        """For each N, the number of neutral-control runs equals the
        max of treatment-arm runs at that N (so analyses can pair 1:1)."""
        from src.runner.schedule import schedule_exp2_with_controls
        from src.conditioning.prompts import Condition

        plan = schedule_exp2_with_controls(
            conditions=[Condition.STRONG_POSITIVE, Condition.STRONG_NEGATIVE],
            n_values=[3],
            num_runs_per_cell=4,
        )
        # 4 strong-positive + 4 strong-negative + at least 4 neutral controls
        n3 = [cell for cell in plan if cell["n_value"] == 3]
        neutral = [c for c in n3 if c["condition"] == Condition.NEUTRAL.value]
        assert len(neutral) >= 4

    def test_invalid_inputs_raise_value_error(self):
        from src.runner.schedule import schedule_exp2_with_controls
        from src.conditioning.prompts import Condition

        with pytest.raises(ValueError, match="num_runs_per_cell"):
            schedule_exp2_with_controls(
                conditions=[Condition.STRONG_POSITIVE],
                n_values=[1],
                num_runs_per_cell=0,
            )
