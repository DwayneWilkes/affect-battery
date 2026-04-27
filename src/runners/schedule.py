"""Cross-experiment scheduling primitives.

Per persistence-dynamics spec "Neutral control runs alongside",
schedule_exp2_with_controls emits a flat plan of (condition,
N, run_idx) cells covering both treatment arms and paired NEUTRAL
controls at each N. The plan is consumed by Exp 2 batch driver(s) so
controls run alongside treatment, not as a separate post-hoc run.

Pairing convention: 1:1 by (model, N). For each (treatment_arm, N) we
emit num_runs_per_cell entries; we ALSO emit max(num_runs_per_cell,
treatment_count_at_N) NEUTRAL entries at the same N so analyses can
pair runs by index without imbalance bias.
"""

from __future__ import annotations

from src.conditioning.prompts import Condition


def schedule_exp2_with_controls(
    conditions: list[Condition],
    n_values: list[int],
    num_runs_per_cell: int,
) -> list[dict]:
    """Build a list of run-plan cells: {condition, n_value, run_idx}.

    Treatment cells: cartesian product of `conditions` x `n_values` x
    range(num_runs_per_cell).
    Control cells: for each N value present in the treatment plan, emit
    enough NEUTRAL cells to match the treatment count at that N.
    """
    if num_runs_per_cell <= 0:
        raise ValueError(
            f"num_runs_per_cell must be > 0; got {num_runs_per_cell}"
        )
    if not n_values:
        raise ValueError("n_values must be non-empty")
    if not conditions:
        raise ValueError("conditions must be non-empty")

    plan: list[dict] = []

    # Treatment cells.
    treatment_count_per_n: dict[int, int] = {}
    for cond in conditions:
        if cond == Condition.NEUTRAL:
            # Skip; NEUTRAL is handled in the control loop below.
            continue
        for n in n_values:
            treatment_count_per_n[n] = (
                treatment_count_per_n.get(n, 0) + num_runs_per_cell
            )
            for run_idx in range(num_runs_per_cell):
                plan.append({
                    "condition": cond.value,
                    "n_value": n,
                    "run_idx": run_idx,
                })

    # Paired NEUTRAL control cells per N. Count = max(treatment_count_at_N,
    # num_runs_per_cell) so controls aren't underpowered when only one
    # treatment arm is scheduled at a given N.
    for n in n_values:
        control_count = max(
            treatment_count_per_n.get(n, 0),
            num_runs_per_cell,
        )
        for run_idx in range(control_count):
            plan.append({
                "condition": Condition.NEUTRAL.value,
                "n_value": n,
                "run_idx": run_idx,
            })

    return plan
