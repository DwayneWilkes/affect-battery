"""Budget-contingency decision record.

Per power-analysis spec "Budget-contingency decision record" (resolves
review AI-2 / VA-M2):

If simulation recommends per-condition n exceeding §8 ~$200 / ~15k-call
budget ceiling, OSF amendment records ONE of three priority-ordered
options. Silent under-powered run is NEVER permitted.

Options:
  (a) defer non-primary experiment to the project follow-up queue
  (b) accept lower power on H3b/H3c (secondary) + cite paper §6 all-
      conditions reporting commitment
  (c) request additional compute budget before advancing
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


VALID_OPTIONS = {"a", "b", "c"}


_RATIONALES = {
    "a": (
        "Defer a non-primary experiment to a named follow-up change in "
        "follow-up queue. Preserves primary H1/H2/H3a power. Per power-analysis "
        "spec budget-contingency option (a)."
    ),
    "b": (
        "Accept lower power for secondary hypotheses (H3b, H3c). Document "
        "the trade-off in OSF amendment. Paper §6 all-conditions reporting "
        "commitment still holds — final results report includes all "
        "conditions for all experiments. Per power-analysis spec "
        "budget-contingency option (b)."
    ),
    "c": (
        "Request additional compute budget before advancing. Pending "
        "approval blocks data collection. Per power-analysis spec "
        "budget-contingency option (c)."
    ),
}


def emit_decision(
    recommended_n_per_condition: int,
    budget_ceiling_n: int,
    output_dir: Path,
    chosen_option: str | None = None,
) -> dict[str, Any] | None:
    """Emit a budget-contingency decision record if n > ceiling.

    Returns None if within budget (no decision required). Otherwise
    writes a YAML record at
    `<output_dir>/budget_contingency_<timestamp>.yaml` and returns
    the decision dict.

    Raises:
        ValueError: if chosen_option not in {"a", "b", "c"}.
    """
    if recommended_n_per_condition <= budget_ceiling_n:
        return None

    if chosen_option not in VALID_OPTIONS:
        raise ValueError(
            f"chosen_option must be one of {sorted(VALID_OPTIONS)}; "
            f"got {chosen_option!r}. See power-analysis spec "
            "'Budget-contingency decision record'."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decision: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recommended_n_per_condition": recommended_n_per_condition,
        "budget_ceiling_n": budget_ceiling_n,
        "chosen_option": chosen_option,
        "rationale": _RATIONALES[chosen_option],
    }

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out_path = output_dir / f"budget_contingency_{timestamp}.yaml"
    out_path.write_text(yaml.safe_dump(decision, sort_keys=True))

    return decision
