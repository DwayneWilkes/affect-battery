"""Base-feasibility decision record (Task 1.4).

Per base-model-comparison spec "Week-1 go/no-go gate for base-model
feasibility": if Llama-3-8B base manipulation-check accuracy < threshold,
demote H4 base-vs-instruct test to exploratory; primary family shrinks
from 5 (incl. H1b-TOST) to 4.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def apply_decision(
    probe_verdict: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Apply the base-feasibility outcome to the OSF pre-registration.

    Args:
        probe_verdict: "pass" | "fail" from BaseModelProbeResult.
        output_dir: Where to write the decision record.

    Returns:
        Amendment dict with h4_status + primary_family_size.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if probe_verdict == "pass":
        h4_status = "primary"
        primary_family_size = 5  # H1, H2, H3a, H4, H1b-TOST
        rationale = (
            "Base-model feasibility passed (Llama-3-8B base baseline "
            "accuracy ≥ 0.30). H4 base-vs-instruct asymmetry test "
            "remains in primary family. No OSF amendment required "
            "beyond the standard probe-result archival."
        )
    elif probe_verdict == "fail":
        h4_status = "exploratory"
        primary_family_size = 4  # H1, H2, H3a, H1b-TOST (no H4)
        rationale = (
            "Base-model feasibility FAILED (Llama-3-8B base baseline "
            "accuracy below 0.30). H4 base-vs-instruct asymmetry test "
            "demoted to exploratory; primary analyses proceed with "
            "instruct-only models. OSF pre-registration amended with "
            "this decision per base-model-comparison spec Week-1 gate."
        )
    else:
        raise ValueError(
            f"probe_verdict must be 'pass' or 'fail'; got {probe_verdict!r}"
        )

    amendment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "probe_verdict": probe_verdict,
        "h4_status": h4_status,
        "primary_family_size": primary_family_size,
        "rationale": rationale,
    }

    timestamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    out_path = output_dir / f"base_feasibility_decision_{timestamp}.json"
    out_path.write_text(json.dumps(amendment, indent=2))

    return amendment
