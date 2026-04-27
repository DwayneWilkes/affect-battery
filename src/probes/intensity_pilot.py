"""Intensity-axis pilot protocol.

Per conditioning-protocol spec "Two intensity axes for Experiment 3a" +
"Intensity-axis pilot-as-gate for Exp 3a": collect ratings from 3 raters
across N items at 7 intensity levels, compute Krippendorff α (overall +
per-pair), decide whether to proceed / collapse / restructure.

Decision rule (per spec):
- α >= 0.8 overall AND all adjacent pairs >= 0.6 -> proceed
- α < 0.6 overall OR any adjacent pair < 0.6 -> collapse (collapse to fewer
  levels per the spec's collapse scenario)
- Otherwise (0.6 <= α < 0.8 overall, all pairs >= 0.6) -> restructure
  (refine prompts and re-run pilot)

Krippendorff implementation: krippendorff library (pip install
krippendorff). DRY check not rolled by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import krippendorff

from src.util import canonical_json_bytes


PROCEED_THRESHOLD = 0.8
COLLAPSE_THRESHOLD = 0.6


def _alpha(ratings_matrix: list[list[float]], level_of_measurement: str = "ordinal") -> float:
    """Wrap krippendorff.alpha with the project's convention defaults.

    `ratings_matrix` is shape (n_raters, n_items) per krippendorff API."""
    return float(krippendorff.alpha(
        reliability_data=ratings_matrix,
        level_of_measurement=level_of_measurement,
    ))


def run_intensity_pilot(ratings: dict[str, list[int]]) -> dict:
    """Compute Krippendorff α from rater ratings + return decision.

    `ratings` maps rater_id -> list of integer ratings across items.
    All raters must rate the same number of items in the same order.
    """
    if len(ratings) != 3:
        raise ValueError(
            f"Pilot requires exactly 3 raters; got {len(ratings)}"
        )

    rater_names = sorted(ratings.keys())
    matrix = [list(ratings[r]) for r in rater_names]
    item_counts = {len(row) for row in matrix}
    if len(item_counts) != 1:
        raise ValueError(
            f"All raters must rate the same number of items; "
            f"got per-rater counts {item_counts}"
        )

    alpha_overall = _alpha(matrix)

    # Pairwise alphas
    pairwise: dict[str, float] = {}
    for i in range(len(rater_names)):
        for j in range(i + 1, len(rater_names)):
            pair_matrix = [matrix[i], matrix[j]]
            pairwise[f"{rater_names[i]}__{rater_names[j]}"] = _alpha(pair_matrix)

    if alpha_overall < COLLAPSE_THRESHOLD or any(
        v < COLLAPSE_THRESHOLD for v in pairwise.values()
    ):
        decision = "collapse"
    elif alpha_overall >= PROCEED_THRESHOLD:
        decision = "proceed"
    else:
        decision = "restructure"

    return {
        "n_raters": len(ratings),
        "n_items": next(iter(item_counts)),
        "alpha_overall": alpha_overall,
        "alpha_pairwise": pairwise,
        "decision": decision,
        "raters": rater_names,
    }


def emit_seed(
    pilot_result: dict,
    axis_id: str,
    n_levels: int,
    pilot_date: str,
    output_path: Path,
) -> Path:
    """Write a signed JSON artifact recording pilot-pass.

    Per conditioning-protocol spec "Pre-registration seed definition":
    seed must include pilot_date, axis_id, n_levels, alpha_overall,
    alpha_pairwise + a SHA-256 digest over the canonicalized payload.
    The seed is the input to the OSF amendment that opens Exp 3a; downstream
    runners (run_exp3a) re-compute the SHA from the file and compare.
    """
    if pilot_result.get("decision") != "proceed":
        raise ValueError(
            f"emit_seed only writes seeds when pilot decision is 'proceed'; "
            f"got {pilot_result.get('decision')!r}"
        )
    payload = {
        "pilot_date": pilot_date,
        "axis_id": axis_id,
        "n_levels": n_levels,
        "alpha_overall": pilot_result["alpha_overall"],
        "alpha_pairwise": pilot_result["alpha_pairwise"],
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    payload["sha256"] = digest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path
