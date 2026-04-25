"""Exp 3a — nonlinear arousal-performance (paper §3.4.1).

Per conditioning-protocol spec "Intensity-axis pilot-as-gate for Exp 3a"
+ tasks.md Task 6.3: run_exp3a iterates intensity levels (typically
1..7) for the configured stimulus bank x model. Each yielded RunResult
carries an Exp3aBody recording the intensity_level used for that run.

Pilot-seed gate: before any runs dispatch, run_exp3a validates that the
SHA-256 in the seed JSON matches the canonicalized payload digest. A
tampered seed raises ValueError. This enforces the pre-registration
chain: pilot pass -> seed emitted -> Exp 3a runs gated on seed integrity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.runner import Exp3aBody, ExperimentType, run_batch


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _validate_pilot_seed(seed_path: Path) -> dict:
    payload = json.loads(Path(seed_path).read_text())
    sha = payload.pop("sha256", None)
    if sha is None:
        raise ValueError(f"pilot-seed {seed_path} missing 'sha256' field")
    expected = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if expected != sha:
        raise ValueError(
            f"pilot-seed SHA mismatch: file says {sha}, recomputed {expected}; "
            f"seed has been modified since emission"
        )
    return payload


async def run_exp3a(
    config,
    client,
    intensity_levels: list[int],
    pilot_seed_path: Path,
    output_dir: Path | None = None,
    **kwargs,
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

    base_dir = Path(output_dir) if output_dir else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    for level in intensity_levels:
        level_dir = (base_dir / f"level_{level}") if base_dir else None
        if level_dir is not None:
            level_dir.mkdir(parents=True, exist_ok=True)
        # run_batch dispatches via run_single's existing path. After yield,
        # we attach Exp3aBody recording the intensity_level for this slice.
        async for result in run_batch(config, client, output_dir=level_dir, **kwargs):
            result.body = Exp3aBody(
                intensity_level=level,
                transfer_responses=list(result.transfer_responses),
                transfer_expected=list(result.transfer_expected),
            )
            yield result
