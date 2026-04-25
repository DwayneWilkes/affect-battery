"""Preflight checks for the sweep launcher.

Pure-function guard that cmd_run invokes before launching a sweep.
Refuses to launch when calibration prerequisites aren't met.

Checks (in order):
    1. A calibration report exists for the selected bank_id under
       `artifacts/calibration-<bank_id>-*.json`. Absence => PreflightError.
    2. The current gate config's content hash matches the hash recorded
       in the most recent calibration report. Mismatch => PreflightError
       (someone edited gate config after calibration ran; pre-registration
       broken).
    3. The sweep's seed range is disjoint from every seed range in the
       calibration report's `seed_ranges_used`. Overlap => PreflightError
       (stimulus leakage: bank tuning on test data).
    4. The gate config's `pre_registration_tag` and `pre_registration_sha`
       are not the `REPLACE_BEFORE_CALIBRATION` sentinel. Placeholder =>
       PreflightError (pilot ran without external pre-registration).

Spec: affect-battery-task-difficulty-calibration::task-difficulty-calibration::
"Pre-flight go/no-go gate" + sweep-launcher preflight. Group 12.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


PRE_REGISTRATION_SENTINEL: str = "REPLACE_BEFORE_CALIBRATION"


class PreflightError(RuntimeError):
    """Raised when a sweep-launcher preflight check refuses to let the
    sweep run. The str representation is the error message cmd_run
    surfaces to the user."""


def _load_latest_calibration_report(artifacts_dir: Path, bank_id: str) -> dict:
    """Return the most recent calibration report for the given bank, or
    raise PreflightError if none exists."""
    if not artifacts_dir.exists():
        raise PreflightError(
            f"no calibration report found for bank '{bank_id}': "
            f"artifacts directory {artifacts_dir} does not exist. "
            f"Run scripts/auto_calibrate_arithmetic.py before launching "
            f"the sweep."
        )
    matches = sorted(artifacts_dir.glob(f"calibration-{bank_id}-*.json"))
    if not matches:
        raise PreflightError(
            f"no calibration report found for bank '{bank_id}' under "
            f"{artifacts_dir}. Run scripts/auto_calibrate_arithmetic.py "
            f"before launching the sweep."
        )
    latest = matches[-1]  # lexical sort of -YYYY-MM-DD.json gives most recent
    try:
        return json.loads(latest.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise PreflightError(
            f"failed to read calibration report {latest}: {e}"
        )


def _check_gate_config_hash(report: dict, current_config_hash: str) -> None:
    recorded = (report.get("gate_verdict") or {}).get("config_hash", "")
    if recorded != current_config_hash:
        raise PreflightError(
            f"gate config hash mismatch: "
            f"calibration recorded '{recorded}', "
            f"current is '{current_config_hash}'. "
            f"configs/calibration-gate.yaml has been edited since "
            f"calibration ran — pre-registration is broken. "
            f"Re-run calibration OR revert the gate config."
        )


def _check_seed_disjointness(report: dict, sweep_seeds: Iterable[int]) -> None:
    seed_ranges = report.get("seed_ranges_used", [])
    sweep_set = set(sweep_seeds)
    for start, end in seed_ranges:
        calibration_set = set(range(int(start), int(end) + 1))
        overlap = sweep_set & calibration_set
        if overlap:
            sample = sorted(overlap)[:5]
            raise PreflightError(
                f"sweep seed range overlaps calibration seed range "
                f"[{start}, {end}]: overlapping seeds include {sample}. "
                f"Sweep seeds MUST be disjoint from calibration seeds to "
                f"prevent the hard bank from being tuned on the test set."
            )


def _check_pre_registration_not_sentinel(
    tag: str, sha: str,
) -> None:
    if tag == PRE_REGISTRATION_SENTINEL or sha == PRE_REGISTRATION_SENTINEL:
        raise PreflightError(
            f"gate config's pre-registration fields still carry the "
            f"'{PRE_REGISTRATION_SENTINEL}' placeholder. Per design D4 the "
            f"sweep MUST NOT run until the gate config has been tagged in "
            f"git (tag: gate-prereg-<bank_id>-<YYYY-MM-DD>) and the tag "
            f"commit SHA recorded in configs/calibration-gate.yaml. "
            f""
        )


def preflight_checks(
    bank_id: str,
    artifacts_dir: Path,
    gate_config_path: Path,
    current_config_hash: str,
    current_pre_reg_tag: str,
    current_pre_reg_sha: str,
    sweep_seeds: Iterable[int],
) -> None:
    """Run all preflight checks. Raises PreflightError on any failure.

    `gate_config_path` is accepted for future extensions (e.g., inline
    re-hashing); current implementation uses `current_config_hash` directly
    so tests don't need to construct a file whose content hashes to an
    arbitrary string.
    """
    _check_pre_registration_not_sentinel(current_pre_reg_tag, current_pre_reg_sha)
    report = _load_latest_calibration_report(Path(artifacts_dir), bank_id)
    _check_gate_config_hash(report, current_config_hash)
    _check_seed_disjointness(report, sweep_seeds)
