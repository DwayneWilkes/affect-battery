"""Pre-flight go/no-go gate for the calibration pilot.

Spec: affect-battery-task-difficulty-calibration, Requirement "Pre-flight go/no-go gate".
Design: D4 (deterministic YAML-driven verdict).

The gate is a PURE FUNCTION of a YAML config file plus a calibration-results
dict. No I/O during evaluation, no wall-clock reads, no implicit state. Given
identical inputs it MUST return an identical Verdict, including a SHA-256
content hash of the YAML bytes so the sweep launcher can detect post-hoc
config edits (D4 + risk "Gate config drift").

Public surface:
    GateConfig, Verdict, VerdictStatus, GateConfigError
    load_config(path) -> GateConfig
    evaluate(results, config) -> Verdict
"""
from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml


PRE_REG_SENTINEL = "REPLACE_BEFORE_CALIBRATION"
_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class GateConfigError(ValueError):
    """Raised when configs/calibration-gate.yaml is missing, malformed, or
    violates a pre-registration invariant (missing null_acceptance block,
    missing pre_registration_tag / pre_registration_sha, malformed SHA)."""


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineWindow:
    min: float
    max: float
    applies_to: str = "at_least_one_of [base, instruct]"


@dataclass(frozen=True)
class PipelineSanity:
    easy_regression_delta_floor_pp: float


@dataclass(frozen=True)
class NullAcceptance:
    baseline_window: tuple[float, float]
    delta_ceiling_pp: float
    min_n_per_condition: int


@dataclass(frozen=True)
class GateConfig:
    baseline_window: BaselineWindow
    pipeline_sanity: PipelineSanity
    null_acceptance: NullAcceptance
    pre_registration_tag: str
    pre_registration_sha: str
    config_hash: str


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


class VerdictStatus(str, Enum):
    """Three-state verdict: a string-valued enum so JSON round-trips cleanly
    in the calibration report."""

    PASS = "PASS"
    RECALIBRATE = "RECALIBRATE"
    PIPELINE_REGRESSION = "PIPELINE_REGRESSION"


@dataclass(frozen=True)
class Verdict:
    status: VerdictStatus
    justification: str
    config_hash: str
    null_accepted: bool = False


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def _require(mapping: Mapping[str, Any] | None, key: str, where: str) -> Any:
    if not mapping or key not in mapping:
        raise GateConfigError(f"{where}: missing required key '{key}'")
    return mapping[key]


def _validate_pre_reg(tag: Any, sha: Any) -> tuple[str, str]:
    if not isinstance(tag, str) or not tag.strip():
        raise GateConfigError(
            "pre_registration_tag must be a non-empty string "
            "(use the sentinel 'REPLACE_BEFORE_CALIBRATION' during scaffold)"
        )
    if not isinstance(sha, str) or not sha.strip():
        raise GateConfigError(
            "pre_registration_sha must be a non-empty string "
            "(use the sentinel 'REPLACE_BEFORE_CALIBRATION' during scaffold)"
        )

    tag_is_sentinel = tag == PRE_REG_SENTINEL
    sha_is_sentinel = sha == PRE_REG_SENTINEL

    if tag_is_sentinel or sha_is_sentinel:
        # Sentinel path: warn but do not block offline infra tests.
        warnings.warn(
            "calibration-gate.yaml uses REPLACE_BEFORE_CALIBRATION sentinel "
            "for pre_registration fields; cmd_run will reject this before "
            "the gated calibration pilot runs.",
            stacklevel=3,
        )
        return tag, sha

    # Real-value path: enforce SHA shape.
    if not _SHA_PATTERN.match(sha):
        raise GateConfigError(
            f"pre_registration_sha is malformed (expected 40-char hex commit "
            f"SHA, got {sha!r})"
        )
    if not tag.startswith("gate-prereg-"):
        raise GateConfigError(
            f"pre_registration_tag must follow 'gate-prereg-<bank_id>-<YYYY-MM-DD>' "
            f"(got {tag!r})"
        )
    return tag, sha


def load_config(path: Path | str) -> GateConfig:
    """Parse a calibration-gate YAML file into a GateConfig.

    Raises GateConfigError on any structural or pre-registration violation.
    Emits a UserWarning (not an error) if pre-reg fields hold the
    REPLACE_BEFORE_CALIBRATION sentinel, so scaffold/infra tests can run
    offline without a real pre-reg tag.
    """
    p = Path(path)
    raw = p.read_bytes()
    config_hash = hashlib.sha256(raw).hexdigest()

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise GateConfigError(f"{p}: invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise GateConfigError(f"{p}: expected top-level mapping, got {type(data).__name__}")

    bw_raw = _require(data, "baseline_window", str(p))
    ps_raw = _require(data, "pipeline_sanity", str(p))

    # null_acceptance MUST be present and non-empty. Missing is a spec violation
    # (Scenario "Null-acceptance criterion is pre-registered").
    if "null_acceptance" not in data or not data["null_acceptance"]:
        raise GateConfigError(
            f"{p}: null_acceptance block is missing or empty. The pre-registered "
            "null-acceptance criteria are required; see design D4."
        )
    na_raw = data["null_acceptance"]

    tag_raw = data.get("pre_registration_tag")
    sha_raw = data.get("pre_registration_sha")
    if tag_raw is None:
        raise GateConfigError(f"{p}: missing required key 'pre_registration_tag'")
    if sha_raw is None:
        raise GateConfigError(f"{p}: missing required key 'pre_registration_sha'")

    tag, sha = _validate_pre_reg(tag_raw, sha_raw)

    baseline_window = BaselineWindow(
        min=float(_require(bw_raw, "min", "baseline_window")),
        max=float(_require(bw_raw, "max", "baseline_window")),
        applies_to=str(bw_raw.get("applies_to", "at_least_one_of [base, instruct]")),
    )
    pipeline_sanity = PipelineSanity(
        easy_regression_delta_floor_pp=float(
            _require(ps_raw, "easy_regression_delta_floor_pp", "pipeline_sanity")
        ),
    )
    na_window = _require(na_raw, "baseline_window", "null_acceptance")
    if not (isinstance(na_window, (list, tuple)) and len(na_window) == 2):
        raise GateConfigError(
            "null_acceptance.baseline_window must be a [min, max] pair"
        )
    null_acceptance = NullAcceptance(
        baseline_window=(float(na_window[0]), float(na_window[1])),
        delta_ceiling_pp=float(_require(na_raw, "delta_ceiling_pp", "null_acceptance")),
        min_n_per_condition=int(_require(na_raw, "min_n_per_condition", "null_acceptance")),
    )

    return GateConfig(
        baseline_window=baseline_window,
        pipeline_sanity=pipeline_sanity,
        null_acceptance=null_acceptance,
        pre_registration_tag=tag,
        pre_registration_sha=sha,
        config_hash=config_hash,
    )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _baseline_passes(results: Mapping[str, Any], cfg: GateConfig) -> tuple[bool, str]:
    window = cfg.baseline_window
    by_variant = results.get("no_conditioning_baseline", {}) or {}
    base = by_variant.get("base")
    instruct = by_variant.get("instruct")

    def _in_window(x: float | None) -> bool:
        return x is not None and window.min <= x <= window.max

    any_in = _in_window(base) or _in_window(instruct)
    summary = (
        f"baseline (base={base}, instruct={instruct}) vs window "
        f"[{window.min}, {window.max}]"
    )
    return any_in, summary


def _pipeline_passes(results: Mapping[str, Any], cfg: GateConfig) -> tuple[bool, str]:
    floor = cfg.pipeline_sanity.easy_regression_delta_floor_pp
    delta = float(results.get("easy_regression_delta_pp", 0.0))
    ok = delta >= floor
    summary = f"easy-regression delta {delta} pp vs floor {floor} pp"
    return ok, summary


def _null_accepted(results: Mapping[str, Any], cfg: GateConfig) -> bool:
    na = cfg.null_acceptance
    by_variant = results.get("no_conditioning_baseline", {}) or {}
    lo, hi = na.baseline_window
    in_window = any(
        v is not None and lo <= v <= hi
        for v in (by_variant.get("base"), by_variant.get("instruct"))
    )
    delta_pp = abs(float(results.get("manipulation_delta_pp", 0.0)))
    n = int(results.get("n_per_condition", 0))
    return in_window and delta_pp <= na.delta_ceiling_pp and n >= na.min_n_per_condition


def evaluate(
    calibration_results: Mapping[str, Any],
    config: GateConfig,
) -> Verdict:
    """Evaluate calibration results against the pre-registered gate config.

    Returns a Verdict. Pure function: no I/O, no wall-clock, no mutation of
    inputs.

    Precedence: pipeline regression is reported before baseline recalibration
    because a broken pipeline invalidates any judgment about bank difficulty.
    """
    pipeline_ok, pipeline_summary = _pipeline_passes(calibration_results, config)
    if not pipeline_ok:
        return Verdict(
            status=VerdictStatus.PIPELINE_REGRESSION,
            justification=(
                f"Pipeline regression suspected: {pipeline_summary}. "
                "Sweep blocked pending pipeline diagnosis."
            ),
            config_hash=config.config_hash,
            null_accepted=False,
        )

    baseline_ok, baseline_summary = _baseline_passes(calibration_results, config)
    if not baseline_ok:
        return Verdict(
            status=VerdictStatus.RECALIBRATE,
            justification=(
                f"Baseline out of window: {baseline_summary}. "
                "Bank must be recalibrated; sweep not authorized."
            ),
            config_hash=config.config_hash,
            null_accepted=False,
        )

    null_flag = _null_accepted(calibration_results, config)
    if null_flag:
        justification = (
            f"PASS with null_accepted=True: {baseline_summary}; "
            f"{pipeline_summary}; |manipulation delta| "
            f"{abs(float(calibration_results.get('manipulation_delta_pp', 0.0)))} pp "
            f"within ceiling, n >= {config.null_acceptance.min_n_per_condition}."
        )
    else:
        justification = (
            f"PASS: {baseline_summary}; {pipeline_summary}. Sweep authorized."
        )
    return Verdict(
        status=VerdictStatus.PASS,
        justification=justification,
        config_hash=config.config_hash,
        null_accepted=null_flag,
    )
