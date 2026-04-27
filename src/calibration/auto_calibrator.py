"""Binary-search auto-calibrator for per-operator difficulty.

For each arithmetic operator, probe at varying `digit_level` and return
either:
    - a SweetSpotResult: a single integer digit_level where the model's
      observed accuracy lands in [target_min, target_max]; or
    - a BracketResult: the nearest probed points on each side of the
      window, when no integer digit_level hits the range.

The caller (scripts/auto_calibrate_arithmetic.py) turns the results into
per-operator GenSpec objects for the bank generator.

Spec: affect-battery-task-difficulty-calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


# ───────────────────────── types ──────────────────────────


@dataclass(frozen=True)
class OperatorProbeResult:
    """One probe: (operator, digit_level) -> accuracy."""
    operator: str
    digit_level: int
    accuracy: float


@dataclass(frozen=True)
class SweetSpotResult:
    """Calibration succeeded: a single digit_level lands in the target window."""
    operator: str
    digit_level: int
    accuracy: float
    probe_history: tuple[OperatorProbeResult, ...]


@dataclass(frozen=True)
class BracketResult:
    """Calibration couldn't find a sweet-spot integer; reports nearest
    probes on each side of the window. Either side may be None if the
    whole probed range is on one side of the window."""
    operator: str
    easier_side: OperatorProbeResult | None  # closest probe above target_max
    harder_side: OperatorProbeResult | None  # closest probe below target_min
    probe_history: tuple[OperatorProbeResult, ...]


CalibrationResult = SweetSpotResult | BracketResult


class Probe(Protocol):
    """Anything that can measure accuracy at a (operator, digit_level) pair.

    The scripted test probe, the real generator-plus-model probe, and any
    future cached wrapper all fit this shape.
    """

    def probe(self, operator: str, digit_level: int, seed: int = 0) -> float:
        ...


@dataclass
class CalibratorConfig:
    target_min: float = 0.60
    target_max: float = 0.85
    digit_range: tuple[int, int] = (2, 8)
    max_iter: int = 6
    probe_seed: int = 0


# ───────────────────────── calibrator ──────────────────────────


class AutoCalibrator:
    """Binary-search for a per-operator digit_level where the probe's
    observed accuracy lands in [target_min, target_max].

    Caching: results are memoized by (operator, digit_level, probe_seed)
    so repeated calibrations don't re-query the model. The cache lives
    for the lifetime of the calibrator instance.
    """

    def __init__(self, config: CalibratorConfig, probe: Probe):
        self._config = config
        self._probe = probe
        self._cache: dict[tuple[str, int, int], float] = {}

    def calibrate_operator(self, operator: str) -> CalibrationResult:
        history: list[OperatorProbeResult] = []
        lo, hi = self._config.digit_range
        iters = 0

        # Track the tightest brackets seen so far on each side of the window.
        easier_side: OperatorProbeResult | None = None  # above target_max
        harder_side: OperatorProbeResult | None = None  # below target_min

        while lo <= hi and iters < self._config.max_iter:
            iters += 1
            mid = (lo + hi) // 2
            acc = self._probe_cached(operator, mid)
            probe_result = OperatorProbeResult(
                operator=operator, digit_level=mid, accuracy=acc,
            )
            history.append(probe_result)

            if acc > self._config.target_max:
                # Too easy at mid; try harder (deeper digit levels).
                if easier_side is None or probe_result.digit_level > easier_side.digit_level:
                    easier_side = probe_result
                lo = mid + 1
            elif acc < self._config.target_min:
                # Too hard at mid; try easier (shallower digit levels).
                if harder_side is None or probe_result.digit_level < harder_side.digit_level:
                    harder_side = probe_result
                hi = mid - 1
            else:
                return SweetSpotResult(
                    operator=operator,
                    digit_level=mid,
                    accuracy=acc,
                    probe_history=tuple(history),
                )

        return BracketResult(
            operator=operator,
            easier_side=easier_side,
            harder_side=harder_side,
            probe_history=tuple(history),
        )

    def _probe_cached(self, operator: str, digit_level: int) -> float:
        key = (operator, digit_level, self._config.probe_seed)
        if key not in self._cache:
            self._cache[key] = self._probe.probe(
                operator=operator,
                digit_level=digit_level,
                seed=self._config.probe_seed,
            )
        return self._cache[key]
