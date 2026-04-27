"""Tests for the difficulty auto-calibrator.

Spec: affect-battery-task-difficulty-calibration::task-difficulty-calibration.
Tasks 1.1b + 1.1c.
"""

import pytest

from src.calibration.auto_calibrator import (
    AutoCalibrator,
    BracketResult,
    CalibratorConfig,
    OperatorProbeResult,
    SweetSpotResult,
)


# ───────────────────────── mock probe ──────────────────────────


class ScriptedProbe:
    """Records probe calls and returns scripted accuracies keyed by digit_level.

    Used as the probe strategy in calibrator tests so we don't actually
    generate items or call a model.
    """

    def __init__(self, accuracy_by_digit_level: dict[int, float]):
        self.accuracies = dict(accuracy_by_digit_level)
        self.calls: list[tuple[str, int]] = []

    def probe(self, operator: str, digit_level: int, seed: int = 0) -> float:
        self.calls.append((operator, digit_level))
        if digit_level not in self.accuracies:
            raise ValueError(
                f"ScriptedProbe has no accuracy for digit_level={digit_level}; "
                f"known levels: {sorted(self.accuracies)}"
            )
        return self.accuracies[digit_level]


# ───────────────────────── monotone-convergence case ──────────────────────────


class TestBinarySearchConvergence:
    """Binary search converges when accuracy is monotone decreasing in digit_level."""

    def test_finds_sweet_spot_at_exact_level(self):
        # Accuracy strictly decreasing in digit level; sweet spot at 4.
        probe = ScriptedProbe({2: 0.98, 3: 0.92, 4: 0.75, 5: 0.40, 6: 0.15})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 6), max_iter=5,
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert isinstance(result, SweetSpotResult)
        assert result.operator == "add"
        assert result.digit_level == 4
        assert 0.60 <= result.accuracy <= 0.85

    def test_finds_sweet_spot_at_lower_bound(self):
        # Even the easiest level (2) lands in window.
        probe = ScriptedProbe({2: 0.70, 3: 0.40, 4: 0.15})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 4),
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert isinstance(result, SweetSpotResult)
        assert result.digit_level == 2

    def test_finds_sweet_spot_at_upper_bound(self):
        probe = ScriptedProbe({2: 0.99, 3: 0.95, 4: 0.90, 5: 0.70})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 5),
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert result.digit_level == 5


# ───────────────────────── no-integer-solution case ──────────────────────────


class TestBracketReporting:
    """When no integer digit_level lands in the target window, report the
    nearest bracket on each side so the user can pick."""

    def test_reports_bracket_when_gap_is_too_wide(self):
        # Digit 3 is 0.92 (too easy); digit 4 is 0.40 (too hard). No integer
        # level hits [0.60, 0.85]. Calibrator must report both sides.
        probe = ScriptedProbe({2: 0.99, 3: 0.92, 4: 0.40, 5: 0.15})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 5),
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert isinstance(result, BracketResult)
        # Closest above-target-max: digit_level=3 at 0.92.
        assert result.easier_side is not None
        assert result.easier_side.digit_level == 3
        assert result.easier_side.accuracy == 0.92
        # Closest below-target-min: digit_level=4 at 0.40.
        assert result.harder_side is not None
        assert result.harder_side.digit_level == 4
        assert result.harder_side.accuracy == 0.40

    def test_reports_none_side_when_range_is_fully_below_window(self):
        # Every level is too hard — harder_side only.
        probe = ScriptedProbe({3: 0.40, 4: 0.20, 5: 0.10})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(3, 5),
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert isinstance(result, BracketResult)
        assert result.easier_side is None
        assert result.harder_side is not None
        assert result.harder_side.digit_level == 3  # closest to sweet spot
        assert result.harder_side.accuracy == 0.40

    def test_reports_none_side_when_range_is_fully_above_window(self):
        # Every level is too easy — easier_side only.
        probe = ScriptedProbe({2: 0.99, 3: 0.95, 4: 0.90})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 4),
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert isinstance(result, BracketResult)
        assert result.easier_side is not None
        assert result.easier_side.digit_level == 4
        assert result.harder_side is None


# ───────────────────────── caching ──────────────────────────


class TestProbeCaching:
    """Identical (operator, digit_level, seed) tuple hits cache on second call."""

    def test_cache_hit_on_repeat_calibration(self):
        probe = ScriptedProbe({2: 0.98, 3: 0.92, 4: 0.75, 5: 0.40})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 5),
        )
        calibrator = AutoCalibrator(cfg, probe)
        calibrator.calibrate_operator("add")
        calls_before = len(probe.calls)
        calibrator.calibrate_operator("add")  # repeat — should be cached
        assert len(probe.calls) == calls_before, (
            f"Expected no new probe calls on repeat, "
            f"got {len(probe.calls) - calls_before} new."
        )

    def test_cache_miss_on_different_operator(self):
        probe = ScriptedProbe({2: 0.98, 3: 0.92, 4: 0.75, 5: 0.40})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 5),
        )
        calibrator = AutoCalibrator(cfg, probe)
        calibrator.calibrate_operator("add")
        calls_before = len(probe.calls)
        calibrator.calibrate_operator("sub")  # same digit_levels, diff operator
        assert len(probe.calls) > calls_before


# ───────────────────────── max-iteration cap ──────────────────────────


class TestMaxIterCap:
    """Pathological accuracy curves (non-monotone) must not run forever."""

    def test_max_iter_bounds_probe_count(self):
        # Non-monotone: accuracy wiggles. Binary search will never converge.
        probe = ScriptedProbe({2: 0.40, 3: 0.90, 4: 0.50, 5: 0.80, 6: 0.30, 7: 0.95})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 7), max_iter=4,
        )
        calibrator = AutoCalibrator(cfg, probe)
        calibrator.calibrate_operator("add")
        # Binary search must have made at most max_iter probes.
        assert len(probe.calls) <= cfg.max_iter, (
            f"Binary search made {len(probe.calls)} probes, "
            f"exceeded max_iter={cfg.max_iter}"
        )


# ───────────────────────── OperatorProbeResult shape ──────────────────────────


class TestResultShape:
    """Verify the result datatypes carry the right fields for downstream
    calibration-bank generation."""

    def test_sweet_spot_result_carries_full_probe_history(self):
        probe = ScriptedProbe({2: 0.98, 3: 0.92, 4: 0.75, 5: 0.40})
        cfg = CalibratorConfig(
            target_min=0.60, target_max=0.85, digit_range=(2, 5),
        )
        calibrator = AutoCalibrator(cfg, probe)
        result = calibrator.calibrate_operator("add")
        assert isinstance(result, SweetSpotResult)
        # History MUST include every probe the search made, in order, so the
        # calibration report can show the full search trail.
        assert len(result.probe_history) >= 1
        for item in result.probe_history:
            assert isinstance(item, OperatorProbeResult)
            assert item.operator == "add"
            assert 2 <= item.digit_level <= 5
            assert 0.0 <= item.accuracy <= 1.0
