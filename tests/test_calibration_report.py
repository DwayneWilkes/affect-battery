"""Tests for the calibration report markdown generator.

Spec: affect-battery-task-difficulty-calibration::task-difficulty-calibration::
"Calibration report schema".

Group 11: generate a markdown report for the committed calibration artifact
(`artifacts/calibration-<bank_id>-<date>.md`). The report consumes:
    - calibration pilot data (manipulation-check results + 2x2 transfer cells
      easy-bank regression delta)
    - gate verdict (from src.calibration.gate)
    - gate config

and produces a deterministic, diff-able markdown string that:
    - names the bank + date in the header
    - shows the gate verdict prominently
    - renders per-model manipulation-check rows (reusing
      src.analysis.report.manipulation_check_report under the hood)
    - covers all four 2x2 transfer cells, emitting `MISSING` for absent cells
    - lists easy-bank regression delta with pipeline-sanity context
    - references GAPS G1 (RLHF-vs-exposure) and G2 (chat-vs-completion
      format) confound status
    - is byte-identical across re-runs given identical inputs
"""

import pytest

from src.analysis.stats import manipulation_check
from src.calibration.gate import (
    BaselineWindow,
    GateConfig,
    NullAcceptance,
    PipelineSanity,
    Verdict,
    VerdictStatus,
)
from src.calibration.report import (
    CalibrationPilotData,
    TransferCell,
    generate_calibration_report,
)
from src.conditioning.prompts import Condition


def _mc_result(model, pos, baseline, neg):
    data = {
        Condition.STRONG_POSITIVE.value: [pos] * 10,
        Condition.NO_CONDITIONING.value: [baseline] * 10,
        Condition.STRONG_NEGATIVE.value: [neg] * 10,
    }
    return manipulation_check(data, model=model)


def _minimal_pilot_data(bank_id="arithmetic_hard_v1") -> CalibrationPilotData:
    return CalibrationPilotData(
        bank_id=bank_id,
        bank_version=1,
        manipulation_check_results=[
            _mc_result("Qwen2.5-7B", 0.90, 0.75, 0.55),
            _mc_result("Qwen2.5-7B-Instruct", 0.82, 0.80, 0.76),
        ],
        transfer_cells={
            "same_easy": TransferCell(
                cell_key="same_easy",
                bank_id="transfer_easy_v1",
                accuracy_by_model={"Qwen2.5-7B": 1.0, "Qwen2.5-7B-Instruct": 1.0},
            ),
            "same_hard": TransferCell(
                cell_key="same_hard",
                bank_id="transfer_hard_arithmetic_v1",
                accuracy_by_model={"Qwen2.5-7B": 0.72, "Qwen2.5-7B-Instruct": 0.78},
            ),
            # diff_easy and diff_hard deliberately missing to exercise MISSING path.
        },
        easy_regression_delta_pp=14.0,
        easy_regression_baseline_pp=0.95,
    )


def _minimal_gate_verdict() -> Verdict:
    return Verdict(
        status=VerdictStatus.PASS,
        justification="baseline in window; regression > floor",
        config_hash="abc123def456",
    )


def _minimal_gate_config() -> GateConfig:
    """Build a minimal GateConfig for rendering via the dataclass directly
    (hermetic; no disk load)."""
    return GateConfig(
        baseline_window=BaselineWindow(min=0.60, max=0.85),
        pipeline_sanity=PipelineSanity(easy_regression_delta_floor_pp=8.0),
        null_acceptance=NullAcceptance(
            baseline_window=(0.60, 0.85),
            delta_ceiling_pp=5.0,
            min_n_per_condition=50,
        ),
        pre_registration_tag="gate-prereg-arithmetic_hard_v1-2026-04-22",
        pre_registration_sha="a" * 40,
        config_hash="abc123def456",
    )


class TestReportStructure:
    def test_report_is_string(self):
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        assert isinstance(md, str)
        assert len(md) > 100  # meaningful content

    def test_header_names_bank_and_date(self):
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(bank_id="arithmetic_hard_v1"),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        assert "arithmetic_hard_v1" in md
        assert "2026-04-22" in md

    def test_gate_verdict_is_prominent(self):
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        # The PASS verdict should appear in the first 500 characters so a
        # reviewer skimming the top of the file sees it.
        assert "PASS" in md[:500]

    def test_per_model_manipulation_check_rows(self):
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        assert "Qwen2.5-7B" in md
        assert "Qwen2.5-7B-Instruct" in md


class TestTransferTwoByTwoCoverage:
    def test_all_four_cells_appear(self):
        """All four 2x2 cell keys (same_easy, same_hard, diff_easy, diff_hard)
        MUST appear in the report, even if their data is missing."""
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        for cell in ("same_easy", "same_hard", "diff_easy", "diff_hard"):
            assert cell in md, f"cell '{cell}' missing from report"

    def test_missing_cells_flagged(self):
        """Diff_easy and diff_hard aren't in _minimal_pilot_data —
        the report must flag those cells MISSING."""
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        assert "MISSING" in md


class TestConfoundStatus:
    def test_gaps_references(self):
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        assert "G1" in md  # RLHF-vs-exposure confound
        assert "G2" in md  # chat-vs-completion format confound


class TestDeterminism:
    def test_byte_identical_across_reruns(self):
        pilot = _minimal_pilot_data()
        verdict = _minimal_gate_verdict()
        cfg = _minimal_gate_config()
        md1 = generate_calibration_report(pilot, verdict, cfg, report_date="2026-04-22")
        md2 = generate_calibration_report(pilot, verdict, cfg, report_date="2026-04-22")
        assert md1 == md2

    def test_no_wall_clock_in_body(self):
        """The body MUST NOT include wall-clock timestamps that shift
        run-to-run. The report_date field is the only time reference."""
        md = generate_calibration_report(
            pilot=_minimal_pilot_data(),
            gate_verdict=_minimal_gate_verdict(),
            gate_config=_minimal_gate_config(),
            report_date="2026-04-22",
        )
        # No HH:MM:SS patterns, no UTC timestamps, no today's wall-clock year/month
        # being injected unrelated to report_date.
        import re
        # HH:MM:SS clock pattern (excluding day-of-week-agnostic timestamps like
        # ISO-formatted date "2026-04-22" which IS the report_date).
        hhmmss = re.findall(r"\d{2}:\d{2}:\d{2}", md)
        assert hhmmss == [], f"unexpected wall-clock times in body: {hhmmss}"
