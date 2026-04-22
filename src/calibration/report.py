"""Markdown generator for the committed calibration-report artifact.

Consumes the calibration pilot's results + the gate verdict + the gate
config, emits a deterministic markdown string ready to write to
`artifacts/calibration-<bank_id>-<YYYY-MM-DD>.md`.

Design constraints (design.md D7):
- Byte-identical across re-runs given identical inputs.
- No wall-clock timestamps in the body; the only time reference is
  `report_date` supplied by the caller.
- All four 2x2 transfer cells appear, MISSING for absent cells.
- References GAPS G1 (RLHF-vs-exposure) and G2 (format confound).

Spec: affect-battery-task-difficulty-calibration::task-difficulty-calibration::
"Calibration report schema". Group 11.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.analysis.report import manipulation_check_report
from src.analysis.stats import ManipulationCheckResult
from src.calibration.gate import GateConfig, Verdict, VerdictStatus


# The four 2x2 transfer cells, in the canonical order the report renders.
# Keeping this ordered-and-explicit prevents silent drift in report layout.
TRANSFER_CELL_ORDER: tuple[str, ...] = (
    "same_easy",
    "same_hard",
    "diff_easy",
    "diff_hard",
)

TRANSFER_CELL_LABELS: dict[str, str] = {
    "same_easy": "Same task-type (arithmetic) × Easy",
    "same_hard": "Same task-type (arithmetic) × Hard (PRIMARY)",
    "diff_easy": "Different task-type (recall) × Easy",
    "diff_hard": "Different task-type (reasoning) × Hard",
}


@dataclass(frozen=True)
class TransferCell:
    """One cell of the 2x2 transfer design."""
    cell_key: str
    bank_id: str | None
    accuracy_by_model: dict[str, float]


@dataclass(frozen=True)
class CalibrationPilotData:
    """Full calibration pilot output consumed by the report generator."""
    bank_id: str
    bank_version: int
    manipulation_check_results: list[ManipulationCheckResult]
    transfer_cells: dict[str, TransferCell]
    easy_regression_delta_pp: float | None
    easy_regression_baseline_pp: float | None


# ───────────────────────── markdown building blocks ──────────────────────────


def _header(pilot: CalibrationPilotData, report_date: str, verdict: Verdict) -> str:
    return (
        f"# Calibration Report — `{pilot.bank_id}`\n"
        f"\n"
        f"**Date**: {report_date}  \n"
        f"**Bank version**: {pilot.bank_version}  \n"
        f"**Gate verdict**: **{verdict.status.value}**  \n"
        f"**Verdict justification**: {verdict.justification}  \n"
        f"**Gate config hash**: `{verdict.config_hash}`  \n"
        f"**Null accepted**: {'yes' if verdict.null_accepted else 'no'}  \n"
    )


def _manipulation_check_section(pilot: CalibrationPilotData) -> str:
    """Render the per-model manipulation-check table. Uses
    manipulation_check_report to get a structured view, then emits one row
    per model with the conditions that matter."""
    structured = manipulation_check_report(pilot.manipulation_check_results)
    lines: list[str] = [
        "## Manipulation check (per model)",
        "",
        "| Model | Verdict | NO_CONDITIONING baseline | STRONG_POSITIVE Δpp | STRONG_NEGATIVE Δpp | SELF_CHECK_NEUTRAL Δpp | Rivalry |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sorted(structured.rows, key=lambda r: r.model):
        baseline = f"{row.baseline_accuracy:.3f}" if row.baseline_accuracy is not None else "—"

        def _cell(cond_key: str) -> str:
            cell = row.conditions.get(cond_key)
            if cell is None:
                return "—"
            return f"{cell.delta_vs_baseline_pp:+.1f}"

        rivalry = "⚠︎" if row.self_check_rivals_strong_negative else ""
        lines.append(
            f"| {row.model} | {row.verdict.value} | {baseline} | "
            f"{_cell('strong_positive')} | {_cell('strong_negative')} | "
            f"{_cell('self_check_neutral')} | {rivalry} |"
        )
    lines.append("")
    return "\n".join(lines)


def _transfer_section(pilot: CalibrationPilotData) -> str:
    """Render the 2x2 transfer coverage table. MISSING for absent cells."""
    lines: list[str] = [
        "## Transfer (2×2 coverage)",
        "",
    ]
    for cell_key in TRANSFER_CELL_ORDER:
        label = TRANSFER_CELL_LABELS[cell_key]
        cell = pilot.transfer_cells.get(cell_key)
        lines.append(f"### {cell_key}: {label}")
        lines.append("")
        if cell is None or not cell.accuracy_by_model:
            lines.append(f"**MISSING** — no data for cell `{cell_key}`.")
        else:
            bank_line = (
                f"**Bank**: `{cell.bank_id}`  " if cell.bank_id else ""
            )
            lines.append(bank_line)
            lines.append("")
            lines.append("| Model | Accuracy |")
            lines.append("|---|---|")
            for model, acc in sorted(cell.accuracy_by_model.items()):
                lines.append(f"| {model} | {acc:.3f} |")
        lines.append("")
    return "\n".join(lines)


def _easy_regression_section(pilot: CalibrationPilotData) -> str:
    if pilot.easy_regression_delta_pp is None:
        body = "Regression arm not run."
    else:
        body = (
            f"Observed manipulation-check delta: "
            f"**{pilot.easy_regression_delta_pp:+.1f} pp** on "
            f"`arithmetic_easy_v1` (base model).  \n"
            f"Baseline (NO_CONDITIONING) accuracy: "
            f"{pilot.easy_regression_baseline_pp:.3f}  \n"
        )
    return (
        "## Easy-bank regression arm\n"
        "\n"
        f"{body}"
        "\n"
        "_Pipeline-sanity gate: compare this delta against the 2026-04-20 "
        "pilot's 12.0 pp signal. Regression below the gate's "
        "`easy_regression_delta_floor_pp` triggers PIPELINE_REGRESSION._\n"
    )


def _gate_config_section(gate_config: GateConfig) -> str:
    return (
        "## Gate configuration (from `configs/calibration-gate.yaml`)\n"
        "\n"
        f"- Baseline window: `[{gate_config.baseline_window.min}, "
        f"{gate_config.baseline_window.max}]`, applies_to: "
        f"`{gate_config.baseline_window.applies_to}`\n"
        f"- Pipeline sanity delta floor: "
        f"`{gate_config.pipeline_sanity.easy_regression_delta_floor_pp} pp`\n"
        f"- Null acceptance window: "
        f"`{gate_config.null_acceptance.baseline_window}`, "
        f"delta ceiling: "
        f"`{gate_config.null_acceptance.delta_ceiling_pp} pp`, "
        f"min n: `{gate_config.null_acceptance.min_n_per_condition}`\n"
        f"- Pre-registration tag: `{gate_config.pre_registration_tag}`\n"
        f"- Pre-registration commit SHA: `{gate_config.pre_registration_sha}`\n"
        f"- Config content hash: `{gate_config.config_hash}`\n"
    )


def _confound_status_section() -> str:
    return (
        "## Known confounds (from `GAPS.md`)\n"
        "\n"
        "- **G1 (RLHF-vs-exposure)**: base-vs-instruct split may reflect "
        "post-training-corpus exposure to EmotionPrompt (2023) rather than "
        "RLHF alignment mechanics. Not resolvable without an SFT-only or "
        "pre-2023 post-training variant of the same base model. Report "
        "both hypotheses when interpreting the sweep.\n"
        "- **G2 (chat-vs-completion format confound)**: base model uses "
        "`/v1/completions`, instruct uses `/v1/chat/completions`. Any "
        "base-vs-instruct difference may reflect prompt-format surface "
        "rather than RLHF. Flag as a non-swappable variable in reporting.\n"
    )


def generate_calibration_report(
    pilot: CalibrationPilotData,
    gate_verdict: Verdict,
    gate_config: GateConfig,
    report_date: str,
) -> str:
    """Build the full markdown report. Deterministic: identical inputs =>
    byte-identical output."""
    sections = [
        _header(pilot, report_date, gate_verdict),
        _manipulation_check_section(pilot),
        _transfer_section(pilot),
        _easy_regression_section(pilot),
        _gate_config_section(gate_config),
        _confound_status_section(),
    ]
    return "\n".join(sections) + "\n"
