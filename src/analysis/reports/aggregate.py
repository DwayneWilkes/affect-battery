"""Aggregate cross-experiment report.

Per design.md:D11 Phase 9: top-level
results/AGGREGATE_REPORT.md provides one entry point summarizing all
six experiments + the H4 cross-experiment contrast + primary-family-
wise correction status. Each experiment section is a brief landing
pad; readers click through to the per-experiment reports for detail.
"""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_SECTIONS = [
    ("exp1a", "Exp 1a — Within-session transfer (H1)"),
    ("exp1b", "Exp 1b — Cross-session falsification (H1b)"),
    ("exp2",  "Exp 2 — Persistence / recovery dynamics (H2)"),
    ("exp3a", "Exp 3a — Inverted-U intensity (H3a)"),
    ("exp3b", "Exp 3b — Cognitive scope (semantic diversity)"),
    ("exp3c", "Exp 3c — Conservative shift (hedging / refusal)"),
]


from src.analysis.reports._format import fmt_value as _format_value  # noqa: E402


def render_aggregate(all_results: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append("# Affect Battery — Aggregate Cross-Experiment Report")
    lines.append("")
    lines.append(
        "Top-level summary of all six paper-aligned experiments + H4 "
        "asymmetry contrast. See per-experiment reports under "
        "`results/<expN>_report.md` for full tables, analyses, and §10 "
        "caveats."
    )
    lines.append("")

    for key, title in EXPERIMENT_SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        cell = all_results.get(key)
        if cell is None:
            lines.append("> No data: experiment not run or results pending.")
        else:
            lines.append(f"- Model(s): {cell.get('model', cell.get('models', '—'))}")
            lines.append(f"- Verdict: `{cell.get('verdict', '—')}`")
            lines.append(f"- Per-experiment report: `results/{key}_report.md`")
        lines.append("")

    # H4 cross-experiment section
    lines.append("## H4 — Cross-experiment asymmetry contrast")
    lines.append("")
    h4 = all_results.get("h4")
    if h4 is None:
        lines.append("> H4 contrast not computed.")
    else:
        lines.append(f"- Model family: {h4.get('model_family', '—')}")
        lines.append(
            f"- ratio_base: {_format_value(h4.get('ratio_base'))} | "
            f"ratio_instruct: {_format_value(h4.get('ratio_instruct'))} | "
            f"asymmetry_delta_ratio: {_format_value(h4.get('asymmetry_delta_ratio'))}"
        )
        lines.append("- Full report: `results/h4_report.md`")
    lines.append("")

    # Primary-family-wise corrections summary
    lines.append("## Primary family-wise corrections (Holm-Bonferroni)")
    lines.append("")
    corrections = all_results.get("primary_family_corrections", {})
    if not corrections:
        lines.append("> No corrected p-values supplied.")
    else:
        lines.append("| Hypothesis | raw p | Holm-corrected p | family |")
        lines.append("|---|---|---|---|")
        for h, cell in sorted(corrections.items()):
            lines.append(
                f"| {h} | {_format_value(cell.get('raw'))} | "
                f"{_format_value(cell.get('corrected'))} | "
                f"{cell.get('family', '—')} |"
            )
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
