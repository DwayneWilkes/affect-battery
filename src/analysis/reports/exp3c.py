"""Exp 3c conservative-shift markdown report.

Per conservative-shift-measurement spec "Every 3c report cites §10":
the report MUST include the §10 caveat warning that observed
conservative-shift output may reflect mood-as-information heuristic
rather than genuine epistemic recalibration.
"""

from __future__ import annotations

from pathlib import Path


SECTION_10_MOOD_AS_INFO_CAVEAT = (
    "**Paper §10 caveat — conservative-shift vs mood-as-information.** "
    "Observed shifts in hedging rate, stated confidence, response length, "
    "or refusal frequency under negative-affect conditioning MAY reflect "
    "the mood-as-information heuristic (Schwarz & Clore 1983): the model "
    "uses its conditioned 'mood' as informational input about the world, "
    "rather than performing genuine epistemic recalibration. Curve shape "
    "alone cannot distinguish these accounts; H3c interpretation should "
    "be hedged accordingly."
)


def _format_value(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_exp3c_report(analysis: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append(f"# Exp 3c — Conservative shift (model: {analysis['model']})")
    lines.append("")
    lines.append(f"**Verdict:** `{analysis['verdict']}`")
    lines.append("")

    lines.append("## Per-(condition, difficulty) metrics")
    lines.append("")
    lines.append(
        "| Condition | Difficulty | n items | hedging /100w | refusal rate | mean length |"
    )
    lines.append("|---|---|---|---|---|---|")
    for (cond, diff), cell in sorted(analysis.get("by_condition_difficulty", {}).items()):
        lines.append(
            f"| {cond} | {diff} | {cell.get('n_items', '—')} | "
            f"{_format_value(cell.get('hedging_rate_per_100w'))} | "
            f"{_format_value(cell.get('refusal_rate'))} | "
            f"{_format_value(cell.get('mean_response_length'))} |"
        )

    lines.append("")
    lines.append(SECTION_10_MOOD_AS_INFO_CAVEAT)
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
