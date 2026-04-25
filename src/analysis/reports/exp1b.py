"""Exp 1b per-experiment markdown report.

Renders the structured dict produced by `src.analysis.exp1b.analyze_exp1b`
as a markdown table with the three-way comparison (session 1, session 2,
no_conditioning baseline). When session-2 effect is equivalent to zero
within +/- 0.10 (TOST), the report frames the outcome as
"context-attention mechanism confirmed (paper §3.2.2 expected)" rather
than "H1b failed" — the spec's null-confirms-context-attention framing.
"""

from __future__ import annotations

from pathlib import Path


CONTEXT_ATTENTION_BLURB = (
    "**Context-attention mechanism confirmed (paper §3.2.2 expected).** "
    "Session-2 effect is statistically equivalent to zero within +/-0.10 "
    "(TOST). This is the predicted outcome under the context-attention "
    "account: conditioning effects do not persist across sessions, which "
    "is a feature of the mechanism rather than a failure of the test."
)

PERSISTENCE_BLURB = (
    "**Session-2 effect did NOT equivalence-pass at +/-0.10 (TOST).** "
    "The conditioning signal appears to persist across sessions; this is "
    "more surprising than null and warrants closer inspection of "
    "session-2 message history for context carryover."
)


def _format_value(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if v == float("inf"):
            return "+inf"
        if v == float("-inf"):
            return "-inf"
        return f"{v:+.2f}" if abs(v) >= 0.005 else f"{v:.4f}"
    return str(v)


def render_exp1b_report(analysis: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append(f"# Exp 1b — Cross-session falsification (model: {analysis['model']})")
    lines.append("")
    lines.append(f"**Verdict:** `{analysis['verdict']}`")
    lines.append("")

    if analysis["verdict"] == "unavailable_no_baseline":
        lines.append(
            "> no_conditioning baseline absent; three-way comparison cannot "
            "be computed. UNAVAILABLE is a measurement gap, not a null finding."
        )
        output_path.write_text("\n".join(lines) + "\n")
        return output_path

    comparison = analysis["three_way_comparison"]

    # Per-condition framing block: any condition with session_2_equivalent=True
    # gets the context-attention blurb; otherwise the persistence blurb.
    any_equivalent = any(
        cell.get("session_2_equivalent") is True for cell in comparison.values()
    )
    any_non_equivalent = any(
        cell.get("session_2_equivalent") is False for cell in comparison.values()
    )

    lines.append("## Outcome framing")
    lines.append("")
    if any_equivalent:
        lines.append(CONTEXT_ATTENTION_BLURB)
        lines.append("")
    if any_non_equivalent:
        lines.append(PERSISTENCE_BLURB)
        lines.append("")

    lines.append("## Three-way comparison: session 1 vs session 2 vs no_conditioning")
    lines.append("")
    lines.append(
        "| Condition | s1 d | s2 d | s1 mean | s2 mean | baseline | "
        "s2 directional p | s2 TOST p | equivalent? |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for cond, cells in sorted(comparison.items()):
        lines.append(
            f"| {cond} | "
            f"{_format_value(cells.get('session_1_effect_size'))} | "
            f"{_format_value(cells.get('session_2_effect_size'))} | "
            f"{_format_value(cells.get('session_1_mean_accuracy'))} | "
            f"{_format_value(cells.get('session_2_mean_accuracy'))} | "
            f"{_format_value(cells.get('no_conditioning_baseline'))} | "
            f"{_format_value(cells.get('session_2_directional_p'))} | "
            f"{_format_value(cells.get('session_2_tost_p'))} | "
            f"{cells.get('session_2_equivalent', '—')} |"
        )

    lines.append("")
    output_path.write_text("\n".join(lines) + "\n")
    return output_path
