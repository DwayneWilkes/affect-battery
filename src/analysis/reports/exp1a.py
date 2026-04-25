"""Exp 1a per-experiment markdown report.

Renders the structured dict produced by `src.analysis.exp1a.analyze_exp1a_corpus`
as a markdown table with per-condition Cohen's d, raw p-values, and
Holm-Bonferroni corrected p-values. Cites the Holm reference inline so a
reviewer reading only the report knows which correction policy was applied.
"""

from __future__ import annotations

from pathlib import Path


HOLM_CITATION = (
    "Holm, S. (1979). 'A simple sequentially rejective multiple test "
    "procedure.' *Scandinavian Journal of Statistics*, 6(2), 65-70."
)


def _format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _format_d(d: float) -> str:
    if d == float("inf"):
        return "+inf"
    if d == float("-inf"):
        return "-inf"
    return f"{d:+.2f}"


def render_exp1a_report(analysis: dict, output_path: Path) -> Path:
    """Render an Exp 1a analysis dict as markdown and write to output_path."""
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append(f"# Exp 1a — H1 Within-Session Transfer (model: {analysis['model']})")
    lines.append("")
    lines.append(f"**Verdict:** `{analysis['verdict']}`")
    lines.append("")

    if analysis["verdict"] == "unavailable_no_baseline":
        lines.append(
            "> NO_CONDITIONING baseline absent; per-condition deltas cannot "
            "be computed. UNAVAILABLE is a measurement gap, not a null finding."
        )
        output_path.write_text("\n".join(lines) + "\n")
        return output_path

    per_condition = analysis["per_condition_vs_baseline"]

    lines.append("## Per-condition effect sizes vs no_conditioning baseline")
    lines.append("")
    lines.append(
        "| Condition | n | Mean acc. | Baseline acc. | Cohen's d | p (raw) | p (Holm) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    for cond, cells in sorted(per_condition.items()):
        lines.append(
            f"| {cond} | {cells['n_runs']} | "
            f"{cells['mean_accuracy']:.3f} | {cells['baseline_mean']:.3f} | "
            f"{_format_d(cells['cohens_d'])} | "
            f"{_format_p(cells['p_raw'])} | "
            f"{_format_p(cells['p_holm_corrected'])} |"
        )

    lines.append("")
    lines.append("## Multiple-comparisons correction")
    lines.append("")
    lines.append(
        "Holm-Bonferroni step-down correction applied across non-baseline "
        "conditions within Exp 1a (controls family-wise error rate)."
    )
    lines.append("")
    lines.append(HOLM_CITATION)
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
