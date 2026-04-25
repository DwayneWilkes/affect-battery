"""Exp 3b cognitive-scope markdown report.

Per cognitive-scope-measurement spec "Every 3b report cites §10": the
report MUST include a §10 caveat that semantic-diversity is a proxy,
not a direct measurement of cognitive scope.
"""

from __future__ import annotations

from pathlib import Path


SECTION_10_PROXY_CAVEAT = (
    "**Paper §10 caveat — semantic-diversity is a proxy, not direct.** "
    "Embedding variance and unique n-gram ratio measure surface-level "
    "diversity in generated text. They do NOT directly measure cognitive "
    "scope, ideational fluency, or any internal cognitive variable. "
    "The proxy is well-correlated with cognitive-scope outcomes in human "
    "subjects but the mapping is not 1:1; treat differences in this "
    "metric as a hypothesis-generating signal, not as direct evidence "
    "of altered cognitive scope."
)


from src.analysis.reports._format import fmt_value as _format_value  # noqa: E402


def render_exp3b_report(analysis: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append(f"# Exp 3b — Cognitive scope (model: {analysis['model']})")
    lines.append("")
    lines.append(f"**Verdict:** `{analysis['verdict']}`")
    lines.append("")

    lines.append("## Per-condition diversity metrics")
    lines.append("")
    lines.append("| Condition | n generations | embedding variance | n-gram ratio |")
    lines.append("|---|---|---|---|")
    for cond, cell in sorted(analysis.get("by_condition", {}).items()):
        lines.append(
            f"| {cond} | {cell.get('n_generations', '—')} | "
            f"{_format_value(cell.get('embedding_variance'))} | "
            f"{_format_value(cell.get('ngram_ratio'))} |"
        )

    lines.append("")
    lines.append(SECTION_10_PROXY_CAVEAT)
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
