"""Exp 2 per-experiment markdown report.

Renders persistence-dynamics analysis results: per-turn accuracy curves,
decay-model fits (exponential vs linear with AIC/BIC), recovery metrics
(time-to-baseline, AUC, asymmetry), and the mandated §10 caveat about
the limits of decay-shape interpretation.

Per persistence-dynamics spec scenario "Decay-shape interpretation is
hedged": when decay-shape comparison is presented, the report MUST
include the §10 caveat that curve shape alone does not distinguish
mood-inertia from context-attention decay.
"""

from __future__ import annotations

from pathlib import Path


SECTION_10_CAVEAT = (
    "**Paper §10 caveat — decay-shape interpretation is limited.** "
    "Transformer attention patterns can themselves be approximately "
    "exponential, which means curve shape alone does NOT distinguish "
    "mood-inertia from context-attention as the underlying mechanism. "
    "AIC/BIC comparison can identify the better-fitting shape, but a "
    "winning exponential fit does not constitute mechanistic evidence "
    "for either account."
)


from src.analysis.reports._format import fmt_value as _format_value  # noqa: E402


def render_exp2_report(analysis: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append(f"# Exp 2 — Persistence / recovery dynamics (model: {analysis['model']})")
    lines.append("")
    lines.append(f"**Verdict:** `{analysis['verdict']}`")
    lines.append(f"**Baseline accuracy:** {_format_value(analysis.get('baseline'))}")
    lines.append(f"**N-values:** {analysis.get('n_values')}")
    if analysis.get("asymmetry_ratio") is not None:
        lines.append(f"**Asymmetry ratio (|neg|/|pos|):** {_format_value(analysis['asymmetry_ratio'])}")
    lines.append("")

    by_cond = analysis.get("by_condition", {})

    lines.append("## Per-condition turn-accuracy curves")
    lines.append("")
    lines.append("| Condition | Curve at N | Recovery metrics |")
    lines.append("|---|---|---|")
    for cond, cell in sorted(by_cond.items()):
        curve = cell.get("turn_accuracies_mean", [])
        curve_str = ", ".join(f"{c:.2f}" for c in curve)
        rm = cell.get("recovery_metrics", {})
        rm_str = ", ".join(
            f"{k}={_format_value(v)}" for k, v in rm.items()
        ) or "—"
        lines.append(f"| {cond} | [{curve_str}] | {rm_str} |")
    lines.append("")

    lines.append("## Decay-model fits (exponential vs linear)")
    lines.append("")
    lines.append("| Condition | Exp amplitude | Exp tau | Exp AIC | Exp BIC | Lin slope | Lin AIC | Lin BIC |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cond, cell in sorted(by_cond.items()):
        df = cell.get("decay_fit", {})
        exp_fit = df.get("exponential", {})
        lin_fit = df.get("linear", {})
        lines.append(
            f"| {cond} | "
            f"{_format_value(exp_fit.get('amplitude'))} | "
            f"{_format_value(exp_fit.get('tau'))} | "
            f"{_format_value(exp_fit.get('aic'))} | "
            f"{_format_value(exp_fit.get('bic'))} | "
            f"{_format_value(lin_fit.get('slope'))} | "
            f"{_format_value(lin_fit.get('aic'))} | "
            f"{_format_value(lin_fit.get('bic'))} |"
        )

    lines.append("")
    lines.append(SECTION_10_CAVEAT)
    lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
