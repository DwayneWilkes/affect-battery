"""Exp 3a per-experiment markdown report.

Renders the structured analysis dict produced by
`src.analysis.exp3a.analyze_exp3a` as a markdown report.

Per scoring-pipeline spec "Per-experiment report views": the Exp 3a
view includes the quadratic fit (β₂ + one-sided p-value), the
quadratic-vs-linear AIC/BIC comparison, and the Krippendorff pilot
outcome (when supplied). The β₂ < 0 directional test and the
quadratic-vs-linear shape comparison are reported as distinct results
per the "β₂ and fit comparison are separate tests" scenario.
"""

from __future__ import annotations

from pathlib import Path

from src.analysis.reports._format import fmt_p, fmt_value


def render_exp3a_report(analysis: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    lines: list[str] = []
    lines.append(
        f"# Exp 3a — Inverted-U intensity (model: {analysis.get('model', '?')})"
    )
    lines.append("")
    lines.append(f"**n samples:** {analysis.get('n', '—')}")
    lines.append(f"**Levels:** {analysis.get('levels', [])}")
    lines.append("")

    lines.append("## Quadratic fit: accuracy ~ b0 + b1*L + b2*L^2")
    lines.append("")
    lines.append("| Coefficient | Value |")
    lines.append("|---|---|")
    lines.append(f"| beta_0 (intercept) | {fmt_value(analysis.get('beta_0'))} |")
    lines.append(f"| beta_1 (linear) | {fmt_value(analysis.get('beta_1'))} |")
    lines.append(f"| beta_2 (quadratic) | {fmt_value(analysis.get('beta_2'))} |")
    lines.append(f"| beta_2 SE | {fmt_value(analysis.get('beta_2_se'))} |")
    lines.append(
        f"| beta_2 one-sided p (H_a: beta_2 < 0) | "
        f"{fmt_p(analysis.get('beta_2_p_one_sided'))} |"
    )
    lines.append("")
    lines.append(
        "An inverted-U arousal-performance relationship is the H3a-confirming "
        "signature: significant `beta_2 < 0` after primary-family Holm "
        "correction."
    )
    lines.append("")

    lines.append("## Shape comparison: quadratic vs linear")
    lines.append("")
    lines.append("| Model | RSS | AIC | BIC |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Quadratic | {fmt_value(analysis.get('quadratic_rss'))} | "
        f"{fmt_value(analysis.get('quadratic_aic'))} | "
        f"{fmt_value(analysis.get('quadratic_bic'))} |"
    )
    lines.append(
        f"| Linear | {fmt_value(analysis.get('linear_rss'))} | "
        f"{fmt_value(analysis.get('linear_aic'))} | "
        f"{fmt_value(analysis.get('linear_bic'))} |"
    )
    lines.append("")
    lines.append(
        "Lower AIC/BIC indicates better fit. Per paper §3.4.1 a quadratic "
        "model with negative leading coefficient should fit significantly "
        "better than a linear model under H3a; this comparison is reported "
        "separately from the `beta_2 < 0` directional test."
    )
    lines.append("")

    pilot = analysis.get("intensity_pilot")
    if pilot is not None:
        lines.append("## Intensity-axis pilot outcome (Krippendorff α)")
        lines.append("")
        lines.append(f"- **Decision:** `{pilot.get('decision', '—')}`")
        lines.append(f"- **n raters:** {pilot.get('n_raters', '—')}")
        lines.append(f"- **n items:** {pilot.get('n_items', '—')}")
        lines.append(f"- **alpha overall:** {fmt_value(pilot.get('alpha_overall'))}")
        if pilot.get("alpha_pairwise"):
            lines.append("- **alpha pairwise:**")
            for pair, alpha in sorted(pilot["alpha_pairwise"].items()):
                lines.append(f"    - {pair}: {fmt_value(alpha)}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
