"""H4 joint-outcome 2x2 report.

Per asymmetry-contrast spec "Honest reporting of inconclusive outcomes" +
"Joint-outcome table is always present": render a 2x2 enumerating
{ratio_base > 1 vs <= 1} x {asymmetry_delta_ratio > 1 vs <= 1}. The
table is always present even when both per-model verdicts come back as
"inconclusive" — the spec specifically requires honest reporting of
inconclusive outcomes, not their suppression.
"""

from __future__ import annotations

from pathlib import Path


def _format_value(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_h4_report(result: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    family = result.get("model_family", "?")
    base = result.get("base_model", "base")
    instruct = result.get("instruct_model", "instruct")
    r_base = result.get("ratio_base")
    r_inst = result.get("ratio_instruct")
    delta = result.get("asymmetry_delta_ratio")
    test_a = result.get("test_a_primary_delta_ratio_gt_1")
    test_b = result.get("test_b_secondary_diff_instruct_gt_diff_base")
    verdicts = result.get("per_model_verdicts", {})

    lines: list[str] = []
    lines.append(f"# H4 — Asymmetry contrast (model family: {family})")
    lines.append("")
    lines.append("## Per-model aggregates")
    lines.append("")
    lines.append("| Model | ratio_geomean | verdict |")
    lines.append("|---|---|---|")
    lines.append(f"| {base} | {_format_value(r_base)} | {verdicts.get(base, '—')} |")
    lines.append(f"| {instruct} | {_format_value(r_inst)} | {verdicts.get(instruct, '—')} |")
    lines.append("")
    lines.append("## Cross-model contrast")
    lines.append("")
    lines.append(f"**asymmetry_delta_ratio** = ratio_instruct / ratio_base = {_format_value(delta)}")
    lines.append("")
    lines.append("Pre-registered tests (both run; reported uncorrected within family per design D8):")
    lines.append("")
    lines.append(f"- **test_a (primary):** delta_ratio > 1 → {_format_value(test_a)}")
    lines.append(f"- **test_b (secondary):** difference_instruct > difference_base → {_format_value(test_b)}")
    lines.append("")
    lines.append("## Joint-outcome 2x2 table")
    lines.append("")
    base_gt_1 = r_base is not None and r_base > 1.0
    delta_gt_1 = delta is not None and delta > 1.0

    lines.append("| | delta_ratio > 1 | delta_ratio <= 1 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| ratio_base > 1 | {('YES — observed' if (base_gt_1 and delta_gt_1) else '—')} "
        f"| {('YES — observed' if (base_gt_1 and not delta_gt_1) else '—')} |"
    )
    lines.append(
        f"| ratio_base <= 1 | {('YES — observed' if (not base_gt_1 and delta_gt_1) else '—')} "
        f"| {('YES — observed' if (not base_gt_1 and not delta_gt_1) else '—')} |"
    )
    lines.append("")

    # Honest-reporting blurb when both verdicts are inconclusive
    if all(v == "inconclusive" for v in verdicts.values()) and verdicts:
        lines.append(
            "> **Honest reporting note:** both per-model verdicts came back as "
            "`inconclusive`. The 2x2 table above still indicates which cell the "
            "observed point estimates fall into, but readers should NOT treat "
            "the marked cell as a confirmed outcome — the test simply could "
            "not distinguish it from neighboring cells."
        )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n")
    return output_path
