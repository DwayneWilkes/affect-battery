"""Family-wise correction wiring across all experiments.

Per the power-analysis spec "Family-wise error correction": group
p-values by `family_membership` (e.g. "primary", "exploratory"),
apply Holm-Bonferroni within "primary" and Benjamini-Hochberg within
"exploratory", return a per-hypothesis dict with raw + corrected
p-values. The math primitives live in src.analysis_corrections.
"""

from __future__ import annotations

from src.analysis_corrections import apply_bh_correction, apply_holm_correction


# Method per family (per scoring-pipeline spec "Multiple-comparisons
# correction policy"). Add new families here when they're declared.
METHOD_BY_FAMILY = {
    "primary": "holm",
    "exploratory": "bh",
}


def apply_family_corrections(
    p_values_by_hypothesis: dict[str, float],
    family_membership: dict[str, str],
) -> dict[str, dict]:
    """Apply per-family multiple-comparisons correction.

    Returns a per-hypothesis dict with keys: raw, corrected, family, method.
    """
    # Group by family while preserving insertion order so positional output
    # alignment is deterministic per call.
    families: dict[str, list[str]] = {}
    for hypothesis in p_values_by_hypothesis:
        fam = family_membership.get(hypothesis)
        if fam is None:
            raise ValueError(
                f"hypothesis {hypothesis!r} has no family in family_membership"
            )
        families.setdefault(fam, []).append(hypothesis)

    out: dict[str, dict] = {}
    for fam, members in families.items():
        method = METHOD_BY_FAMILY.get(fam)
        if method is None:
            raise ValueError(
                f"family {fam!r} has no correction method registered in "
                f"METHOD_BY_FAMILY; allowed: {sorted(METHOD_BY_FAMILY)}"
            )
        raw = [p_values_by_hypothesis[h] for h in members]
        if method == "holm":
            corrected = apply_holm_correction(raw)
        elif method == "bh":
            corrected = apply_bh_correction(raw)
        else:  # pragma: no cover — defensive; METHOD_BY_FAMILY is the gate
            raise ValueError(f"unknown method {method!r}")
        for h, r, c in zip(members, raw, corrected):
            out[h] = {
                "raw": r,
                "corrected": c,
                "family": fam,
                "method": method,
            }
    return out
