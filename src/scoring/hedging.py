"""Hedging language detection for Experiment 3c (conservative shift).

Patterns are loaded from configs/hedging_codebook.yaml at import time so that
pattern authoring (a collaborative deliverable per design.md Decision 3) is a
data edit rather than a code change. Akshansh's Ticket 6 merges into the YAML.
"""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class HedgeCategory(str, Enum):
    EPISTEMIC = "epistemic"
    UNCERTAINTY = "uncertainty"
    QUALIFICATION = "qualification"
    CONFIDENCE_DISCLAIMER = "confidence_disclaimer"
    RLHF_SAFETY = "rlhf_safety"


@dataclass
class HedgeMatch:
    text: str
    category: HedgeCategory
    pattern_name: str
    start: int
    end: int


CODEBOOK_PATH = Path(__file__).parent.parent.parent / "configs" / "hedging_codebook.yaml"


# Paper §3.4.3 hedging patterns (Task 7.4). The loader enforces presence;
# any flagged pattern missing from the YAML raises ValueError so silent
# pattern drift is impossible.
PAPER_REQUIRED_PATTERN_NAMES = frozenset({
    "i_think_claim",
    "not_sure",
    "could_be",
    "cant_be_certain",
})


def _load_codebook(path: Path = CODEBOOK_PATH) -> tuple[
    dict[HedgeCategory, list[tuple[str, re.Pattern]]],
    set[HedgeCategory],
    set[str],
]:
    """Load patterns and primary-exclusion set from YAML.

    Returns (patterns_by_category, exclusions) where exclusions are categories
    that must not be counted in the primary hedging metric.

    Per scoring-pipeline spec MODIFIED "Hedging language codebook with five
    categories" (Task 7.4): every pattern with `paper_pattern: true` from
    PAPER_REQUIRED_PATTERN_NAMES MUST be present, else ValueError.
    """
    raw = yaml.safe_load(path.read_text())
    name_to_category = {cat.name: cat for cat in HedgeCategory}

    patterns: dict[HedgeCategory, list[tuple[str, re.Pattern]]] = {
        cat: [] for cat in HedgeCategory
    }
    found_paper_patterns: set[str] = set()
    for cat_name, entries in raw["categories"].items():
        if cat_name not in name_to_category:
            raise ValueError(f"Unknown hedge category in codebook: {cat_name}")
        cat = name_to_category[cat_name]
        for entry in entries:
            pattern_name = entry["pattern_name"]
            regex = re.compile(entry["regex"], re.IGNORECASE)
            patterns[cat].append((pattern_name, regex))
            if entry.get("paper_pattern") is True:
                found_paper_patterns.add(pattern_name)

    missing_paper = PAPER_REQUIRED_PATTERN_NAMES - found_paper_patterns
    if missing_paper:
        raise ValueError(
            f"hedging codebook missing paper §3.4.3 patterns flagged "
            f"paper_pattern=true: {sorted(missing_paper)}. The loader "
            f"refuses to silently drop required patterns."
        )

    exclusions = {name_to_category[name] for name in raw.get("primary_exclusions", [])}
    return patterns, exclusions, found_paper_patterns


HEDGE_PATTERNS, PRIMARY_EXCLUSIONS, _PAPER_PATTERNS = _load_codebook()


def paper_required_patterns() -> set[str]:
    """Return the set of paper §3.4.3 pattern names present in the loaded
    codebook (entries with paper_pattern: true). Cached at module init
    via _load_codebook (review-finding #12). Returns a copy so callers
    cannot mutate the cached set."""
    return set(_PAPER_PATTERNS)


def detect_hedges(text: str) -> list[HedgeMatch]:
    """Find all hedge instances in text."""
    matches = []
    for category, patterns in HEDGE_PATTERNS.items():
        for name, regex in patterns:
            for m in regex.finditer(text):
                matches.append(HedgeMatch(
                    text=m.group(),
                    category=category,
                    pattern_name=name,
                    start=m.start(),
                    end=m.end(),
                ))
    return sorted(matches, key=lambda m: m.start)


def hedge_summary(text: str) -> dict:
    """Counts per category, total excluding PRIMARY_EXCLUSIONS, normalized per
    100 words."""
    matches = detect_hedges(text)
    word_count = max(len(text.split()), 1)

    counts = {cat.value: 0 for cat in HedgeCategory}
    for m in matches:
        counts[m.category.value] += 1

    exclusion_values = {cat.value for cat in PRIMARY_EXCLUSIONS}
    primary_total = sum(v for k, v in counts.items() if k not in exclusion_values)
    excluded_total = sum(v for k, v in counts.items() if k in exclusion_values)

    return {
        "counts": counts,
        "total_primary": primary_total,
        "total_rlhf": counts[HedgeCategory.RLHF_SAFETY.value],
        "total_excluded": excluded_total,
        "normalized_per_100_words": (primary_total / word_count) * 100,
        "word_count": word_count,
    }
