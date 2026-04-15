"""Hedging language detection for Experiment 3c (conservative shift)."""

import re
from dataclasses import dataclass
from enum import Enum


class HedgeCategory(str, Enum):
    EPISTEMIC = "epistemic"
    UNCERTAINTY = "uncertainty"
    QUALIFICATION = "qualification"
    CONFIDENCE_DISCLAIMER = "confidence_disclaimer"
    RLHF_SAFETY = "rlhf_safety"  # excluded from primary metric


@dataclass
class HedgeMatch:
    text: str
    category: HedgeCategory
    pattern_name: str
    start: int
    end: int


# Patterns keyed by category.
# Each entry: (pattern_name, compiled_regex)
HEDGE_PATTERNS: dict[HedgeCategory, list[tuple[str, re.Pattern]]] = {
    HedgeCategory.EPISTEMIC: [
        ("i_think_claim", re.compile(r'\bI think\b(?!\s+about\b)', re.IGNORECASE)),
        ("perhaps", re.compile(r'\bperhaps\b', re.IGNORECASE)),
        ("its_possible", re.compile(r"\bit(?:'s| is) possible\b", re.IGNORECASE)),
        ("i_believe", re.compile(r'\bI believe\b', re.IGNORECASE)),
        ("it_seems", re.compile(r'\bit seems\b', re.IGNORECASE)),
        ("arguably", re.compile(r'\barguably\b', re.IGNORECASE)),
    ],
    HedgeCategory.UNCERTAINTY: [
        ("not_sure", re.compile(r"\bI(?:'m| am) not (?:entirely |completely |fully )?sure\b", re.IGNORECASE)),
        ("might_be", re.compile(r'\bit might be\b', re.IGNORECASE)),
        ("could_be", re.compile(r'\bcould be\b', re.IGNORECASE)),
        ("not_certain", re.compile(r"\bI(?:'m| am) not (?:entirely |completely )?certain\b", re.IGNORECASE)),
        ("if_i_recall", re.compile(r'\bif I recall\b', re.IGNORECASE)),
    ],
    HedgeCategory.QUALIFICATION: [
        ("to_some_extent", re.compile(r'\bto some extent\b', re.IGNORECASE)),
        ("in_some_cases", re.compile(r'\bin some cases\b', re.IGNORECASE)),
        ("generally", re.compile(r'\bgenerally\b', re.IGNORECASE)),
        ("typically", re.compile(r'\btypically\b', re.IGNORECASE)),
        ("it_depends", re.compile(r'\bit depends\b', re.IGNORECASE)),
    ],
    HedgeCategory.CONFIDENCE_DISCLAIMER: [
        ("cant_be_certain", re.compile(r"\bI can(?:'t| not) be (?:entirely |completely )?certain\b", re.IGNORECASE)),
        ("not_enough_info", re.compile(r"\bI don(?:'t| not) have enough information\b", re.IGNORECASE)),
        ("hard_to_say", re.compile(r"\bit(?:'s| is) (?:hard|difficult) to say\b", re.IGNORECASE)),
    ],
    HedgeCategory.RLHF_SAFETY: [
        ("as_an_ai", re.compile(r'\bas an AI\b', re.IGNORECASE)),
        ("i_should_note", re.compile(r'\bI should note\b', re.IGNORECASE)),
        ("i_cant_provide", re.compile(r"\bI can(?:'t| not) provide\b", re.IGNORECASE)),
        ("important_to_note", re.compile(r"\bit(?:'s| is) important to (?:note|mention)\b", re.IGNORECASE)),
    ],
}


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
    """Counts per category, total (excluding RLHF_SAFETY), normalized by word count."""
    matches = detect_hedges(text)
    word_count = max(len(text.split()), 1)
    
    counts = {cat.value: 0 for cat in HedgeCategory}
    for m in matches:
        counts[m.category.value] += 1
    
    primary_total = sum(v for k, v in counts.items() if k != HedgeCategory.RLHF_SAFETY.value)
    
    return {
        "counts": counts,
        "total_primary": primary_total,
        "total_rlhf": counts[HedgeCategory.RLHF_SAFETY.value],
        "normalized_per_100_words": (primary_total / word_count) * 100,
        "word_count": word_count,
    }
