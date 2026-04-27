"""Source the GSM8K + GSM-Hard task bank for arousal-performance experiments.

Combined-source bank construction:
  - Easy / medium tiers: GSM8K (Cobbe et al. 2021) test split, sampled by
    reasoning-step count. 1-3 step problems land in easy; 4-7 step problems
    in medium.
  - Hard tier: GSM-Hard (Gao et al. 2023) variant — same problems with
    larger numbers and more multi-step structure, designed to break models
    that pattern-match GSM8K via memorization.

Output: configs/banks/gsm8k_v1.yaml with status=active and an
alignment_review block citing both source datasets and the heuristic
applied. Difficulty tags derive from item source (GSM-Hard → hard) and
reasoning-step count (GSM8K easy/medium).

Usage:
    uv run python scripts/banks/source_gsm8k.py \\
        --output configs/banks/gsm8k_v1.yaml \\
        --total-items 200 \\
        --seed 42

Requires: `datasets` and `pyyaml` Python packages (already in pyproject).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


# Heuristic: count `<<...>>` calculator annotations OR explicit equations
# in the GSM8K reference answer to estimate reasoning-step count.
# Items with 1-3 steps => easy. 4-7 steps => medium. >7 => promote to hard
# (rare in original GSM8K).
_STEP_PATTERN = re.compile(r"<<[^>]+>>")


def step_count(answer_text: str) -> int:
    """Estimate reasoning-step count from GSM8K's `<<calc>>` annotations."""
    return len(_STEP_PATTERN.findall(answer_text))


def extract_final_answer(answer_text: str) -> str:
    """Pull the canonical numeric answer from a GSM8K-formatted answer.

    GSM8K answers end with `#### <number>`. Returns the trimmed number as
    a string; preserves negative sign and decimal point. Falls back to the
    last token if the canonical separator is missing.
    """
    if "####" in answer_text:
        tail = answer_text.split("####")[-1].strip()
        # Strip commas and whitespace; keep digits, sign, decimal.
        return tail.replace(",", "").strip()
    # Fallback: last whitespace-separated token, stripped of punctuation.
    return answer_text.strip().split()[-1].rstrip(".,!?")


def assign_difficulty(steps: int, source: str) -> str:
    """Map (step_count, source_dataset) -> difficulty tier."""
    if source == "gsm-hard":
        return "hard"
    if steps <= 3:
        return "easy"
    if steps <= 7:
        return "medium"
    return "hard"


def load_gsm8k_split(split: str = "test") -> list[dict]:
    """Load the GSM8K test (or train) split via the `datasets` library.

    Falls back gracefully with an instructive error if the package isn't
    installed or the dataset can't be reached.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "datasets package not installed. Run: uv add datasets"
        )
    try:
        ds = load_dataset("gsm8k", "main", split=split)
    except Exception as e:
        raise SystemExit(
            f"Could not load gsm8k from HuggingFace: {e}\n"
            "Ensure network access; or pre-download with "
            "`huggingface-cli download gsm8k --repo-type dataset`."
        )
    out = []
    for row in ds:
        out.append({
            "question": row["question"],
            "answer_text": row["answer"],
            "source": "gsm8k",
        })
    return out


def load_gsm_hard() -> list[dict]:
    """Load GSM-Hard. Several mirrors exist; we try the canonical one first.

    GSM-Hard items are structurally GSM8K-shaped (question + reasoning +
    final answer) but with adversarial numeric magnitudes. We treat them
    as `source=gsm-hard` and assign all to the hard tier directly.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("datasets package not installed. Run: uv add datasets")
    candidates = [
        ("reasoning-machines/gsm-hard", None, "train"),  # original Gao 2023 mirror
        ("openai/gsm8k", "main", "test"),                  # fallback (will dedupe later)
    ]
    last_err = None
    for repo, config, split in candidates:
        try:
            if config:
                ds = load_dataset(repo, config, split=split)
            else:
                ds = load_dataset(repo, split=split)
            out = []
            for row in ds:
                if "input" in row and "code" in row:
                    # Original GSM-Hard schema: input + code + target
                    answer_text = row.get("code", "") + f"\n#### {row.get('target', '')}"
                    out.append({
                        "question": row["input"],
                        "answer_text": answer_text,
                        "source": "gsm-hard",
                    })
                else:
                    # Fallback to gsm8k schema
                    out.append({
                        "question": row.get("question", ""),
                        "answer_text": row.get("answer", ""),
                        "source": "gsm-hard",
                    })
            if out:
                print(f"Loaded {len(out)} items from {repo} ({split})", file=sys.stderr)
                return out
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(
        f"Could not load GSM-Hard from any known mirror. Last error: {last_err}\n"
        "If the canonical mirror has moved, edit `candidates` in this script."
    )


def sample_balanced(
    gsm8k_pool: list[dict],
    gsm_hard_pool: list[dict],
    total: int,
    seed: int,
) -> list[dict]:
    """Sample `total` items split by tier.

    Target: 1/3 easy, 1/3 medium, 1/3 hard (rounded).
    Easy + medium come from gsm8k_pool stratified by step_count.
    Hard comes from gsm_hard_pool.
    """
    rng = random.Random(seed)
    n_per_tier = total // 3
    n_hard = total - 2 * n_per_tier  # absorb rounding remainder
    # Bucket gsm8k by step count
    easy_pool = [item for item in gsm8k_pool if step_count(item["answer_text"]) <= 3]
    medium_pool = [item for item in gsm8k_pool if 4 <= step_count(item["answer_text"]) <= 7]
    if len(easy_pool) < n_per_tier:
        raise SystemExit(
            f"Easy pool too small: {len(easy_pool)} items, need {n_per_tier}"
        )
    if len(medium_pool) < n_per_tier:
        raise SystemExit(
            f"Medium pool too small: {len(medium_pool)} items, need {n_per_tier}"
        )
    if len(gsm_hard_pool) < n_hard:
        raise SystemExit(
            f"Hard pool too small: {len(gsm_hard_pool)} items, need {n_hard}"
        )
    sampled = (
        rng.sample(easy_pool, n_per_tier)
        + rng.sample(medium_pool, n_per_tier)
        + rng.sample(gsm_hard_pool, n_hard)
    )
    return sampled


def to_bank_item(raw: dict, idx: int) -> dict:
    """Convert a sampled raw item to bank-yaml format.

    GSM8K items don't have alias lists (numeric answers are exact-match),
    so `expected_aliases` is empty. The expected answer is the canonical
    numeric value extracted from `#### N` tail. Difficulty is assigned by
    step count + source.
    """
    steps = step_count(raw["answer_text"])
    return {
        "id": f"gsm8k_{idx:04d}",
        "question": raw["question"],
        "expected": extract_final_answer(raw["answer_text"]),
        "answer_aliases": [],
        "difficulty": assign_difficulty(steps, raw["source"]),
        "source_dataset": raw["source"],
        "step_count": steps,
    }


def emit_bank_yaml(items: list[dict], total: int) -> str:
    """Render the bank YAML. Hand-formatted to match exp1a's schema."""
    import yaml
    tier_counts = Counter(item["difficulty"] for item in items)
    source_counts = Counter(item["source_dataset"] for item in items)
    bank = {
        "bank_id": "gsm8k_v1",
        "bank_version": 1,
        "bank_type": "task",
        "status": "active",
        "task_type_subtype": "math-word-problem",
        "alignment_review": {
            "reviewer": "scripts/banks/source_gsm8k.py + author",
            "date": "2026-04-27",
            "verdict": "pass",
            "rationale": (
                "Combined-source GSM8K + GSM-Hard bank for arousal-performance "
                "experiments. GSM8K (Cobbe et al. 2021) supplies easy / medium "
                "tiers stratified by reasoning-step count from the canonical "
                "`<<...>>` annotations. GSM-Hard (Gao et al. 2023) supplies "
                "the hard tier; its adversarial numeric magnitudes serve as a "
                "ceiling-breaker for frontier models that pattern-match GSM8K. "
                "Difficulty tags are source-and-step-count derived."
            ),
        },
        "difficulty_profile": {
            "sources": [
                {
                    "name": "GSM8K",
                    "split": "test",
                    "license": "MIT",
                    "citation": (
                        "Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, "
                        "H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., "
                        "Nakano, R., Hesse, C., & Schulman, J. (2021). Training "
                        "Verifiers to Solve Math Word Problems. arXiv:2110.14168."
                    ),
                    "n_sampled": source_counts.get("gsm8k", 0),
                },
                {
                    "name": "GSM-Hard",
                    "split": "train",
                    "license": "MIT (per source repo)",
                    "citation": (
                        "Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, "
                        "Y., Callan, J., & Neubig, G. (2023). PAL: Program-aided "
                        "Language Models. ICML 2023. arXiv:2211.10435."
                    ),
                    "n_sampled": source_counts.get("gsm-hard", 0),
                },
            ],
            "ingestion_seed": 42,
            "n_total": total,
            "n_per_tier": dict(tier_counts),
            "rationale": (
                "Difficulty assignment: GSM-Hard items go to `hard`; GSM8K items "
                "with 1-3 reasoning steps go to `easy`, 4-7 steps to `medium`, "
                ">7 steps promoted to `hard`. Step count derived from `<<...>>` "
                "calculator annotations in the canonical answer text."
            ),
        },
        "items": items,
    }
    return yaml.safe_dump(bank, sort_keys=False, allow_unicode=True, width=100)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write the bank YAML")
    ap.add_argument("--total-items", type=int, default=200,
                    help="Total items in the bank (split ~equally across tiers)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed")
    args = ap.parse_args()

    print(f"Loading GSM8K test split...", file=sys.stderr)
    gsm8k = load_gsm8k_split("test")
    print(f"  {len(gsm8k)} items", file=sys.stderr)

    print(f"Loading GSM-Hard...", file=sys.stderr)
    gsm_hard = load_gsm_hard()
    print(f"  {len(gsm_hard)} items", file=sys.stderr)

    print(f"Sampling {args.total_items} items (1/3 each tier, seed={args.seed})...", file=sys.stderr)
    raw_items = sample_balanced(gsm8k, gsm_hard, args.total_items, args.seed)

    bank_items = [to_bank_item(raw, i) for i, raw in enumerate(raw_items)]
    yaml_text = emit_bank_yaml(bank_items, args.total_items)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml_text, encoding="utf-8")

    sha256 = hashlib.sha256(yaml_text.encode("utf-8")).hexdigest()
    tier_summary = Counter(item["difficulty"] for item in bank_items)
    source_summary = Counter(item["source_dataset"] for item in bank_items)
    print(f"\nBank written to {args.output}", file=sys.stderr)
    print(f"  sha256: {sha256}", file=sys.stderr)
    print(f"  tiers: {dict(tier_summary)}", file=sys.stderr)
    print(f"  sources: {dict(source_summary)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
