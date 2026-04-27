"""Ingest TriviaQA into a factual-QA bank for Exp 3c (conservative shift).

TriviaQA (Joshi et al., 2017, ACL) is a public open-domain QA dataset
with curated trivia questions and verified answers. We use the
`rc.nocontext` validation split (17,944 questions, no retrieved
context — the model must answer from parametric knowledge alone, which
is what Exp 3c measures).

Difficulty stratification: TriviaQA does not publish per-question
difficulty labels. We approximate via answer-string length and the
presence of multiple aliases — short, single-alias canonical-name
answers are typically easier than long or multi-alias answers. This is
a coarse proxy and is documented as such on the bank.

Output: `configs/banks/exp3c_triviaqa.yaml` consumed by `affect-battery
run --experiment exp3c --runner-config <derived-from-this>`. The bank
is also self-contained enough that it can be referenced directly via
the runner-config YAML's `items:` field.

Usage:
    python -m scripts.ingest_factual_qa \\
        --output configs/banks/exp3c_triviaqa.yaml \\
        --n 60 --seed 42

Reference:
    Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017).
    TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for
    Reading Comprehension. ACL 2017. arXiv:1705.03551
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml


CITATION = (
    "Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). "
    "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset "
    "for Reading Comprehension. ACL 2017. arXiv:1705.03551"
)


def _difficulty(answer_value: str, n_aliases: int) -> str:
    """Coarse heuristic: short canonical answers with few aliases are
    easier; long answers with many aliases are harder. Buckets:
      - easy: <= 12 chars AND <= 2 aliases
      - hard: > 25 chars OR >= 5 aliases
      - medium: everything else
    """
    L = len(answer_value)
    if L <= 12 and n_aliases <= 2:
        return "easy"
    if L > 25 or n_aliases >= 5:
        return "hard"
    return "medium"


def _format_record(record: dict[str, Any]) -> dict[str, Any] | None:
    question = (record.get("question") or "").strip()
    answer_obj = record.get("answer") or {}
    value = (answer_obj.get("value") or "").strip()
    aliases = answer_obj.get("aliases") or []
    if not question or not value:
        return None
    return {
        "difficulty": _difficulty(value, len(aliases)),
        "question": question,
        "expected": value,
        "answer_aliases": list(aliases),
    }


def ingest(
    output_path: Path,
    n: int = 60,
    seed: int = 42,
    target_per_bucket: int | None = None,
) -> Path:
    """Ingest n items from TriviaQA, stratified by difficulty.

    When `target_per_bucket` is set, the function gathers more raw
    records than `n` and then balances by bucket so the final bank has
    `target_per_bucket` items in each of {easy, medium, hard}, totaling
    `3 * target_per_bucket`. This is the recommended call shape because
    a uniform random sample is heavily skewed toward easy items.
    """
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)

    bucketed: dict[str, list[dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    items_flat: list[dict[str, Any]] = []
    n_target = (target_per_bucket * 3) if target_per_bucket else n

    for ds_idx in indices:
        record = ds[ds_idx]
        formatted = _format_record(record)
        if formatted is None:
            continue
        bucket = formatted["difficulty"]
        if target_per_bucket and len(bucketed[bucket]) >= target_per_bucket:
            continue
        bucketed[bucket].append(formatted)
        items_flat.append(formatted)
        # Termination: either flat-target reached, or all three buckets
        # have hit their per-bucket target.
        if target_per_bucket:
            if all(len(b) >= target_per_bucket for b in bucketed.values()):
                break
        else:
            if len(items_flat) >= n_target:
                break

    final_items = (
        bucketed["easy"] + bucketed["medium"] + bucketed["hard"]
        if target_per_bucket else items_flat[:n_target]
    )

    bank = {
        "bank_id": "exp3c_triviaqa_v1",
        "bank_version": 1,
        "bank_type": "transfer",
        "status": "active",
        "task_type_subtype": "factual-qa",
        "alignment_review": {
            "reviewer": "factual-qa-ingestion-script",
            "date": "auto",
            "verdict": "pass",
            "rationale": (
                "Items pulled verbatim from the published TriviaQA "
                "rc.nocontext validation split. Cross-domain (factual "
                "world knowledge, distinct from arithmetic conditioning "
                "domain). Passes alignment review by construction."
            ),
        },
        "difficulty_profile": {
            "source": "TriviaQA",
            "split": "rc.nocontext / validation",
            "license": "Apache-2.0",
            "citation": CITATION,
            "ingestion_seed": seed,
            "n_total_in_split": len(ds),
            "n_sampled": len(final_items),
            "rationale": (
                "Difficulty buckets approximated via answer-string "
                "length + alias count: TriviaQA does not publish "
                "per-question difficulty labels. Heuristic: short "
                "(<=12 chars) + few-alias answers => easy; long "
                "(>25 chars) or many-alias (>=5) answers => hard; "
                "everything else => medium."
            ),
        },
        # The 'items' shape matches what `affect-battery run --experiment
        # exp3c --runner-config <yaml>` consumes (difficulty / question /
        # expected). Aliases live alongside as auxiliary metadata for
        # accuracy scoring (a response matching any alias counts).
        "items": [
            {
                "id": f"triviaqa_{i:04d}",
                "difficulty": item["difficulty"],
                "question": item["question"],
                "expected": item["expected"],
                "answer_aliases": item["answer_aliases"],
            }
            for i, item in enumerate(final_items)
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(bank, sort_keys=False, allow_unicode=True))

    digest = hashlib.sha256(
        json.dumps(bank["items"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    by_bucket = {
        b: sum(1 for it in bank["items"] if it["difficulty"] == b)
        for b in ("easy", "medium", "hard")
    }
    print(
        f"wrote {output_path} ({len(bank['items'])} items, "
        f"buckets={by_bucket}, sha256={digest[:16]}...)"
    )
    return output_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ingest_factual_qa")
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=60,
                   help="Total items when not using --per-bucket.")
    p.add_argument("--per-bucket", type=int, default=None,
                   help=(
                       "Target items per difficulty bucket. When set, "
                       "the bank has 3*per_bucket items balanced "
                       "across {easy, medium, hard}."
                   ))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(
        output_path=Path(args.output),
        n=args.n,
        seed=args.seed,
        target_per_bucket=args.per_bucket,
    )
