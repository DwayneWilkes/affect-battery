"""H3a variance probe.

Sweeps the 7 INTENSITY_LEVELS as system prompts against a math-word-problem
bank (typically the GSM8K + GSM-Hard bank) and measures per-level accuracy
variance. The output JSON feeds the simulation-based power analysis at
scripts/probes/h3a_power_report.py.

For each intensity level i in 1..7:
    Sample n_per_level problems from the bank.
    For each problem:
        messages = [
            {"role": "system",    "content": INTENSITY_LEVELS[i].feedback_text},
            {"role": "user",      "content": problem.question},
        ]
        response = client.complete(messages, temperature, max_tokens)
        extract numeric answer; score against problem.expected.
    Per-level: mean accuracy, sample standard deviation across n_per_level
    binary outcomes.

Output JSON shape (matches the input format of h3a_power_report.py):
    {
        "model":          "gpt-5.4-nano",
        "task":           "GSM8K + GSM-Hard",
        "n_levels":       7,
        "n_per_level":    15,
        "sigma_per_level": [s1, s2, ..., s7],
        "mean_per_level": [m1, m2, ..., m7],
        "icc":            null,
        "notes":          "..."
    }

Usage:
    uv run python scripts/probes/h3a_variance_probe.py \\
        --bank configs/banks/gsm8k_v1.yaml \\
        --provider openai --model gpt-5.4-nano \\
        --n-per-level 15 \\
        --output results/probes/h3a_variance_2026-04-27.json

Use --dry-run to validate the script end-to-end without API spend; canned
responses are returned by DryRunClient and accuracy will be near zero.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.conditioning.prompts import INTENSITY_LEVELS  # noqa: E402
from src.models import (  # noqa: E402
    AnthropicClient,
    DryRunClient,
    ModelClient,
    OpenAIClient,
)
from src.scoring.accuracy import extract_numeric_answer  # noqa: E402


def load_bank_items(bank_path: Path) -> list[dict]:
    """Load items from a bank YAML. Returns a list of {id, question, expected}."""
    data = yaml.safe_load(bank_path.read_text())
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError(f"{bank_path}: bank has no `items` array")
    out = []
    for item in items:
        out.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected"],
            "difficulty": item.get("difficulty"),
        })
    return out


def score_response(response: str, expected: str) -> int:
    """Score a single GSM8K response. Returns 1 on match, 0 otherwise.

    Uses extract_numeric_answer to pull the final number from the model's
    response (which often includes chain-of-reasoning). The expected value
    is converted to float; integer answers like "31" and float answers
    like "2796088.0" both compare cleanly.
    """
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return 0
    try:
        target = float(expected)
    except (TypeError, ValueError):
        return 0
    return int(abs(extracted - target) < 0.01)


def make_client(provider: str, model: str, dry_run: bool) -> ModelClient:
    if dry_run:
        # Returns canned responses; intended for end-to-end validation.
        return DryRunClient(model=f"{provider}-{model}-dryrun")
    if provider == "openai":
        return OpenAIClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise ValueError(f"unsupported provider: {provider!r}")


async def probe_one_level(
    client: ModelClient,
    intensity_text: str,
    items: list[dict],
    temperature: float,
    max_tokens: int,
) -> list[int]:
    """Run the model against `items` under one intensity stimulus.

    Returns a list of binary correctness scores, one per item.
    """
    scores = []
    for item in items:
        messages = [
            {"role": "system", "content": intensity_text},
            {"role": "user", "content": item["question"]},
        ]
        response = await client.complete(
            messages, temperature=temperature, max_tokens=max_tokens,
        )
        scores.append(score_response(response, item["expected"]))
    return scores


def sample_items(items: list[dict], n_per_level: int, n_levels: int, seed: int) -> list[list[dict]]:
    """Deterministically sample n_per_level items per level.

    Each level gets a disjoint random sample so per-level variance is not
    artificially correlated by sharing items across levels.
    """
    rng = random.Random(seed)
    if len(items) < n_per_level * n_levels:
        raise ValueError(
            f"bank has {len(items)} items; need at least "
            f"{n_per_level * n_levels} for non-overlapping per-level samples"
        )
    shuffled = items[:]
    rng.shuffle(shuffled)
    return [
        shuffled[i * n_per_level:(i + 1) * n_per_level]
        for i in range(n_levels)
    ]


async def run_probe(args) -> dict:
    bank_items = load_bank_items(args.bank)
    print(f"Loaded {len(bank_items)} items from {args.bank}", file=sys.stderr)

    if len(INTENSITY_LEVELS) != 7:
        raise ValueError(
            f"expected 7 INTENSITY_LEVELS; got {len(INTENSITY_LEVELS)}"
        )

    client = make_client(args.provider, args.model, dry_run=args.dry_run)
    per_level_items = sample_items(
        bank_items, args.n_per_level, len(INTENSITY_LEVELS), args.seed,
    )

    sigma_per_level: list[float] = []
    mean_per_level: list[float] = []
    for level_idx, level in enumerate(INTENSITY_LEVELS):
        items = per_level_items[level_idx]
        print(f"Level {level.level} ({level.label}): "
              f"{args.n_per_level} items...", file=sys.stderr)
        scores = await probe_one_level(
            client=client,
            intensity_text=level.feedback_text,
            items=items,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        # Sample standard deviation for the level. With binary outcomes the
        # mean and std are tightly coupled (std = sqrt(p(1-p))) but we
        # report sample-std anyway since that's what the power simulation
        # consumes.
        if len(scores) >= 2:
            sigma = statistics.stdev(scores)
        else:
            sigma = 0.0
        mean = statistics.mean(scores)
        print(f"  mean={mean:.3f}  sigma={sigma:.3f}", file=sys.stderr)
        sigma_per_level.append(sigma)
        mean_per_level.append(mean)

    # Compute a content hash over the bank + sample seed so the probe's
    # output is reproducible: same bank + same seed -> same items sampled.
    bank_hash = hashlib.sha256(args.bank.read_bytes()).hexdigest()

    return {
        "model": args.model,
        "provider": args.provider,
        "task": "GSM8K + GSM-Hard" if "gsm8k" in args.bank.name else args.bank.stem,
        "n_levels": len(INTENSITY_LEVELS),
        "n_per_level": args.n_per_level,
        "sigma_per_level": sigma_per_level,
        "mean_per_level": mean_per_level,
        "icc": None,
        "bank_path": str(args.bank),
        "bank_sha256": bank_hash,
        "sample_seed": args.seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "dry_run": args.dry_run,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Per-level accuracy variance probe. The sigma_per_level vector "
            "is the input to the H3a power simulation at "
            "scripts/probes/h3a_power_report.py. Each level's items are a "
            "disjoint deterministic sample from the bank, sized to "
            "n_per_level."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", required=True, type=Path,
                    help="Bank YAML to sample items from (e.g. configs/banks/gsm8k_v1.yaml)")
    ap.add_argument("--provider", default="openai",
                    choices=["openai", "anthropic"],
                    help="API provider")
    ap.add_argument("--model", required=True,
                    help="Model identifier (e.g. gpt-5.4-nano)")
    ap.add_argument("--n-per-level", type=int, default=15,
                    help="Items sampled per intensity level (default 15)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Sampling seed")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="Max output tokens per response (default 512)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Where to write the variance JSON")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use DryRunClient (no API calls). For end-to-end "
                         "smoke test; accuracy will be near zero.")
    args = ap.parse_args()

    if not args.bank.exists():
        print(f"bank file not found: {args.bank}", file=sys.stderr)
        return 2

    result = asyncio.run(run_probe(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nVariance probe written to {args.output}", file=sys.stderr)
    print(f"  sigma_per_level: {[f'{s:.3f}' for s in result['sigma_per_level']]}",
          file=sys.stderr)
    print(f"  mean_per_level:  {[f'{m:.3f}' for m in result['mean_per_level']]}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
