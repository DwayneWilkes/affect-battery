"""Probe-runner parity check for Exp 3a.

Dispatches run_exp3a at the variance probe's parameters (n_per_level=15,
seed=42, gsm8k_v1 bank) and writes per-level mean and σ to a JSON for
side-by-side comparison with the recorded probe output.

If per-level σ from the runner matches the probe's recorded σ within
Monte Carlo error, the probe's σ estimates transfer cleanly to the
production runner and the n=122 run can proceed using the pre-reg's
power analysis. If σ diverges, surface as a blocker before n=122.

Usage:
    uv run python scripts/probes/h3a_runner_parity.py \\
        --bank configs/banks/gsm8k_v1.yaml \\
        --provider openai --model gpt-5.4-nano \\
        --pilot-seed-path configs/intensity_pilot_seed.json \\
        --output results/probes/h3a_runner_parity_2026-04-27.json

Use --dry-run for offline validation; canned responses produce sigma=0.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.conditioning.prompts import Condition  # noqa: E402
from src.models import (  # noqa: E402
    AnthropicClient,
    DryRunClient,
    OpenAIClient,
)
from src.runner import ExperimentConfig, ExperimentType  # noqa: E402
from src.runners.exp3a import run_exp3a  # noqa: E402


def make_client(provider: str, model: str, dry_run: bool):
    if dry_run:
        return DryRunClient(model=f"{provider}-{model}-dryrun")
    if provider == "openai":
        return OpenAIClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise ValueError(f"unsupported provider: {provider!r}")


async def run_parity(args) -> dict:
    client = make_client(args.provider, args.model, dry_run=args.dry_run)
    config = ExperimentConfig(
        model_name=args.model,
        condition=Condition.STRONG_POSITIVE,
        experiment_type=ExperimentType.AROUSAL_PERFORMANCE,
        num_runs=args.n_per_level,
        seed=args.seed,
        temperature=args.temperature,
        transfer_bank=str(args.bank),
    )

    by_level: dict[int, list[int]] = {i: [] for i in range(1, 8)}
    async for r in run_exp3a(
        config, client,
        intensity_levels=[1, 2, 3, 4, 5, 6, 7],
        pilot_seed_path=args.pilot_seed_path,
        output_dir=args.output.parent / "_parity_runs",
    ):
        by_level[r.body.intensity_level].append(r.body.binary_correct)

    mean_per_level = []
    sigma_per_level = []
    for level in range(1, 8):
        scores = by_level[level]
        if len(scores) >= 2:
            sigma = statistics.stdev(scores)
        else:
            sigma = 0.0
        mean_per_level.append(statistics.mean(scores) if scores else 0.0)
        sigma_per_level.append(sigma)

    return {
        "model": args.model,
        "provider": args.provider,
        "task": "GSM8K + GSM-Hard",
        "n_levels": 7,
        "n_per_level": args.n_per_level,
        "sigma_per_level": sigma_per_level,
        "mean_per_level": mean_per_level,
        "bank_path": str(args.bank),
        "pilot_seed_path": str(args.pilot_seed_path),
        "sample_seed": args.seed,
        "temperature": args.temperature,
        "dry_run": args.dry_run,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Probe-runner parity check. Compare sigma_per_level here "
            "against results/probes/h3a_variance_2026-04-27.json. If "
            "per-level sigma matches within Monte Carlo error, the "
            "probe's variance estimates transfer cleanly to the "
            "production runner and the n=122 run is power-justified."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-per-level", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--pilot-seed-path", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = asyncio.run(run_parity(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"\nParity output written to {args.output}", file=sys.stderr)
    print(f"  sigma_per_level: {[f'{s:.3f}' for s in result['sigma_per_level']]}", file=sys.stderr)
    print(f"  mean_per_level:  {[f'{m:.3f}' for m in result['mean_per_level']]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
