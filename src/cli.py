"""CLI entry point for the Affect Battery eval harness."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import yaml


def cmd_run(args):
    """Run an experiment from config."""
    from .runner import ExperimentConfig, ExperimentType, run_single, save_result
    from .conditioning.prompts import Condition
    from .models import VLLMClient, DryRunClient
    
    config = ExperimentConfig(
        model_name=args.model,
        condition=Condition(args.condition),
        experiment_type=ExperimentType(args.experiment),
        num_runs=args.num_runs,
        temperature=args.temperature,
        seed=args.seed,
    )
    
    if args.dry_run:
        client = DryRunClient(model=args.model)
    else:
        client = VLLMClient(base_url=args.base_url, model=args.model)
    
    output_dir = Path(args.output_dir)
    
    async def _run():
        for i in range(config.num_runs):
            result = await run_single(config, client, i)
            path = save_result(result, output_dir)
            print(f"[{i+1}/{config.num_runs}] Saved: {path}")
        if hasattr(client, 'close'):
            await client.close()
    
    asyncio.run(_run())


def cmd_pilot(args):
    """Run a quick pilot: 5 runs, all core conditions, one model."""
    from .runner import ExperimentConfig, ExperimentType, run_single, save_result
    from .conditioning.prompts import Condition
    from .models import VLLMClient, DryRunClient
    
    conditions = [Condition.STRONG_POSITIVE, Condition.NEUTRAL, Condition.STRONG_NEGATIVE]
    output_dir = Path(args.output_dir) / "pilot"
    
    if args.dry_run:
        client = DryRunClient(model=args.model)
    else:
        client = VLLMClient(base_url=args.base_url, model=args.model)
    
    async def _run():
        for cond in conditions:
            config = ExperimentConfig(
                model_name=args.model,
                condition=cond,
                experiment_type=ExperimentType.TRANSFER_WITHIN,
                num_runs=5,
                temperature=args.temperature,
                seed=42,
            )
            for i in range(5):
                result = await run_single(config, client, i)
                path = save_result(result, output_dir)
                print(f"[{cond.value}] Run {i+1}/5: {path}")
        if hasattr(client, 'close'):
            await client.close()
    
    asyncio.run(_run())
    print(f"\nPilot complete. Results in {output_dir}/")


def cmd_score(args):
    """Score existing result files."""
    from .scoring.accuracy import score_factual_qa
    from .scoring.hedging import hedge_summary
    
    results_dir = Path(args.results_dir)
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        transfer = data.get("transfer_responses", [])
        expected = data.get("transfer_expected", [])
        
        # Score accuracy
        correct = sum(
            score_factual_qa(r, e)
            for r, e in zip(transfer, expected)
            if e  # skip creative tasks with no expected answer
        )
        total = sum(1 for e in expected if e)
        accuracy = correct / total if total > 0 else 0
        
        # Score hedging
        all_text = " ".join(transfer)
        hedging = hedge_summary(all_text)
        
        cond = data.get("config", {}).get("condition", "?")
        print(f"{path.name}: accuracy={accuracy:.2f} hedging={hedging['normalized_per_100_words']:.1f}/100w ({cond})")


def main():
    parser = argparse.ArgumentParser(
        prog="affect-battery",
        description="Eval harness for the Affect Battery study",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    
    # run
    p_run = sub.add_parser("run", help="Run an experiment")
    p_run.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct)")
    p_run.add_argument("--condition", required=True, help="Condition: strong_positive, neutral, strong_negative, etc.")
    p_run.add_argument("--experiment", default="transfer_within", help="Experiment type")
    p_run.add_argument("--num-runs", type=int, default=50)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM API base URL")
    p_run.add_argument("--output-dir", default="results")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--dry-run", action="store_true", help="Use canned responses instead of real API")
    p_run.set_defaults(func=cmd_run)
    
    # pilot
    p_pilot = sub.add_parser("pilot", help="Quick pilot: 5 runs x 3 conditions x 1 model")
    p_pilot.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p_pilot.add_argument("--base-url", default="http://localhost:8000/v1")
    p_pilot.add_argument("--temperature", type=float, default=0.7)
    p_pilot.add_argument("--output-dir", default="results")
    p_pilot.add_argument("--dry-run", action="store_true")
    p_pilot.set_defaults(func=cmd_pilot)
    
    # score
    p_score = sub.add_parser("score", help="Score existing result files")
    p_score.add_argument("--results-dir", default="results", help="Directory with result JSON files")
    p_score.set_defaults(func=cmd_score)
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
