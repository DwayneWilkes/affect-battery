"""CLI entry point for the Affect Battery eval harness."""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from .conditioning.prompts import Condition


# Pilot default conditions: all seven. SELF_CHECK_NEUTRAL is included as the
# pre-registered length-matched control for STRONG_NEGATIVE so the pilot
# report can distinguish valence-mediated effects from length/metacognitive-
# mediated effects without requiring a per-invocation override. See
# specs/main/task-difficulty-calibration/spec.md "Default pilot conditions
# list" for the rationale.
DEFAULT_PILOT_CONDITIONS: tuple[Condition, ...] = (
    Condition.STRONG_POSITIVE,
    Condition.MILD_NEGATIVE,
    Condition.STRONG_NEGATIVE,
    Condition.NEUTRAL,
    Condition.NO_CONDITIONING,
    Condition.ACCURATE_NEGATIVE,
    Condition.SELF_CHECK_NEUTRAL,
)


DEFAULT_BANK_ID: str = "arithmetic_easy_v1"


def _list_available_banks() -> list[str]:
    """List available bank_ids by scanning configs/banks/*.yaml."""
    banks_dir = Path(__file__).resolve().parent.parent / "configs" / "banks"
    if not banks_dir.exists():
        return []
    return sorted(p.stem for p in banks_dir.glob("*.yaml"))


def _resolve_bank(bank_id: str | None) -> tuple[str, str]:
    """Validate `bank_id` against configs/banks/*.yaml listing and return
    (bank_id, stimulus_bank_hash). An unknown bank_id exits non-zero with
    a message listing known bank ids.

    When bank_id is None (caller didn't pass --bank), falls back to
    DEFAULT_BANK_ID for backward compatibility with pre-Group-5 invocations.
    """
    resolved_id = bank_id or DEFAULT_BANK_ID
    available = _list_available_banks()
    if resolved_id not in available:
        print(
            f"error: unknown bank '{resolved_id}'. "
            f"Available banks: {', '.join(available) if available else '(none found)'}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        from .conditioning.banks import ArithmeticBank
        bank = ArithmeticBank.load(resolved_id)
        return resolved_id, bank.stimulus_bank_hash
    except Exception as e:  # bank loader errors include BankNotFoundError
        print(
            f"error: failed to load bank '{resolved_id}': {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _install_sigint_handler(cancel_event: asyncio.Event) -> None:
    """Set cancel_event on SIGINT so run_batch drains gracefully.

    A second SIGINT is a no-op (the handler re-installs itself each time,
    which is idempotent). The OS-level default behaviour for a third
    (still Ctrl-C repeatedly) is preserved because we don't raise_signal.
    """
    def handler(signum, frame):
        if not cancel_event.is_set():
            print("\n[cli] SIGINT received; draining in-flight runs...", file=sys.stderr)
            cancel_event.set()

    signal.signal(signal.SIGINT, handler)


def _build_budget(args) -> object:
    """Build a BatchBudget from CLI args, or None if no caps configured."""
    from .runner import BatchBudget

    if args.budget_max_calls is None and args.cost_per_call is None:
        return None
    return BatchBudget(
        max_api_calls=args.budget_max_calls,
        cost_per_call=args.cost_per_call,
    )


def cmd_run(args):
    """Run an experiment from CLI args."""
    from .runner import ExperimentConfig, ExperimentType, run_batch
    from .models import VLLMClient, VLLMCompletionClient, DryRunClient

    bank_id, bank_hash = _resolve_bank(getattr(args, "bank", None))

    config = ExperimentConfig(
        model_name=args.model,
        condition=Condition(args.condition),
        experiment_type=ExperimentType(args.experiment),
        num_runs=args.num_runs,
        temperature=args.temperature,
        seed=args.seed,
        is_base_model=args.base_model,
        stimulus_bank=bank_id,
        stimulus_bank_hash=bank_hash,
    )

    if args.dry_run:
        client = DryRunClient(model=args.model)
    elif args.base_model:
        client = VLLMCompletionClient(base_url=args.base_url, model=args.model)
    else:
        client = VLLMClient(base_url=args.base_url, model=args.model)

    output_dir = Path(args.output_dir)
    budget = _build_budget(args)
    cancel_event = asyncio.Event()
    _install_sigint_handler(cancel_event)

    async def _run():
        count = 0
        async for result in run_batch(
            config, client,
            max_concurrent=args.max_concurrent,
            output_dir=output_dir,
            circuit_breaker_threshold=args.circuit_breaker_threshold,
            budget=budget,
            rate_limit_rps=args.rate_limit_rps,
            cancel_event=cancel_event,
        ):
            count += 1
            print(f"[{count}] {result.config.get('condition')} run_number={result.run_number}")
        if hasattr(client, 'close'):
            await client.close()

    asyncio.run(_run())


def cmd_pilot(args):
    """Run a quick pilot: 5 runs × all seven conditions × one model.

    The default conditions list is `DEFAULT_PILOT_CONDITIONS` (exposed at
    module level so the pipeline orchestrator and tests can reference it
    without invoking the CLI). SELF_CHECK_NEUTRAL is included as the
    pre-registered length-matched control for STRONG_NEGATIVE; see
    specs/main/task-difficulty-calibration/spec.md "Default pilot conditions
    list" for the rationale.
    """
    from .runner import ExperimentConfig, ExperimentType, run_batch
    from .models import VLLMClient, VLLMCompletionClient, DryRunClient

    bank_id, bank_hash = _resolve_bank(getattr(args, "bank", None))
    conditions = list(DEFAULT_PILOT_CONDITIONS)
    output_dir = Path(args.output_dir) / "pilot"

    if args.dry_run:
        client = DryRunClient(model=args.model)
    elif args.base_model:
        client = VLLMCompletionClient(base_url=args.base_url, model=args.model)
    else:
        client = VLLMClient(base_url=args.base_url, model=args.model)

    budget = _build_budget(args)
    cancel_event = asyncio.Event()
    _install_sigint_handler(cancel_event)

    async def _run():
        for cond in conditions:
            if cancel_event.is_set():
                break
            config = ExperimentConfig(
                model_name=args.model,
                condition=cond,
                experiment_type=ExperimentType.TRANSFER_WITHIN,
                num_runs=5,
                temperature=args.temperature,
                seed=42,
                is_base_model=args.base_model,
                stimulus_bank=bank_id,
                stimulus_bank_hash=bank_hash,
            )
            async for result in run_batch(
                config, client,
                max_concurrent=args.max_concurrent,
                output_dir=output_dir,
                circuit_breaker_threshold=args.circuit_breaker_threshold,
                budget=budget,
                rate_limit_rps=args.rate_limit_rps,
                cancel_event=cancel_event,
            ):
                print(f"[{cond.value}] run_number={result.run_number}")
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

        correct = sum(
            score_factual_qa(r, e)
            for r, e in zip(transfer, expected)
            if e  # skip creative tasks with no expected answer
        )
        total = sum(1 for e in expected if e)
        accuracy = correct / total if total > 0 else 0

        all_text = " ".join(transfer)
        hedging = hedge_summary(all_text)

        cond = data.get("config", {}).get("condition", "?")
        print(f"{path.name}: accuracy={accuracy:.2f} hedging={hedging['normalized_per_100_words']:.1f}/100w ({cond})")


def _add_guardrail_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by `run` and `pilot` for compute-guardrails control."""
    p.add_argument("--max-concurrent", type=int, default=5,
                   help="Max concurrent in-flight API calls (default 5).")
    p.add_argument("--budget-max-calls", type=int, default=None,
                   help="Hard cap on total API calls for this batch. None = unbounded.")
    p.add_argument("--cost-per-call", type=float, default=None,
                   help="Dollar cost per API call for pre-flight cost estimate. None = no cost estimate.")
    p.add_argument("--rate-limit-rps", type=float, default=None,
                   help="Token-bucket rate limit in calls per second. None = no rate limit.")
    p.add_argument("--circuit-breaker-threshold", type=int, default=5,
                   help="Halt after N consecutive non-retryable failures (default 5).")


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
    p_run.add_argument("--base-model", action="store_true",
                       help="Use /v1/completions + few-shot scaffold (for non-instruct models).")
    p_run.add_argument("--bank", default=None,
                       help=f"Stimulus bank id (default: {DEFAULT_BANK_ID}). "
                            f"Must exist in configs/banks/<bank>.yaml.")
    _add_guardrail_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # pilot
    p_pilot = sub.add_parser("pilot", help="Quick pilot: 5 runs x 3 conditions x 1 model")
    p_pilot.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p_pilot.add_argument("--base-url", default="http://localhost:8000/v1")
    p_pilot.add_argument("--temperature", type=float, default=0.7)
    p_pilot.add_argument("--output-dir", default="results")
    p_pilot.add_argument("--dry-run", action="store_true")
    p_pilot.add_argument("--base-model", action="store_true",
                         help="Use /v1/completions + few-shot scaffold (for non-instruct models).")
    p_pilot.add_argument("--bank", default=None,
                         help=f"Stimulus bank id (default: {DEFAULT_BANK_ID}). "
                              f"Must exist in configs/banks/<bank>.yaml.")
    _add_guardrail_args(p_pilot)
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
