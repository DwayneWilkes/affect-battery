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


def cmd_pipeline_run(args) -> None:
    """Run the content-addressed pipeline from a YAML config.

    Stages default to the 6 canonical ones (bank_gen, calibration, gate,
    experiment, analysis, archive). The config can list a `stages:` subset
    to run only a prefix — useful for offline smoke-tests where the
    pod-dependent stages (calibration, experiment) are skipped.
    """
    from src.pipeline.stages import run_pipeline_from_config

    artifacts = run_pipeline_from_config(args.config)
    print("Pipeline complete. Accumulated artifacts:")
    for key, val in sorted(artifacts.items()):
        print(f"  {key}: {val}")


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
    """Run an experiment from CLI args.

    Dispatches through `src.runners.RUNNERS[args.experiment]` per design.md
    D5 — exp1a delegates to legacy run_batch; exp1b/2/3a/3b/3c raise
    NotImplementedError until their implementation tasks land.
    """
    from .runner import ExperimentConfig, ExperimentType
    from .runners import RUNNERS
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

    runner = RUNNERS[args.experiment]

    # Per-experiment extra kwargs loaded from --runner-config YAML when the
    # runner needs them. Per review-finding #6: exp3a/exp3b/exp3c require
    # additional positional kwargs (intensity_levels / pilot_seed_path,
    # prompts + n_generations, items) that aren't covered by the base
    # ExperimentConfig. We surface them via a YAML config rather than
    # adding a new --flag for each.
    extra_kwargs: dict = {}
    if args.experiment in {"exp3a", "exp3b", "exp3c"}:
        if not args.runner_config:
            print(
                f"error: --experiment {args.experiment} requires "
                f"--runner-config <path-to-yaml>",
                file=sys.stderr,
            )
            raise SystemExit(2)
        import yaml
        runner_cfg = yaml.safe_load(Path(args.runner_config).read_text())
        if args.experiment == "exp3a":
            extra_kwargs["intensity_levels"] = runner_cfg["intensity_levels"]
            extra_kwargs["pilot_seed_path"] = Path(runner_cfg["pilot_seed_path"])
        elif args.experiment == "exp3b":
            extra_kwargs["prompts"] = runner_cfg["prompts"]
            extra_kwargs["n_generations"] = runner_cfg.get("n_generations", 10)
        elif args.experiment == "exp3c":
            extra_kwargs["items"] = runner_cfg["items"]

    # exp1a / exp1b / exp2 delegate to run_batch and accept the standard
    # batch kwargs. exp3a-c don't (they have their own iteration shape).
    if args.experiment in {"exp1a", "exp1b", "exp2"}:
        batch_kwargs = dict(
            max_concurrent=args.max_concurrent,
            circuit_breaker_threshold=args.circuit_breaker_threshold,
            budget=budget,
            rate_limit_rps=args.rate_limit_rps,
            cancel_event=cancel_event,
        )
    else:
        batch_kwargs = {}

    async def _run():
        count = 0
        async for result in runner(
            config, client,
            output_dir=output_dir,
            **extra_kwargs,
            **batch_kwargs,
        ):
            count += 1
            print(f"[{count}] {result.config.get('condition')} run_number={result.run_number}")
        if hasattr(client, 'close'):
            await client.close()

    asyncio.run(_run())


def cmd_analyze(args):
    """End-to-end analysis: load result JSONs, render per-experiment
    reports + aggregate landing page (review-finding #11).

    Detects which experiments have results under <results-dir>/<exp>/
    and produces <results-dir>/<exp>_report.md for each, plus
    <results-dir>/AGGREGATE_REPORT.md tying them together.
    """
    from src.analysis.pipeline import analyze_results_dir

    rendered = analyze_results_dir(
        results_dir=Path(args.results_dir),
        model=args.model,
    )
    print(f"Rendered {len(rendered)} report(s):")
    for exp, path in sorted(rendered.items()):
        print(f"  {exp}: {path}")


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


EXPERIMENT_CHOICES = ["exp1a", "exp1b", "exp2", "exp3a", "exp3b", "exp3c"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Extracted from main() so tests can exercise the argparse structure
    without invoking commands. Per design.md D5: `run --experiment <name>`
    dispatches to src.runners.RUNNERS[<name>]; `probe <kind>` runs Week-0
    variance / base-model feasibility probes.
    """
    parser = argparse.ArgumentParser(
        prog="affect-battery",
        description="Eval harness for the Affect Battery study",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run an experiment")
    p_run.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct)")
    p_run.add_argument("--condition", required=True, help="Condition: strong_positive, neutral, strong_negative, etc.")
    p_run.add_argument("--experiment", default="exp1a", choices=EXPERIMENT_CHOICES,
                       help="Experiment type (paper §3 alignment; default exp1a)")
    p_run.add_argument("--num-runs", type=int, default=50)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM API base URL")
    p_run.add_argument("--output-dir", default="results")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--dry-run", action="store_true", help="Use canned responses instead of real API")
    p_run.add_argument("--base-model", action="store_true",
                       help="Use /v1/completions + few-shot scaffold (for non-instruct models).")
    p_run.add_argument(
        "--runner-config", default=None,
        help=(
            "YAML config with per-experiment extra kwargs. Required for "
            "exp3a (intensity_levels + pilot_seed_path), exp3b (prompts "
            "+ n_generations), exp3c (items). Example: "
            "configs/exp3a_runner.yaml"
        ),
    )
    p_run.add_argument("--bank", default=None,
                       help=f"Stimulus bank id (default: {DEFAULT_BANK_ID}). "
                            f"Must exist in configs/banks/<bank>.yaml.")
    _add_guardrail_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # pilot
    p_pilot = sub.add_parser("pilot", help="Quick pilot: 5 runs x 3 conditions x 1 model")
    p_pilot.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p_pilot.add_argument("--experiment", default="exp1a", choices=EXPERIMENT_CHOICES,
                         help="Experiment type (paper §3 alignment; default exp1a)")
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

    # pipeline (nested: pipeline run <config.yaml>)
    p_pipe = sub.add_parser(
        "pipeline",
        help="Content-addressed pipeline orchestrator",
    )
    pipe_sub = p_pipe.add_subparsers(dest="pipeline_command", required=True)
    p_pipe_run = pipe_sub.add_parser(
        "run",
        help="Run the pipeline from a YAML config",
    )
    p_pipe_run.add_argument(
        "config",
        help="Path to pipeline config YAML (stages, bank_gen params, cache_root, etc.)",
    )
    p_pipe_run.set_defaults(func=cmd_pipeline_run)

    # analyze: end-to-end analysis + per-experiment reports + aggregate.
    p_analyze = sub.add_parser(
        "analyze",
        help="Analyze result JSONs + render per-experiment + aggregate reports",
    )
    p_analyze.add_argument(
        "--results-dir", default="results",
        help="Directory containing per-experiment result subdirs (default: results)",
    )
    p_analyze.add_argument(
        "--model", default="aggregate",
        help="Label to record in the analysis output (default: 'aggregate')",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # probe (Week-0 probes per design.md Phase 1)
    p_probe = sub.add_parser(
        "probe",
        help="Week-0 probes: variance estimation + base-model feasibility",
    )
    probe_sub = p_probe.add_subparsers(dest="probe_kind", required=True)

    p_probe_variance = probe_sub.add_parser(
        "variance",
        help="Variance probe for MDE grounding (Task 1.1)",
    )
    p_probe_variance.add_argument("--model", required=True)
    p_probe_variance.add_argument("--base-url", default="http://localhost:8000/v1")
    p_probe_variance.add_argument("--n", type=int, default=20,
                                  help="Samples per condition (default 20).")
    p_probe_variance.add_argument("--output-dir", default="results/probes")
    p_probe_variance.add_argument("--dry-run", action="store_true")
    p_probe_variance.set_defaults(func=cmd_probe_variance)

    p_probe_base_model = probe_sub.add_parser(
        "base-model",
        help="Base-model feasibility probe (Task 1.2)",
    )
    p_probe_base_model.add_argument("--model", required=True)
    p_probe_base_model.add_argument("--base-url", default="http://localhost:8000/v1")
    p_probe_base_model.add_argument("--n", type=int, default=5,
                                    help="GSM8K problems for baseline accuracy (default 5).")
    p_probe_base_model.add_argument("--output-dir", default="results/probes")
    p_probe_base_model.add_argument("--dry-run", action="store_true")
    p_probe_base_model.set_defaults(func=cmd_probe_base_model)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


def cmd_probe_variance(args):
    """Variance probe — Task 1.1 implements."""
    raise NotImplementedError(
        "Variance probe not yet implemented; see tasks.md Task 1.1"
    )


def cmd_probe_base_model(args):
    """Base-model feasibility probe — Task 1.2 implements."""
    raise NotImplementedError(
        "Base-model probe not yet implemented; see tasks.md Task 1.2"
    )


if __name__ == "__main__":
    main()
