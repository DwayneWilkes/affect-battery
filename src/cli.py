"""CLI entry point for the Affect Battery eval harness."""

import argparse
import asyncio
import json
import logging
import re
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
    """First SIGINT: set cancel_event and cancel pending asyncio tasks
    so the run drains gracefully (cached cells skipped, in-flight cells
    interrupted). Second SIGINT: hard-exit with the OS default handler.

    Uses `loop.add_signal_handler()` rather than `signal.signal()` so the
    handler can interrupt active `await` calls in addition to setting the
    flag. Combined with explicit task cancellation, awaits raise
    CancelledError immediately rather than blocking until the in-flight
    API call returns. Must be called from within an async context (so a
    running loop exists).
    """
    import os

    loop = asyncio.get_running_loop()

    def handler() -> None:
        if not cancel_event.is_set():
            print(
                "\n[cli] SIGINT received; cancelling in-flight runs. "
                "Press Ctrl-C again to force exit.",
                file=sys.stderr,
            )
            cancel_event.set()
            # Cancel every pending task except the one calling us.
            current = asyncio.current_task()
            for task in asyncio.all_tasks(loop):
                if task is not current and not task.done():
                    task.cancel()
        else:
            # Second SIGINT: restore default and re-raise so the
            # process exits immediately.
            print(
                "\n[cli] Second SIGINT — force exit.", file=sys.stderr,
            )
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)

    try:
        loop.add_signal_handler(signal.SIGINT, handler)
    except (NotImplementedError, RuntimeError):
        # Windows or non-main-thread asyncio loops don't support
        # add_signal_handler. Fall back to signal.signal() which
        # at least sets the flag (won't interrupt awaits but better
        # than nothing).
        def fallback(_signum, _frame):
            handler()
        signal.signal(signal.SIGINT, fallback)


def _build_budget(args) -> object:
    """Build a BatchBudget from CLI args, or None if no caps configured."""
    from .runner import BatchBudget

    if args.budget_max_calls is None and args.cost_per_call is None:
        return None
    return BatchBudget(
        max_api_calls=args.budget_max_calls,
        cost_per_call=args.cost_per_call,
    )


def _check_runtime_gates(args) -> None:
    """Refuse a real run without a pre-registration + power-report
    reference in the config. Both are recorded in every result file so
    downstream analysis can reconstruct what was pre-registered when
    the data was collected.

    Pre-registration vehicles accepted:
      - OSF: `--pre-registration-osf-url https://osf.io/<id>`
      - GitHub commit: `--pre-registration-github-commit owner/repo@<sha>`
        Equivalent in this project because the methodology lives in
        the repo (specs/, configs/osf_prereg_v1.yaml, scripts/). A
        signed Git tag at the cited commit provides timestamping +
        immutability. The OSF version can be filed later citing the
        same commit SHA.

    Per power-analysis spec "OSF pre-registration top-level gate" and
    "Data-collection gate". Bypass with --dry-run for offline testing.

    The --skip-prereg-gate / --skip-power-gate flags exist for the
    edge case of running an explicit pilot under an existing pre-reg
    that authorizes pilot inclusion; they emit the
    `pre_registration_violation: pilot_promoted_to_primary` audit code
    if the pilot results are subsequently promoted to the primary
    corpus (handled at analysis time, not runner time).
    """
    has_prereg = bool(
        args.pre_registration_osf_url or args.pre_registration_github_commit
    )
    if not args.skip_prereg_gate and not has_prereg:
        print(
            "error: missing pre-registration reference. Pass one of:\n"
            "  --pre-registration-osf-url <https://osf.io/...>\n"
            "  --pre-registration-github-commit <owner/repo@sha>\n"
            "or use --dry-run for offline testing. Bypass-with-rationale "
            "via --skip-prereg-gate (will emit a violation flag if pilot "
            "data is promoted to primary).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.pre_registration_github_commit:
        try:
            _validate_github_commit_ref(args.pre_registration_github_commit)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(2)
    if not args.skip_power_gate and not args.power_report_path:
        print(
            "error: missing power-report reference. Pass "
            "--power-report-path <path> --power-report-sha <sha256> or "
            "use --dry-run. Bypass-with-rationale via --skip-power-gate.",
            file=sys.stderr,
        )
        raise SystemExit(2)


_GITHUB_COMMIT_RE = re.compile(r"^[\w.-]+/[\w.-]+@[0-9a-f]{7,40}$")


def _validate_github_commit_ref(ref: str) -> None:
    """Format check for a GitHub commit reference: `owner/repo@<sha>`.

    Accepts 7-40 hex chars for the SHA so both short (display) and full
    forms work. Does NOT fetch the commit at runtime — network calls
    don't belong in the pre-flight gate. Result-file consumers can
    verify the commit exists out-of-band.
    """
    if not _GITHUB_COMMIT_RE.match(ref):
        raise ValueError(
            f"invalid GitHub commit ref {ref!r}; expected "
            "'owner/repo@<sha>' with 7-40 hex chars in the SHA"
        )


def _write_pilot_manifest(
    pilot_root: Path,
    args,
    conditions: list,
    bank_id: str,
    bank_hash: str,
    started_utc: str,
    completed_utc: str,
    per_cond_elapsed: dict[str, float],
    per_cond_count: dict[str, int],
) -> Path:
    """Write a manifest.yaml at the pilot root capturing everything needed
    to reproduce the run from disk alone: model, experiment, conditions,
    n, seed, transfer/stimulus banks, prereg refs, timing.

    The manifest is the canonical answer to 'what was in this pilot dir?'
    so consumers don't have to crack a JSON to find out.
    """
    import subprocess as _sp

    import yaml as _yaml

    manifest_path = pilot_root / "manifest.yaml"
    pilot_root.mkdir(parents=True, exist_ok=True)

    # git SHA: best-effort. If not in a repo (or git missing), record None
    # rather than crashing the pilot.
    try:
        sha = _sp.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True,
            stderr=_sp.DEVNULL,
        ).strip()
    except (_sp.CalledProcessError, FileNotFoundError, OSError):
        sha = None

    # Build prereg / power_report sections so they're informative rather
    # than emitting null subfields. When the gate is skipped, surface a
    # `status: "skipped (...)"` string; when it's set, surface the ref.
    osf = getattr(args, "pre_registration_osf_url", None)
    gh = getattr(args, "pre_registration_github_commit", None)
    if osf or gh:
        prereg = {"osf_url": osf, "github_commit": gh}
    elif getattr(args, "skip_prereg_gate", False):
        prereg = "skipped (--skip-prereg-gate)"
    elif getattr(args, "dry_run", False):
        prereg = "skipped (--dry-run)"
    else:
        prereg = "missing (gate would have failed)"

    power_path = getattr(args, "power_report_path", None)
    power_sha = getattr(args, "power_report_sha", None)
    if power_path or power_sha:
        power = {"path": power_path, "sha": power_sha}
    elif getattr(args, "skip_power_gate", False):
        power = "skipped (--skip-power-gate)"
    elif getattr(args, "dry_run", False):
        power = "skipped (--dry-run)"
    else:
        power = "missing (gate would have failed)"

    transfer_bank = getattr(args, "transfer_bank", None) or None
    if transfer_bank is None:
        transfer_bank = "hardcoded_factual_qa_v0_legacy (no --transfer-bank set)"

    payload: dict = {
        "pilot_kind": "exp1a_pilot" if args.experiment == "exp1a" else f"{args.experiment}_pilot",
        "model": args.model,
        "provider": getattr(args, "provider", None),
        "experiment": args.experiment,
        "conditions": [c.value if hasattr(c, "value") else str(c) for c in conditions],
        "num_runs": args.num_runs,
        "seed": args.seed,
        "temperature": getattr(args, "temperature", None),
        "is_base_model": getattr(args, "base_model", False),
        "stimulus_bank": {"id": bank_id, "sha256": bank_hash},
        "transfer_bank": transfer_bank,
        "pre_registration": prereg,
        "power_report": power,
        "git_sha": sha if sha else "unknown (not in a git repo)",
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "timing_per_condition": {
            cond: {
                "runs": per_cond_count.get(cond, 0),
                "total_seconds": round(per_cond_elapsed.get(cond, 0.0), 3),
            }
            for cond in (c.value if hasattr(c, "value") else str(c) for c in conditions)
        },
    }
    manifest_path.write_text(_yaml.safe_dump(payload, sort_keys=False))
    return manifest_path


def _maybe_backup_pilot_dir(pilot_root: Path, overwrite: bool) -> Path | None:
    """Move an existing non-empty pilot dir to a timestamped backup
    sibling so the prior run's state is preserved as an audit trail.

    Default (overwrite=False): no-op. Per-cell correctness is handled
    by the cache layer (`is_valid_cached_result` + transfer_bank_hash);
    same-config re-runs hit cache, differing-config re-runs invalidate
    stale cells per write.

    --overwrite=True: if the pilot dir exists AND has any *.json files
    or a manifest.yaml, move it to <pilot_root>.bak.<UTC-timestamp>/.
    No data is destroyed; you can `rm -rf` the backup once verified.
    """
    if not overwrite:
        return None
    if not pilot_root.exists():
        return None
    has_data = any(pilot_root.rglob("*.json"))
    has_manifest = (pilot_root / "manifest.yaml").exists()
    if not (has_data or has_manifest):
        return None
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = pilot_root.with_name(f"{pilot_root.name}.bak.{stamp}")
    pilot_root.rename(backup)
    return backup


def _hash_transfer_bank(bank_path: str | None) -> str:
    """SHA-256 over the transfer-bank file contents. Returns "" when no
    bank is set (the runner falls back to the legacy hardcoded pool).
    Used for cache identity so re-piloting with a different bank
    invalidates cached results from the prior bank.
    """
    if not bank_path:
        return ""
    import hashlib
    return hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()


# Anthropic API pricing in USD per million tokens, source:
#   https://platform.claude.com/docs/en/about-claude/pricing
# (snapshotted 2026-04-26). Input + output rates are split because
# output tokens cost 5x input on most tiers — the old blended-rate
# shortcut systematically over-estimated input-heavy workloads (long
# conversation context, short answers) and under-estimated output-
# heavy ones (open-ended generation in exp3b).
_ANTHROPIC_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00},
    "claude-haiku-3-5":  {"input": 0.80,  "output": 4.00},
    "claude-haiku-3":    {"input": 0.25,  "output": 1.25},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00,  "output": 15.00},
    "claude-sonnet-4":   {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-6":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-5":   {"input": 5.00,  "output": 25.00},
    "claude-opus-4-1":   {"input": 15.00, "output": 75.00},
    "claude-opus-4":     {"input": 15.00, "output": 75.00},
}


# OpenAI API pricing in USD per million tokens (Standard tier,
# short-context). Source:
#   https://developers.openai.com/api/docs/pricing
# (snapshotted 2026-04-26). Like Anthropic, output rates are 5-6x
# input. OpenAI's Batch + Flex tiers offer 50% discount on Standard
# (matching Anthropic's batch discount); Priority tier is 2-2.5x
# Standard for faster latency guarantees.
#
# Note: long-context pricing (>200k tokens) is roughly 2x these rates
# but our affect-battery prompts are well under 1k tokens, so the
# short-context column is what applies.
_OPENAI_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "gpt-5.5":      {"input": 5.00,  "output": 30.00},
    "gpt-5.5-pro":  {"input": 30.00, "output": 180.00},
    "gpt-5.4":      {"input": 2.50,  "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75,  "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20,  "output": 1.25},
    "gpt-5.4-pro":  {"input": 30.00, "output": 180.00},
    # Older generation rates retained for users on prior model IDs.
    # Verify against the current pricing page if these IDs are still
    # active for your account.
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":  {"input": 10.00, "output": 30.00},
}


# Per-turn token estimates for the affect-battery message shape.
# Sourced from manual inspection of result files in the 2026-04-26
# Haiku pilot. The conversation-growth model below sums these
# turn-by-turn so per-call input tokens reflect accumulating context.
_TOKENS_PER_TURN = {
    "system_prompt":         50,
    "conditioning_q":        20,
    "conditioning_a":         5,   # short numeric answer
    "conditioning_feedback": 30,
    "transfer_q":            30,
    "transfer_a_qa":         80,   # factual QA + hedging (exp1a/1b/2/3a/3c)
    "transfer_a_creative":  250,   # open-ended generation (exp3b)
}


# Anthropic prompt-caching minimum prefix size by model. Below this
# threshold cache_control markers are no-ops; the API processes the
# request without caching and there's no cost savings.
# Source: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
# (snapshotted 2026-04-26).
_PROMPT_CACHE_MIN_TOKENS = {
    "claude-haiku-4-5":  4096,
    "claude-haiku-3-5":  2048,
    "claude-sonnet-4-6": 2048,
    "claude-sonnet-4-5": 1024,
    "claude-sonnet-4":   1024,
    "claude-opus-4-7":   4096,
    "claude-opus-4-6":   4096,
    "claude-opus-4-5":   4096,
    "claude-opus-4-1":   1024,
    "claude-opus-4":     1024,
}


def _resolve_cache_threshold(model: str) -> int:
    """Minimum cacheable prefix length for `model`. Returns a
    conservative default (4096) when the model isn't in the table —
    erring high so we don't promise caching won't fire and then have
    it miss because the threshold turned out higher."""
    if model in _PROMPT_CACHE_MIN_TOKENS:
        return _PROMPT_CACHE_MIN_TOKENS[model]
    lo = model.lower()
    if "haiku-4" in lo or "opus-4-7" in lo or "opus-4-6" in lo or "opus-4-5" in lo:
        return 4096
    if "sonnet-4-6" in lo or "haiku-3-5" in lo:
        return 2048
    return 1024


def _estimate_call_token_sequence(args, extra_kwargs: dict) -> list[tuple[int, int]]:
    """Return [(input_tok, output_tok)] per API call in one
    (condition × run) cell, modeling conversation-length growth.

    Replaces the prior flat per-call token estimate. Each call's
    input now reflects the accumulated context the model would
    actually see at that point in the conversation, not a flat
    400-token average.
    """
    sys_tok = _TOKENS_PER_TURN["system_prompt"]
    cq = _TOKENS_PER_TURN["conditioning_q"]
    ca = _TOKENS_PER_TURN["conditioning_a"]
    cfb = _TOKENS_PER_TURN["conditioning_feedback"]
    tq = _TOKENS_PER_TURN["transfer_q"]
    cond_turns = getattr(args, "num_conditioning_turns", 5) or 5

    calls: list[tuple[int, int]] = []

    # Exp 3a is single-turn per pre-reg §3.4.1: each cell sends a
    # system message (intensity stimulus, ~22 tokens) + one user
    # question, with no conditioning prefix and no neutral buffers.
    # Short-circuit before the multi-turn conditioning accumulator
    # builds up.
    if args.experiment == "exp3a":
        levels = extra_kwargs.get("intensity_levels") or [1]
        ta = _TOKENS_PER_TURN["transfer_a_qa"]
        return [(sys_tok + tq, ta) for _ in levels]

    # Phase 1: conditioning. Each turn adds Q + A + feedback to the
    # accumulated context. Input on turn N includes the system prompt
    # plus all of turns 0..N-1.
    accumulated = sys_tok
    for _ in range(cond_turns):
        calls.append((accumulated + cq, ca))
        accumulated += cq + ca + cfb

    # Phase 1.5: neutral buffer turns (exp2 only). These accumulate
    # like conditioning turns.
    if args.experiment == "exp2":
        for _ in range(args.neutral_turns or 0):
            calls.append((accumulated + cq, ca))
            accumulated += cq + ca

    # Phase 2: transfer. Per-experiment dispatch.
    if args.experiment in {"exp1a", "exp2"}:
        n_transfer = getattr(args, "num_transfer_questions", 5) or 5
        ta = _TOKENS_PER_TURN["transfer_a_qa"]
        # Within-session transfer carries the conditioning history.
        for _ in range(n_transfer):
            calls.append((accumulated + tq, ta))
            accumulated += tq + ta
    elif args.experiment == "exp1b":
        # Cross-session resets context for the transfer phase per
        # the conditioning-protocol spec ("system prompt MUST be
        # identical across all conditions" for the fresh session).
        n_transfer = getattr(args, "num_transfer_questions", 5) or 5
        ta = _TOKENS_PER_TURN["transfer_a_qa"]
        transfer_acc = sys_tok  # fresh session, no conditioning carry-over
        for _ in range(n_transfer):
            calls.append((transfer_acc + tq, ta))
            transfer_acc += tq + ta
    elif args.experiment == "exp3b":
        # n_generations per prompt, each independent (re-uses the same
        # conditioning prefix; doesn't accumulate across generations).
        prompts = extra_kwargs.get("prompts") or []
        n_gens = extra_kwargs.get("n_generations", 10) or 10
        ta = _TOKENS_PER_TURN["transfer_a_creative"]
        for _ in prompts:
            for _ in range(n_gens):
                calls.append((accumulated + tq, ta))
    elif args.experiment == "exp3c":
        # One call per item, each independent.
        items = extra_kwargs.get("items") or []
        ta = _TOKENS_PER_TURN["transfer_a_qa"]
        for _ in items:
            calls.append((accumulated + tq, ta))

    return calls


def _resolve_token_pricing(model: str) -> tuple[float, float, str]:
    """Return (input_per_mtok, output_per_mtok, source) for a model.

    Source is one of:
      'exact'   — model name matched a row in the pricing table
      'tier'    — substring match on 'haiku' / 'sonnet' / 'opus' / 'gpt'
      'default' — no match; using a conservative mid-tier fallback
    """
    if model in _ANTHROPIC_PRICING_PER_MTOK:
        p = _ANTHROPIC_PRICING_PER_MTOK[model]
        return p["input"], p["output"], "exact"
    if model in _OPENAI_PRICING_PER_MTOK:
        p = _OPENAI_PRICING_PER_MTOK[model]
        return p["input"], p["output"], "exact"
    lo = model.lower()
    if "haiku" in lo:
        return 1.00, 5.00, "tier (haiku-4-5 rates)"
    if "sonnet" in lo:
        return 3.00, 15.00, "tier (sonnet-4-6 rates)"
    if "opus" in lo:
        return 5.00, 25.00, "tier (opus-4-7 rates; opus-4/4.1 are 3x higher)"
    if "nano" in lo:
        return 0.20, 1.25, "tier (gpt-5.4-nano rates)"
    if "mini" in lo:
        return 0.75, 4.50, "tier (gpt-5.4-mini rates)"
    if "gpt-5.4" in lo:
        return 2.50, 15.00, "tier (gpt-5.4 rates)"
    if "gpt-5" in lo:
        return 5.00, 30.00, "tier (gpt-5.5 rates)"
    if "gpt-4o" in lo or "gpt-4" in lo:
        return 2.50, 10.00, "tier (gpt-4o rates)"
    return 3.00, 15.00, "default (sonnet-tier fallback)"


def _empirical_seconds_per_call(model: str, results_root: Path) -> float | None:
    """Average wall-clock seconds per API call across prior pilots of this
    model. Pulled from manifest.yaml `timing_per_condition.*` and
    derived calls-per-result. Returns None when no prior pilot data
    exists for `model`.

    Lets `--estimate` give realistic time forecasts for repeat models
    rather than always falling back to coarse tier defaults.
    """
    import yaml as _yaml

    if not results_root.exists():
        return None
    samples: list[float] = []
    for manifest_path in results_root.glob("**/manifest.yaml"):
        try:
            m = _yaml.safe_load(manifest_path.read_text())
        except Exception:
            continue
        if not isinstance(m, dict) or m.get("model") != model:
            continue
        timing = m.get("timing_per_condition") or {}
        n_runs = m.get("num_runs") or 0
        n_cond = len(m.get("conditions") or [])
        cond_turns = m.get("num_conditioning_turns") or 5
        transfer_qs = m.get("num_transfer_questions") or 5
        calls_per_run = cond_turns + transfer_qs
        total_calls = n_runs * n_cond * calls_per_run
        # `wall_clock_seconds` is the tightest empirical signal; fall
        # back to summing per-condition wall_clock_seconds.
        wall = m.get("wall_clock_seconds")
        if wall is None:
            wall = sum(
                (t.get("wall_clock_seconds") or t.get("total_seconds") or 0)
                for t in timing.values() if isinstance(t, dict)
            )
        if total_calls > 0 and wall and wall > 0:
            samples.append(wall / total_calls)
    return sum(samples) / len(samples) if samples else None


def _yield_per_condition(args, extra_kwargs: dict) -> int:
    """Number of result files yielded per condition for the given
    experiment shape. Drives both the progress-bar total and the
    --estimate computation.
    """
    if args.experiment in {"exp1a", "exp1b", "exp2"}:
        return args.num_runs
    if args.experiment == "exp3a":
        return args.num_runs * len(extra_kwargs["intensity_levels"])
    if args.experiment == "exp3b":
        return args.num_runs * len(extra_kwargs["prompts"])
    if args.experiment == "exp3c":
        return args.num_runs * len(extra_kwargs["items"])
    return args.num_runs


def _estimate_pilot(args, conditions, extra_kwargs, per_cond_yield) -> dict:
    """Compute the (results, calls, cost, wall-clock) estimate without
    touching the API. Pure function over (args, conditions, extra_kwargs)
    so it's straightforward to test and to call from cmd_pilot/cmd_run."""
    n_cond = len(conditions)

    # Conversation-length growth model: per-call input tokens accumulate
    # across the run, so the average is below early calls and above
    # transfer calls. Compute the per-call sequence for one cell
    # (one (condition × run) trajectory), then multiply by n_cells.
    per_cell_calls = _estimate_call_token_sequence(args, extra_kwargs)
    cell_input_tokens = sum(c[0] for c in per_cell_calls)
    cell_output_tokens = sum(c[1] for c in per_cell_calls)
    cell_max_input = max((c[0] for c in per_cell_calls), default=0)

    # n_cells = (n_conditions × n_runs); each cell runs the per_cell_calls
    # sequence once. n_results uses the existing per_cond_yield definition
    # (1 result per run for exp1a/1b/2; intensity_levels/prompts/items per
    # run for exp3 family) so the display payload stays stable.
    n_cells = n_cond * args.num_runs
    n_calls = n_cells * len(per_cell_calls)
    n_results = n_cond * per_cond_yield
    calls_per_result = n_calls / n_results if n_results > 0 else 0.0
    total_input_tokens = n_cells * cell_input_tokens
    total_output_tokens = n_cells * cell_output_tokens

    # Cost: prefer proper input + output token pricing from the
    # pricing table. --cost-per-call overrides with a flat blended
    # rate (useful when the user has a contract rate or wants a
    # conservative ceiling).
    if getattr(args, "cost_per_call", None):
        cost_per_call = float(args.cost_per_call)
        total_cost_usd = n_calls * cost_per_call
        input_cost_usd = None
        output_cost_usd = None
        cost_source = "user-override (--cost-per-call, blended)"
        input_per_mtok = output_per_mtok = None
    else:
        input_per_mtok, output_per_mtok, cost_source = _resolve_token_pricing(args.model)
        input_cost_usd = total_input_tokens * input_per_mtok / 1_000_000
        output_cost_usd = total_output_tokens * output_per_mtok / 1_000_000
        total_cost_usd = input_cost_usd + output_cost_usd
        cost_per_call = total_cost_usd / n_calls if n_calls > 0 else 0.0

    # Prompt caching diagnostic: would caching engage at this prompt size?
    cache_threshold = _resolve_cache_threshold(args.model)
    cache_engages = cell_max_input >= cache_threshold

    # Wall-clock: prefer empirical from prior pilots of the same model.
    sec_per_call = _empirical_seconds_per_call(
        args.model, Path("results/pilots"),
    )
    if sec_per_call is None:
        # Tier defaults (rough; calibrated to typical SDK round-trip).
        lo = args.model.lower()
        if "haiku" in lo or "mini" in lo:
            sec_per_call = 1.5
        elif "sonnet" in lo:
            sec_per_call = 2.5
        elif "opus" in lo:
            sec_per_call = 4.0
        else:
            sec_per_call = 2.0
        time_source = "tier default"
    else:
        time_source = "empirical (prior pilot manifest)"

    max_concurrent = max(getattr(args, "max_concurrent", 1) or 1, 1)
    rate_limit = getattr(args, "rate_limit_rps", None)
    # Effective throughput is the lower of: concurrency-throughput vs
    # rate-limit-throughput. With high concurrency and unset rate limit,
    # concurrency dominates.
    concurrency_rps = max_concurrent / sec_per_call
    if rate_limit:
        effective_rps = min(concurrency_rps, float(rate_limit))
    else:
        effective_rps = concurrency_rps
    wall_clock_sec = n_calls / effective_rps + 5.0  # +5s overhead

    return {
        "model": args.model,
        "experiment": args.experiment,
        "n_conditions": n_cond,
        "num_runs": args.num_runs,
        "n_results_yielded": n_results,
        "n_api_calls": n_calls,
        "calls_per_result": round(calls_per_result, 2),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_input_per_call": (
            total_input_tokens / n_calls if n_calls > 0 else 0
        ),
        "avg_output_per_call": (
            total_output_tokens / n_calls if n_calls > 0 else 0
        ),
        "max_input_in_a_call": cell_max_input,
        "input_per_mtok_usd": input_per_mtok,
        "output_per_mtok_usd": output_per_mtok,
        "input_cost_usd": input_cost_usd,
        "output_cost_usd": output_cost_usd,
        "cost_per_call_usd": cost_per_call,
        "cost_source": cost_source,
        "total_cost_usd": total_cost_usd,
        "sec_per_call": sec_per_call,
        "time_source": time_source,
        "wall_clock_sec": wall_clock_sec,
        "wall_clock_min": wall_clock_sec / 60.0,
        "max_concurrent": max_concurrent,
        "effective_rps": effective_rps,
        "cache_threshold_tokens": cache_threshold,
        "cache_engages": cache_engages,
    }


def _print_estimate(est: dict) -> None:
    """Pretty-print the estimate dict from _estimate_pilot."""
    print(
        f"\nEstimate: {est['model']} / {est['experiment']} / "
        f"{est['num_runs']} runs × {est['n_conditions']} conditions"
    )
    print(f"  results yielded:    {est['n_results_yielded']}")
    print(
        f"  API calls:          {est['n_api_calls']:>6} "
        f"(~{est['calls_per_result']:.1f} calls/result)"
    )
    # Token + cost breakdown. When --cost-per-call overrides we print
    # the blended rate; otherwise we show the input/output split so
    # the user can sanity-check whether the workload is input-heavy
    # (long context, short answers) or output-heavy (open-ended gen).
    if est["input_per_mtok_usd"] is not None:
        in_cost = est["input_cost_usd"] or 0.0
        out_cost = est["output_cost_usd"] or 0.0
        avg_in = est["avg_input_per_call"]
        avg_out = est["avg_output_per_call"]
        max_in = est["max_input_in_a_call"]
        print(
            f"  tokens (avg):       "
            f"~{avg_in:.0f} input / ~{avg_out:.0f} output per call "
            f"(max input in a call: {max_in})"
        )
        print(
            f"  total tokens:       "
            f"{est['total_input_tokens']:,} input / "
            f"{est['total_output_tokens']:,} output"
        )
        print(
            f"  pricing:            "
            f"${est['input_per_mtok_usd']:.2f}/MTok in, "
            f"${est['output_per_mtok_usd']:.2f}/MTok out "
            f"({est['cost_source']})"
        )
        print(
            f"  cost (input):       ${in_cost:.2f} "
            f"({in_cost / est['total_cost_usd'] * 100:.0f}% of total)"
            if est['total_cost_usd'] > 0 else f"  cost (input):       ${in_cost:.2f}"
        )
        print(
            f"  cost (output):      ${out_cost:.2f} "
            f"({out_cost / est['total_cost_usd'] * 100:.0f}% of total)"
            if est['total_cost_usd'] > 0 else f"  cost (output):      ${out_cost:.2f}"
        )
        # Prompt-cache diagnostic — useful to see whether the protocol's
        # prompt size has crossed the model's caching threshold.
        if est["cache_engages"]:
            print(
                f"  prompt caching:     ENGAGES "
                f"(max input {max_in} ≥ threshold "
                f"{est['cache_threshold_tokens']}); cached input cost ~10% of base"
            )
        else:
            print(
                f"  prompt caching:     no-op "
                f"(max input {max_in} < threshold "
                f"{est['cache_threshold_tokens']} for {est['model']})"
            )
    else:
        print(
            f"  cost per call:      ${est['cost_per_call_usd']:.4f} "
            f"({est['cost_source']})"
        )
    print(f"  total cost:         ${est['total_cost_usd']:.2f}")
    print(
        f"  seconds per call:   {est['sec_per_call']:.2f}s "
        f"({est['time_source']})"
    )
    mins = est['wall_clock_min']
    if mins < 1.0:
        time_str = f"{est['wall_clock_sec']:.0f}s"
    elif mins < 60.0:
        time_str = f"{mins:.1f} min"
    else:
        time_str = f"{mins / 60.0:.1f} hr"
    print(
        f"  wall-clock:         {time_str} "
        f"(~{est['effective_rps']:.1f} calls/s effective)"
    )
    print(f"  max-concurrent:     {est['max_concurrent']}")
    # Machine-readable summary line so orchestrators can aggregate
    # cost + wall-clock across multiple per-experiment estimates.
    print(
        f"[ESTIMATE_SUMMARY] cost_usd={est['total_cost_usd']:.4f} "
        f"wall_clock_sec={est['wall_clock_sec']:.1f} "
        f"experiment={est['experiment']} model={est['model']}"
    )
    print()


def _quiet_per_run_logger() -> None:
    """Demote per-API-call INFO logs to DEBUG so the tqdm progress bar
    isn't interleaved with one log line per run. The runner module's
    'Run N/M | <condition> | <model>' message and the run_batch
    preflight summary both come from src.runner; this lifts the
    src.runner logger threshold to WARNING for the duration of the
    process.

    The SDK loggers (httpx / openai / anthropic) are silenced
    separately in main() at process startup.
    """
    logging.getLogger("src.runner").setLevel(logging.WARNING)


def _build_client(args):
    """Construct the right ModelClient subclass for `args.provider`.

    Provider matrix:
      - dry-run (any provider) -> DryRunClient
      - openai -> OpenAIClient (chat only; refuses --base-model)
      - anthropic -> AnthropicClient (chat only; refuses --base-model)
      - vllm + --base-model -> VLLMCompletionClient (raw completions
        endpoint for the few-shot scaffold path)
      - vllm (default) -> VLLMClient (chat-completions endpoint)

    The OpenAI / Anthropic clients refuse --base-model because neither
    provider exposes a raw-completion endpoint for their newer models;
    base-model studies require a vLLM endpoint with a non-instruct
    checkpoint.
    """
    from .models import (
        AnthropicClient,
        DryRunClient,
        OpenAIClient,
        VLLMClient,
        VLLMCompletionClient,
    )

    if args.dry_run:
        return DryRunClient(model=args.model)

    provider = getattr(args, "provider", "vllm")

    if provider == "openai":
        if args.base_model:
            print(
                "error: --base-model is not supported with --provider openai "
                "(no raw-completion endpoint on modern OpenAI chat models)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return OpenAIClient(model=args.model)

    if provider == "anthropic":
        if args.base_model:
            print(
                "error: --base-model is not supported with --provider anthropic "
                "(no raw-completion endpoint on Anthropic API)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return AnthropicClient(model=args.model)

    if provider != "vllm":
        print(
            f"error: unknown provider {provider!r}; expected one of "
            "{vllm, openai, anthropic}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if args.base_model:
        return VLLMCompletionClient(base_url=args.base_url, model=args.model)
    return VLLMClient(base_url=args.base_url, model=args.model)


def _build_runner_extra_kwargs(args) -> dict:
    """Per-experiment extra kwargs loaded from --runner-config YAML.

    exp3a/3b/3c require additional inputs (intensity_levels +
    pilot_seed_path, prompts + n_generations, items) that don't fit on
    the base ExperimentConfig. Both cmd_run and cmd_pilot consume this
    helper so the dispatch contract stays in one place.
    """
    extra: dict = {}
    if args.experiment not in {"exp3a", "exp3b", "exp3c"}:
        return extra
    if not getattr(args, "runner_config", None):
        print(
            f"error: --experiment {args.experiment} requires "
            f"--runner-config <path-to-yaml>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    import yaml as _yaml
    cfg = _yaml.safe_load(Path(args.runner_config).read_text())
    if args.experiment == "exp3a":
        extra["intensity_levels"] = cfg["intensity_levels"]
        extra["pilot_seed_path"] = Path(cfg["pilot_seed_path"])
        extra["sampling_mode"] = cfg.get("sampling_mode", "cross_level_disjoint")
    elif args.experiment == "exp3b":
        extra["prompts"] = cfg["prompts"]
        extra["n_generations"] = cfg.get("n_generations", 10)
    elif args.experiment == "exp3c":
        extra["items"] = cfg["items"]
    return extra


def _build_runner_batch_kwargs(args, budget, cancel_event) -> dict:
    """Per-experiment batch-control kwargs.

    exp1a/1b/2/3a accept the full batch_kwargs set (max_concurrent +
    circuit-breaker threshold + budget + rate limit). exp3b/3c implement
    their own loop and only accept the budget + rate limit + cancel
    subset. Centralized here so cmd_run and cmd_pilot don't drift.
    """
    if args.experiment in {"exp3b", "exp3c"}:
        return dict(
            budget=budget,
            rate_limit_rps=args.rate_limit_rps,
            cancel_event=cancel_event,
        )
    return dict(
        max_concurrent=args.max_concurrent,
        circuit_breaker_threshold=args.circuit_breaker_threshold,
        budget=budget,
        rate_limit_rps=args.rate_limit_rps,
        cancel_event=cancel_event,
    )


def cmd_run(args):
    """Run an experiment from CLI args.

    Dispatches through `src.runners.RUNNERS[args.experiment]` per design.md D5.

    Pre-registration + power-report gates fire before any runner-specific
    work. Both gates can be bypassed with --dry-run for local testing,
    but a real run with --pre-registration-osf-url and
    --power-report-path provided will validate the SHA chain before
    spending API budget.
    """
    from .runner import ExperimentConfig, ExperimentType
    from .runners import RUNNERS
    from .models import (
        AnthropicClient,
        DryRunClient,
        OpenAIClient,
        VLLMClient,
        VLLMCompletionClient,
    )

    _check_exp3a_cli_compat(args)
    _require_condition_for_non_exp3a(args)

    # Pre-flight gates: refuse to start a real run without OSF pre-reg
    # URL or a current power report. Skip in --dry-run mode (offline
    # testing path) per power-analysis spec exception.
    if not args.dry_run:
        _check_runtime_gates(args)

    bank_id, bank_hash = _resolve_bank(getattr(args, "bank", None))

    # exp3a's runner ignores condition (single-turn intensity-stimulus
    # paradigm per pre-reg §3.4.1); _check_exp3a_cli_compat already
    # rejected --condition for exp3a, so args.condition is None here.
    # Pick NEUTRAL as a typed placeholder so the ExperimentConfig
    # construction passes; the runner does not read this value when
    # constructing messages.
    condition_value = args.condition if args.condition is not None else "neutral"

    config = ExperimentConfig(
        model_name=args.model,
        condition=Condition(condition_value),
        experiment_type=ExperimentType(args.experiment),
        num_runs=args.num_runs,
        temperature=args.temperature,
        seed=args.seed,
        is_base_model=args.base_model,
        neutral_turns=args.neutral_turns,
        stimulus_bank=bank_id,
        stimulus_bank_hash=bank_hash,
        transfer_bank=getattr(args, "transfer_bank", None) or "",
        transfer_bank_hash=_hash_transfer_bank(getattr(args, "transfer_bank", None)),
    )

    client = _build_client(args)

    # Pilot-root layout: <output_dir> is the pilot root. Data lives under
    # <output_dir>/data/<experiment>/ so it sits alongside reports/ and
    # manifest.yaml. Same shape cmd_pilot uses; this gives full-experiment
    # runs the same on-disk structure as quick pilots.
    runner = RUNNERS[args.experiment]
    extra_kwargs = _build_runner_extra_kwargs(args)

    # --estimate: print cost + wall-clock and exit without running.
    # cmd_run is single-condition, so n_conditions = 1 in the estimate.
    if getattr(args, "estimate", False):
        est = _estimate_pilot(
            args,
            conditions=[Condition(condition_value)],
            extra_kwargs=extra_kwargs,
            per_cond_yield=_yield_per_condition(args, extra_kwargs),
        )
        _print_estimate(est)
        return

    pilot_root = Path(args.output_dir)
    backup = _maybe_backup_pilot_dir(pilot_root, getattr(args, "overwrite", False))
    if backup is not None:
        print(f"[--overwrite] moved prior pilot dir → {backup}", file=sys.stderr)
    output_dir = pilot_root / "data"
    budget = _build_budget(args)
    cancel_event = asyncio.Event()
    # SIGINT handler installed inside _run() below so a running loop
    # exists for loop.add_signal_handler.

    batch_kwargs = _build_runner_batch_kwargs(args, budget, cancel_event)

    # Single non-scrolling progress bar over the expected run count.
    # exp3a / exp3b iterate per (level / prompt) on top of num_runs;
    # we use args.num_runs as a conservative lower bound and let
    # tqdm's rate-of-progress display handle the actual count via
    # `total=None` overflow semantics for the multi-axis experiments.
    import time as _time
    from datetime import datetime, timezone
    from tqdm import tqdm
    _quiet_per_run_logger()
    bar_total = args.num_runs if args.experiment in {"exp1a", "exp1b", "exp2"} else None
    started = _time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    runs_completed = 0

    async def _run():
        nonlocal runs_completed
        _install_sigint_handler(cancel_event)
        with tqdm(
            total=bar_total,
            desc=f"{args.experiment}/{args.condition}",
            unit="run",
            dynamic_ncols=True,
        ) as bar:
            async for _result in runner(
                config, client,
                output_dir=output_dir,
                **extra_kwargs,
                **batch_kwargs,
            ):
                bar.update(1)
                runs_completed += 1
        if hasattr(client, 'close'):
            await client.close()

    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # SIGINT path: handler in _install_sigint_handler cancelled
        # in-flight tasks; CancelledError or KeyboardInterrupt bubbles
        # up here. Exit code 130 (128 + SIGINT) is the convention.
        print(
            "\n[cli] Cancelled. Partial results in "
            f"{output_dir}/ are still cache-valid for resume.",
            file=sys.stderr,
        )
        sys.exit(130)
    elapsed = _time.time() - started
    completed_utc = datetime.now(timezone.utc).isoformat()

    # Write a manifest at the pilot root. cmd_run is single-condition,
    # so the conditions list has exactly one entry; timing is per-run-batch
    # (not per-condition aggregate) since there's only one condition.
    _write_pilot_manifest(
        pilot_root=pilot_root,
        args=args,
        conditions=[Condition(condition_value)],
        bank_id=bank_id,
        bank_hash=bank_hash,
        started_utc=started_utc,
        completed_utc=completed_utc,
        per_cond_elapsed={condition_value: elapsed},
        per_cond_count={condition_value: runs_completed},
    )
    print(f"\nRun complete in {elapsed:.1f}s. Results in {output_dir}/")


def cmd_analyze(args):
    """End-to-end analysis: load result JSONs, render per-experiment
    reports + aggregate landing page.

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
    """Run a quick pilot: `args.num_runs` runs × all seven conditions ×
    one model.

    The default conditions list is `DEFAULT_PILOT_CONDITIONS` (exposed at
    module level so the pipeline orchestrator and tests can reference it
    without invoking the CLI). SELF_CHECK_NEUTRAL is included as the
    pre-registered length-matched control for STRONG_NEGATIVE; see
    specs/main/task-difficulty-calibration/spec.md "Default pilot conditions
    list" for the rationale.

    Pre-registration + power-report gates fire before any real run; pass
    --skip-prereg-gate / --skip-power-gate to bypass when running an
    explicit pilot under an existing pre-reg.
    """
    from .runner import ExperimentConfig, ExperimentType, run_batch
    from .runners import RUNNERS

    _check_exp3a_cli_compat(args)

    estimate_only = getattr(args, "estimate", False)
    # Skip runtime gates for --estimate (no API spend, no real run).
    if not args.dry_run and not estimate_only:
        _check_runtime_gates(args)

    bank_id, bank_hash = _resolve_bank(getattr(args, "bank", None))
    conditions = list(DEFAULT_PILOT_CONDITIONS)
    experiment_type = ExperimentType(args.experiment)

    # Per-experiment runner dispatch (parity with cmd_run). exp3a/3b/3c
    # need --runner-config; the helper raises SystemExit(2) if missing.
    # Built early so --estimate can read the multipliers without
    # constructing a client or touching disk.
    runner = RUNNERS[args.experiment]
    extra_kwargs = _build_runner_extra_kwargs(args)
    per_cond_yield_for_est = _yield_per_condition(args, extra_kwargs)

    # --estimate: print cost + wall-clock and return without running.
    if estimate_only:
        est = _estimate_pilot(args, conditions, extra_kwargs, per_cond_yield_for_est)
        _print_estimate(est)
        return

    # Pilot-root layout: <output_dir> is the pilot root. Data lives under
    # <output_dir>/data/<experiment>/ so it can sit alongside reports/
    # (written by analyze) and manifest.yaml (written below).
    pilot_root = Path(args.output_dir)
    backup = _maybe_backup_pilot_dir(pilot_root, getattr(args, "overwrite", False))
    if backup is not None:
        print(f"[--overwrite] moved prior pilot dir → {backup}", file=sys.stderr)
    output_dir = pilot_root / "data"

    client = _build_client(args)

    budget = _build_budget(args)
    cancel_event = asyncio.Event()
    # SIGINT handler installed inside _run() below so a running loop
    # exists for loop.add_signal_handler (asyncio-aware).

    # Single non-scrolling progress bar across (conditions × num_runs).
    # The runner.py per-run INFO logger continues to fire but is
    # demoted to DEBUG via _quiet_per_run_logger() so it doesn't
    # interleave with the bar.
    import time as _time
    from datetime import datetime, timezone
    from tqdm import tqdm
    _quiet_per_run_logger()

    # Bar total: results per condition × n_conditions. Computed via the
    # shared `_yield_per_condition` helper so the progress bar and the
    # --estimate calculator agree on the work breakdown.
    per_cond_yield = _yield_per_condition(args, extra_kwargs)
    bar_total = len(conditions) * per_cond_yield
    started = _time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    # Per-condition timing for the end-of-run summary table.
    per_cond_elapsed: dict[str, float] = {}
    per_cond_count: dict[str, int] = {}

    async def _run():
        _install_sigint_handler(cancel_event)
        with tqdm(
            total=bar_total, desc="pilot", unit="run", dynamic_ncols=True,
        ) as bar:
            for cond in conditions:
                if cancel_event.is_set():
                    break
                cond_started = _time.time()
                cond_completed = 0
                bar.set_postfix_str(f"cond={cond.value}", refresh=True)
                config = ExperimentConfig(
                    model_name=args.model,
                    condition=cond,
                    experiment_type=experiment_type,
                    num_runs=args.num_runs,
                    temperature=args.temperature,
                    seed=args.seed,
                    is_base_model=args.base_model,
                    # neutral_turns is the persistence-recovery sweep
                    # parameter (exp2). Default 0 for non-exp2 experiments.
                    neutral_turns=getattr(args, "neutral_turns", 0) or 0,
                    stimulus_bank=bank_id,
                    stimulus_bank_hash=bank_hash,
                    transfer_bank=getattr(args, "transfer_bank", None) or "",
                    transfer_bank_hash=_hash_transfer_bank(
                        getattr(args, "transfer_bank", None)
                    ),
                )
                # Build batch_kwargs per-condition since cancel_event +
                # budget are shared across the loop.
                batch_kwargs = _build_runner_batch_kwargs(
                    args, budget, cancel_event,
                )
                async for _result in runner(
                    config, client,
                    output_dir=output_dir,
                    **extra_kwargs,
                    **batch_kwargs,
                ):
                    bar.update(1)
                    cond_completed += 1
                cond_elapsed = _time.time() - cond_started
                per_cond_elapsed[cond.value] = cond_elapsed
                per_cond_count[cond.value] = cond_completed
                # Persist a per-condition summary line above the bar so
                # the user sees timing per-feature as it accumulates.
                avg = cond_elapsed / max(cond_completed, 1)
                tqdm.write(
                    f"  done {cond.value:<20} {cond_completed} runs "
                    f"in {cond_elapsed:5.1f}s ({avg:.2f}s/run)"
                )
        if hasattr(client, 'close'):
            await client.close()

    try:
        asyncio.run(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # SIGINT path: handler in _install_sigint_handler cancelled
        # in-flight tasks; CancelledError or KeyboardInterrupt bubbles
        # up here. Exit code 130 (128 + SIGINT) is the convention.
        print(
            "\n[cli] Cancelled. Partial results in "
            f"{output_dir}/ are still cache-valid for resume.",
            file=sys.stderr,
        )
        sys.exit(130)
    elapsed = _time.time() - started
    completed_utc = datetime.now(timezone.utc).isoformat()

    # Persist a manifest at the pilot root so the run is self-documenting.
    _write_pilot_manifest(
        pilot_root=pilot_root,
        args=args,
        conditions=conditions,
        bank_id=bank_id,
        bank_hash=bank_hash,
        started_utc=started_utc,
        completed_utc=completed_utc,
        per_cond_elapsed=per_cond_elapsed,
        per_cond_count=per_cond_count,
    )

    # Aggregate timing table: per-condition + total. Aligned columns make
    # this scannable when conditions vary in cost (e.g. strong_negative
    # tends to be longer than no_conditioning because the model writes
    # more in negative-affect states).
    print()
    print(f"  {'condition':<20} {'runs':>5}  {'total':>8}  {'per-run':>8}")
    print(f"  {'-' * 20} {'-' * 5}  {'-' * 8}  {'-' * 8}")
    for cond in conditions:
        c = cond.value
        if c not in per_cond_elapsed:
            continue
        n = per_cond_count[c]
        t = per_cond_elapsed[c]
        print(f"  {c:<20} {n:>5}  {t:>7.1f}s  {t / max(n, 1):>7.2f}s")
    print(f"  {'-' * 20} {'-' * 5}  {'-' * 8}  {'-' * 8}")
    n_total = sum(per_cond_count.values())
    print(
        f"  {'TOTAL':<20} {n_total:>5}  {elapsed:>7.1f}s  "
        f"{elapsed / max(n_total, 1):>7.2f}s"
    )
    # Machine-readable summary so the multi-experiment orchestrator can
    # aggregate elapsed + estimated cost across all 5 experiments. We
    # call the estimator post-hoc with the same config to get the cost
    # number; wall-clock is the actual measured value.
    try:
        post_est = _estimate_pilot(
            args, conditions, extra_kwargs, per_cond_yield_for_est,
        )
        cost_str = f"cost_usd={post_est['total_cost_usd']:.4f}"
    except Exception:
        cost_str = "cost_usd=unknown"
    # Include yielded-count so the orchestrator can distinguish "ran
    # successfully" from "circuit-broke at start". Both look the same
    # to the SDK exit code (cmd_pilot returns 0 in both cases) but the
    # yielded count is the truth.
    expected_results = len(conditions) * (per_cond_yield_for_est or args.num_runs)
    # Recompute cost using token-based pricing for the [RUN_SUMMARY] —
    # ignoring any --cost-per-call override the bash script passes
    # (that's a conservative blended rate for the budget guard, not
    # an accurate cost number). _estimate_pilot honors --cost-per-call
    # by default; we need to bypass it for honest post-hoc reporting.
    try:
        # Build a shadow-args namespace with cost_per_call=None so the
        # estimator uses tier-based input/output pricing.
        from types import SimpleNamespace as _SN
        shadow = _SN(**{
            k: getattr(args, k) for k in dir(args)
            if not k.startswith("_") and not callable(getattr(args, k, None))
        })
        shadow.cost_per_call = None
        token_est = _estimate_pilot(
            shadow, conditions, extra_kwargs, per_cond_yield_for_est,
        )
        cost_str = f"cost_usd={token_est['total_cost_usd']:.4f}"
    except Exception:
        # If shadow-args trick fails, fall back to the blended-rate
        # post-hoc number (still better than nothing).
        pass
    print(
        f"[RUN_SUMMARY] {cost_str} wall_clock_sec={elapsed:.1f} "
        f"yielded={n_total} expected={expected_results} "
        f"experiment={args.experiment} model={args.model}"
    )
    if n_total == 0 and expected_results > 0:
        # Zero results yielded despite expecting some — the circuit
        # breaker likely opened or every cell hit a NonRetryableAPIError.
        # Make the failure mode loud at the CLI exit code so the
        # orchestrator's success/fail count reflects reality.
        print(
            f"\nERROR: pilot yielded 0 of {expected_results} expected results. "
            f"Check {output_dir}/events.jsonl for circuit_open / run_failed events.",
            file=sys.stderr,
        )
        sys.exit(3)
    print(f"\nResults in {output_dir}/")


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


def _check_exp3a_cli_compat(args) -> None:
    """Reject CLI flags that have no effect on Exp 3a.

    Exp 3a runs a single-turn intensity-stimulus paradigm per
    pre-registration §3.4.1. --condition and --neutral-turns
    parameterize multi-turn conditioning and have no effect on the
    messages sent to the model. Silently accepting them would pollute
    manifests with values that look like experimental variables in
    later meta-analysis. Other experiments are unaffected by this
    check.
    """
    if getattr(args, "experiment", None) != "exp3a":
        return
    if getattr(args, "condition", None) is not None:
        print(
            "error: --condition is incompatible with --experiment exp3a; "
            "intensity levels are the sole stimulus per pre-reg §3.4.1.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if getattr(args, "neutral_turns", 0) != 0:
        print(
            "error: --neutral-turns is incompatible with --experiment exp3a "
            "(single-turn paradigm).",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _require_condition_for_non_exp3a(args) -> None:
    """The `run` subcommand requires --condition for every experiment except
    exp3a. (`pilot` sweeps DEFAULT_PILOT_CONDITIONS automatically and does
    not accept --condition.)"""
    if getattr(args, "experiment", None) == "exp3a":
        return
    if getattr(args, "condition", None) is None:
        print(
            f"error: --condition is required for --experiment {args.experiment}.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _add_prereg_gate_args(p: argparse.ArgumentParser) -> None:
    """Pre-registration + power-report gating flags shared between
    `run` and `pilot`. Both subcommands enforce the same gates so a
    pilot that escalates to primary without amendment is detectable."""
    p.add_argument(
        "--pre-registration-osf-url", default=None,
        help=(
            "OSF pre-registration URL (https://osf.io/...). Either this "
            "or --pre-registration-github-commit is required for "
            "non-dry-run invocations. Recorded in every result file."
        ),
    )
    p.add_argument(
        "--pre-registration-github-commit", default=None,
        help=(
            "GitHub commit reference of the pre-registration "
            "(owner/repo@sha). Equivalent to --pre-registration-osf-url "
            "for projects whose methodology is fully encoded in the "
            "repo (specs/, configs/, scripts/); a signed Git tag at "
            "the commit provides timestamping. Recorded in every "
            "result file."
        ),
    )
    p.add_argument(
        "--power-report-path", default=None,
        help=(
            "Path to the per-experiment power report JSON. Required for "
            "non-dry-run invocations. Recorded in every result file's "
            "config alongside --power-report-sha."
        ),
    )
    p.add_argument(
        "--power-report-sha", default=None,
        help="SHA-256 of --power-report-path (passed for cross-check).",
    )
    p.add_argument(
        "--skip-prereg-gate", action="store_true",
        help=(
            "Bypass the pre-registration gate. ONLY for explicit pilot "
            "runs under a pre-registration that authorizes pilot "
            "inclusion. Promoting pilot results to primary will emit a "
            "pre_registration_violation audit flag at analysis time."
        ),
    )
    p.add_argument(
        "--skip-power-gate", action="store_true",
        help="Bypass the power-report gate (paired with --skip-prereg-gate).",
    )


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
    p_run.add_argument("--condition", default=None,
                       help="Condition: strong_positive, neutral, strong_negative, etc. "
                            "Required for exp1a/1b/2/3b/3c; rejected for exp3a "
                            "(intensity levels are the sole stimulus per pre-reg §3.4.1).")
    p_run.add_argument("--experiment", default="exp1a", choices=EXPERIMENT_CHOICES,
                       help="Experiment type (paper §3 alignment; default exp1a)")
    p_run.add_argument("--num-runs", type=int, default=50)
    p_run.add_argument("--temperature", type=float, default=0.7)
    p_run.add_argument(
        "--provider", default="vllm", choices=["vllm", "openai", "anthropic"],
        help=(
            "Inference provider. 'vllm' (default) uses a local/remote "
            "OpenAI-compatible server (typically RunPod). 'openai' uses "
            "the OpenAI API (OPENAI_API_KEY env var). 'anthropic' uses "
            "the Anthropic API (ANTHROPIC_API_KEY env var). "
            "OpenAI/Anthropic do not support --base-model."
        ),
    )
    p_run.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM API base URL (ignored for openai/anthropic providers)")
    p_run.add_argument("--output-dir", default="results")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--dry-run", action="store_true", help="Use canned responses instead of real API")
    p_run.add_argument("--base-model", action="store_true",
                       help="Use /v1/completions + few-shot scaffold (for non-instruct models).")
    p_run.add_argument(
        "--neutral-turns", type=int, default=0,
        help=(
            "Number of neutral buffer turns between conditioning and "
            "transfer. For Exp 2 N-values sweeps, set this to the N value "
            "of the slice (1, 3, 5, or 10). Default 0 (Exp 1a/1b)."
        ),
    )
    p_run.add_argument(
        "--runner-config", default=None,
        help=(
            "YAML config with per-experiment extra kwargs. Required for "
            "exp3a (intensity_levels + pilot_seed_path), exp3b (prompts "
            "+ n_generations), exp3c (items). Example: "
            "configs/exp3a_runner.yaml"
        ),
    )
    _add_prereg_gate_args(p_run)
    p_run.add_argument("--bank", default=None,
                       help=f"Stimulus bank id (default: {DEFAULT_BANK_ID}). "
                            f"Must exist in configs/banks/<bank>.yaml.")
    p_run.add_argument(
        "--transfer-bank", default=None, dest="transfer_bank",
        help="Path to a transfer-question bank YAML "
             "(e.g. configs/banks/exp1a_factual_qa_hard_v1.yaml). "
             "When unset, the runner falls back to the hardcoded "
             "factual_qa pool — fine for legacy parity, but will "
             "ceiling out on frontier models.",
    )
    p_run.add_argument(
        "--overwrite", action="store_true",
        help="Move any existing pilot dir at --output-dir to "
             "<dir>.bak.<UTC-timestamp>/ before running. Default "
             "(no flag) is resume-by-default: the cache layer handles "
             "per-cell correctness without touching prior data on disk. "
             "Use --overwrite when you want a clean before/after audit "
             "trail (the backup is preserved, not deleted).",
    )
    p_run.add_argument(
        "--estimate", action="store_true",
        help="Print a cost + wall-clock estimate for this run (no API "
             "calls, no client constructed). Estimate uses tier-based "
             "per-call cost defaults and either empirical "
             "seconds-per-call from prior pilot manifests of the same "
             "model OR tier-default fallbacks. Override per-call cost "
             "with --cost-per-call.",
    )
    _add_guardrail_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # pilot
    p_pilot = sub.add_parser("pilot", help="Quick pilot run across all 7 conditions")
    p_pilot.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p_pilot.add_argument("--experiment", default="exp1a", choices=EXPERIMENT_CHOICES,
                         help="Experiment type (paper §3 alignment; default exp1a)")
    p_pilot.add_argument(
        "--provider", default="vllm", choices=["vllm", "openai", "anthropic"],
        help="Inference provider; see `affect-battery run --help`.",
    )
    p_pilot.add_argument("--base-url", default="http://localhost:8000/v1")
    p_pilot.add_argument("--num-runs", type=int, default=5,
                         help="Runs per condition (default: 5).")
    p_pilot.add_argument("--seed", type=int, default=42)
    p_pilot.add_argument("--temperature", type=float, default=0.7)
    p_pilot.add_argument("--output-dir", default="results")
    p_pilot.add_argument("--dry-run", action="store_true")
    p_pilot.add_argument("--base-model", action="store_true",
                         help="Use /v1/completions + few-shot scaffold (for non-instruct models).")
    p_pilot.add_argument("--bank", default=None,
                         help=f"Stimulus bank id (default: {DEFAULT_BANK_ID}). "
                              f"Must exist in configs/banks/<bank>.yaml.")
    p_pilot.add_argument(
        "--transfer-bank", default=None, dest="transfer_bank",
        help="Path to a transfer-question bank YAML "
             "(e.g. configs/banks/exp1a_factual_qa_hard_v1.yaml). "
             "When unset, the runner falls back to the hardcoded "
             "factual_qa pool — fine for legacy parity, but will "
             "ceiling out on frontier models.",
    )
    p_pilot.add_argument(
        "--runner-config", default=None,
        help="Per-experiment runner config YAML. Required for "
             "--experiment in {exp3a, exp3b, exp3c} (intensity_levels / "
             "prompts / items). Optional otherwise.",
    )
    p_pilot.add_argument(
        "--neutral-turns", type=int, default=0,
        help="Number of neutral conditioning turns between conditioning "
             "and transfer phases. Used by exp2 persistence-recovery; "
             "ignored by other experiments. Default 0.",
    )
    p_pilot.add_argument(
        "--overwrite", action="store_true",
        help="Move any existing pilot dir at --output-dir to "
             "<dir>.bak.<UTC-timestamp>/ before running. Default "
             "(no flag) is resume-by-default: the cache layer handles "
             "per-cell correctness without touching prior data on disk. "
             "Use --overwrite when you want a clean before/after audit "
             "trail (the backup is preserved, not deleted).",
    )
    p_pilot.add_argument(
        "--estimate", action="store_true",
        help="Print a cost + wall-clock estimate for this pilot (no API "
             "calls, no client constructed). Estimate uses tier-based "
             "per-call cost defaults and either empirical "
             "seconds-per-call from prior pilot manifests of the same "
             "model OR tier-default fallbacks. Override per-call cost "
             "with --cost-per-call.",
    )
    _add_prereg_gate_args(p_pilot)
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
        help="Variance probe for MDE grounding",
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
        help="Base-model feasibility probe",
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Silence the chatty per-request HTTP loggers from the OpenAI /
    # Anthropic SDKs (and httpx underneath). These emit one INFO line
    # per API call, which spams the terminal during a 35-run pilot.
    # WARNING level still surfaces 4xx/5xx + auth errors.
    for noisy in ("httpx", "anthropic", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


def cmd_probe_variance(args):
    """Variance probe: per-condition arithmetic accuracy + observed
    effect size, used to ground per-hypothesis MDEs."""
    from .models import VLLMClient, DryRunClient
    from .probes.variance import run_variance_probe

    client = (
        DryRunClient(model=args.model)
        if args.dry_run
        else VLLMClient(base_url=args.base_url, model=args.model)
    )

    async def _run():
        result = await run_variance_probe(
            client=client,
            model_name=args.model,
            n_per_condition=args.n,
            output_dir=Path(args.output_dir),
            base_url=args.base_url,
        )
        print(
            f"variance probe complete: variance_estimate="
            f"{result.variance_estimate:.4f}, std_err={result.std_err:.4f}"
        )
        if hasattr(client, "close"):
            await client.close()

    asyncio.run(_run())


def cmd_probe_base_model(args):
    """Base-model feasibility probe: GSM8K accuracy under NO_CONDITIONING
    few-shot scaffold; decides whether the base model goes in the
    primary H4 family."""
    from .models import VLLMCompletionClient, DryRunClient
    from .probes.base_model import run_base_model_probe

    client = (
        DryRunClient(model=args.model)
        if args.dry_run
        else VLLMCompletionClient(base_url=args.base_url, model=args.model)
    )

    async def _run():
        result = await run_base_model_probe(
            client=client,
            model_name=args.model,
            n=args.n,
            output_dir=Path(args.output_dir),
        )
        print(
            f"base-model probe complete: "
            f"baseline_accuracy={result.baseline_accuracy:.3f}, "
            f"verdict={result.feasibility_verdict}"
        )
        if hasattr(client, "close"):
            await client.close()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
