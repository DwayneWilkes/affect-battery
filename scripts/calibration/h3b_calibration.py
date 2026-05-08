"""H3b probe G: mini-calibration of GSM-Hard items to p̂ ≈ 0.5.

For each candidate item, sends the question with no stimulus context
n_reps times at temperature=0.7. Records per-item p̂ (fraction correct
across reps). Filters to items with p̂ ∈ [target_lo, target_hi].

Output: per-item p̂ and the calibrated subset (item_ids and questions)
that the follow-on concavity probe will use.

Environment:
    AFFECT_BATTERY_ROOT must point at the affect-battery repo checkout.
    The script imports `src.banks.loader`, `src.models`, and
    `src.scoring.accuracy` from there.

Usage:
    AFFECT_BATTERY_ROOT=/path/to/affect-battery \\
    direnv exec . uv run --active python scripts/h3b_calibration_robust.py \\
        --bank configs/banks/gsm8k_v1.yaml \\
        --provider openai --model gpt-5.4-nano \\
        --n-candidates 500 --n-reps 100 \\
        --target-lo 0.40 --target-hi 0.60 \\
        --output configs/h3b_calibration_2026-05-08.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

def _find_affect_battery_root() -> Path:
    """Locate the affect-battery repo for sys.path injection. Looks at:
      1. AFFECT_BATTERY_ROOT env var if set (portable across machines)
      2. CWD, if it looks like an affect-battery checkout (works under
         `cd /path/to/affect-battery && direnv exec . python ...`,
         which strips external env vars)
    Errors with a clear message if neither path resolves."""
    env_path = os.environ.get("AFFECT_BATTERY_ROOT", "").strip()
    if env_path:
        p = Path(env_path).resolve()
        if not p.is_dir():
            raise SystemExit(
                f"AFFECT_BATTERY_ROOT={p!s} is not a directory."
            )
        return p
    cwd = Path.cwd().resolve()
    sentinel = cwd / "src" / "banks" / "loader.py"
    if sentinel.is_file():
        return cwd
    raise SystemExit(
        "Cannot locate the affect-battery repo. Either set "
        "AFFECT_BATTERY_ROOT to your checkout, or run this script from "
        "the affect-battery repo root (cwd must contain src/banks/loader.py)."
    )


AFFECT_BATTERY_ROOT = _find_affect_battery_root()
sys.path.insert(0, str(AFFECT_BATTERY_ROOT))

from src.banks.loader import load_bank_items  # noqa: E402
from src.models import OpenAIClient, AnthropicClient, DryRunClient, NonRetryableAPIError  # noqa: E402
from src.scoring.accuracy import extract_numeric_answer  # noqa: E402
from src.lib.tracking import ExperimentTracker  # noqa: E402


def make_client(provider: str, model: str, dry_run: bool):
    if dry_run:
        return DryRunClient(model=f"{provider}-{model}-dryrun")
    if provider == "openai":
        return OpenAIClient(model=model)
    if provider == "anthropic":
        return AnthropicClient(model=model)
    raise ValueError(f"unsupported provider: {provider!r}")


async def run_cell(client, question: str, expected: str):
    messages = [{"role": "user", "content": question}]
    response = await client.complete(messages, temperature=0.7, max_tokens=512)
    extracted = extract_numeric_answer(response)
    correct = 0
    if extracted is not None:
        try:
            target = float(expected)
            correct = int(abs(extracted - target) < 0.01)
        except (TypeError, ValueError):
            pass
    return correct


async def run_one_candidate(client, item, n_reps: int, sem: asyncio.Semaphore):
    """Run all n_reps for a single candidate concurrently (gated by the
    shared semaphore so total in-flight calls stay bounded). Self-heals
    rate-limit failures: when OpenAIClient's retry budget is exhausted
    by a 429, the per-rep code sleeps outside the semaphore (so other
    workers continue and the rate limit can recover) and retries up to
    three more times. True non-retryable errors (moderation,
    invalid_prompt) and unrelated exceptions still get recorded but
    don't cascade out of the per-candidate `gather`."""
    async def one_rep():
        for attempt in range(4):  # initial + up to 3 patient retries
            async with sem:
                try:
                    return ("ok", await run_cell(
                        client, item["question"], item["expected"]
                    ))
                except NonRetryableAPIError as exc:
                    msg = str(exc)
                    if "rate-limit" in msg.lower() and attempt < 3:
                        # Fall through to back off outside the semaphore,
                        # then retry. Don't return yet.
                        retry_msg = msg
                    else:
                        return ("blocked", msg[:200])
                except Exception as exc:
                    return ("error", f"{type(exc).__name__}: {str(exc)[:180]}")
            # Outside semaphore: yield concurrency slot and sleep so the
            # rate limit window can reset before retry. Backoff grows
            # with attempts: 30s, 60s, 90s.
            await asyncio.sleep(30 * (attempt + 1))
        return ("blocked", f"rate_limit_after_4_attempts: {retry_msg[:140]}")

    results = await asyncio.gather(
        *[one_rep() for _ in range(n_reps)],
        return_exceptions=True,
    )
    # Coerce any unexpected exception that escaped one_rep into the
    # "error" kind so aggregation below is uniform.
    normalised = []
    for r in results:
        if isinstance(r, BaseException):
            normalised.append(("error", f"{type(r).__name__}: {str(r)[:180]}"))
        else:
            normalised.append(r)
    correct_count = sum(r[1] for r in normalised if r[0] == "ok")
    successful_reps = sum(1 for r in normalised if r[0] == "ok")
    blocked_msgs = [r[1] for r in normalised if r[0] == "blocked"]
    error_msgs = [r[1] for r in normalised if r[0] == "error"]
    if successful_reps == 0:
        return {
            "kind": "blocked",
            "item_id": item["id"],
            "question": item["question"],
            "reason": (blocked_msgs[0] if blocked_msgs
                       else error_msgs[0] if error_msgs
                       else "all_reps_failed"),
        }
    return {
        "kind": "scored",
        "item_id": item["id"],
        "question": item["question"],
        "expected": item["expected"],
        "n_reps": successful_reps,
        "n_correct": correct_count,
        "p_hat": correct_count / successful_reps,
        "n_blocked_reps": len(blocked_msgs),
        "n_error_reps": len(error_msgs),
    }


def _bank_fingerprint(bank_path: Path) -> str:
    """SHA-256 of the bank YAML file's bytes. Used to isolate the
    ExperimentTracker's output_dir per-input-bank: a different bank
    file (even with overlapping item IDs) gets its own tracker
    directory; the old data is preserved unchanged."""
    import hashlib
    return hashlib.sha256(bank_path.read_bytes()).hexdigest()


async def run_probe(args):
    items = load_bank_items(args.bank)
    if args.difficulty == "all":
        difficulty_filter = None
    else:
        accepted = {d.strip() for d in args.difficulty.split(",") if d.strip()}
        difficulty_filter = accepted
    if difficulty_filter is None:
        filtered = list(items)
    else:
        filtered = [it for it in items if it.get("difficulty") in difficulty_filter]
    selected = filtered[args.skip_first : args.skip_first + args.n_candidates]

    client = make_client(args.provider, args.model, dry_run=args.dry_run)

    # ExperimentTracker handles run metadata, per-item caching, stage
    # timing, and (optional) MLflow integration. Output dir is
    # bank-specific so different input banks get isolated state — old
    # caches are preserved when the bank changes, never overwritten.
    bank_fp = _bank_fingerprint(args.bank)
    tracker_root = (args.tracker_dir if args.tracker_dir
                    else args.output.with_suffix(args.output.suffix + ".tracker"))
    tracker_dir = tracker_root / f"bank_{bank_fp[:12]}"
    sibling_dirs = []
    if tracker_root.is_dir():
        sibling_dirs = [
            p for p in tracker_root.iterdir()
            if p.is_dir() and p.name.startswith("bank_") and p != tracker_dir
        ]
    if sibling_dirs:
        names = ", ".join(sorted(p.name for p in sibling_dirs))
        print(
            f"  note: {len(sibling_dirs)} other bank tracker(s) at "
            f"{tracker_root} ({names}); using {tracker_dir.name} for this "
            f"run (--bank {args.bank}). Old data is preserved.",
            file=sys.stderr,
        )

    tracker = ExperimentTracker(
        output_dir=tracker_dir,
        experiment_name=f"h3b_calibration_{args.output.stem}",
    )
    tracker.log_params(
        bank=str(args.bank),
        bank_sha256=bank_fp,
        provider=args.provider,
        model=args.model,
        n_candidates=args.n_candidates,
        n_reps=args.n_reps,
        skip_first=args.skip_first,
        target_lo=args.target_lo,
        target_hi=args.target_hi,
        max_concurrent=args.max_concurrent,
        candidates_per_batch=args.candidates_per_batch,
        temperature=0.7,
    )

    # Bounded concurrency:
    #   - call_sem caps in-flight API calls (per-rep) across the whole run
    #   - candidate batches cap how many candidates have any state alive
    #     at once. Without this, dispatching 500 candidates × 100 reps
    #     would create ~50,000 asyncio coroutines simultaneously, which
    #     can crash the python process at scale.
    call_sem = asyncio.Semaphore(args.max_concurrent)

    # Resume: items already in tracker's cache are skipped on dispatch
    # but counted in the final aggregation.
    to_dispatch = [it for it in selected if not tracker.is_cached(it["id"])]
    n_cached = len(selected) - len(to_dispatch)
    if n_cached:
        print(
            f"  resuming: {n_cached} of {len(selected)} candidates cached "
            f"at {tracker_dir / 'cache'}",
            file=sys.stderr,
        )

    async def run_and_cache(item):
        result = await run_one_candidate(client, item, args.n_reps, call_sem)
        payload = {
            "item_id": item["id"],
            "n_reps_target": args.n_reps,
            "difficulty": item.get("difficulty", "?"),
            **result,
        }
        tracker.log_item(item["id"], payload)
        return {**result, "_difficulty": item.get("difficulty", "?")}

    per_item: list[dict] = []
    blocked_items: list[dict] = []

    # Seed per_item / blocked_items with cached entries.
    for item in selected:
        if not tracker.is_cached(item["id"]):
            continue
        entry = tracker.load_cached(item["id"])
        if entry.get("n_reps_target") != args.n_reps:
            continue  # stale config; will be re-run
        if entry.get("kind") == "blocked":
            blocked_items.append({
                "item_id": entry["item_id"],
                "question": entry.get("question", ""),
                "reason": "non_retryable_api_error",
                "difficulty": entry.get("difficulty",
                                        item.get("difficulty", "?")),
            })
        else:
            rec = {
                k: entry[k]
                for k in ("item_id", "question", "expected",
                          "n_reps", "n_correct", "p_hat")
            }
            rec["difficulty"] = entry.get("difficulty",
                                          item.get("difficulty", "?"))
            per_item.append(rec)

    from tqdm.asyncio import tqdm as atqdm
    total_new = len(to_dispatch)
    batch_size = args.candidates_per_batch
    pbar = atqdm(
        total=total_new,
        desc=f"calibrating ({n_cached} cached, {total_new} new)",
        unit="cand",
        file=sys.stderr,
    )

    async def _drain_one_batch(batch_items):
        batch_tasks = [
            asyncio.create_task(run_and_cache(it)) for it in batch_items
        ]
        for fut in asyncio.as_completed(batch_tasks):
            yield await fut

    # tracker.stage gives us auto-recorded start/end/duration in the
    # run_metadata.json, so the calibration's wall-clock is captured
    # alongside the params and metrics.
    with tracker.stage("pre_screen"):
        for i in range(0, total_new, batch_size):
            batch = to_dispatch[i : i + batch_size]
            async for result in _drain_one_batch(batch):
                pbar.update(1)
                if result["kind"] == "blocked":
                    blocked_items.append({
                        "item_id": result["item_id"],
                        "question": result["question"],
                        "reason": "non_retryable_api_error",
                    })
                    pbar.write(
                        f"  {result['item_id']}: BLOCKED ({result['reason']})",
                        file=sys.stderr,
                    )
                    continue
                per_item.append({
                    k: result[k]
                    for k in ("item_id", "question", "expected",
                              "n_reps", "n_correct", "p_hat")
                })
                pbar.set_postfix_str(
                    f"last={result['item_id']} p̂={result['p_hat']:.2f}",
                    refresh=False,
                )
    pbar.close()

    # Reorder by original bank order (cache iteration + as_completed
    # both yield in non-bank order).
    bank_order = {it["id"]: idx for idx, it in enumerate(selected)}
    per_item.sort(key=lambda p: bank_order.get(p["item_id"], 10**9))

    calibrated = [
        p for p in per_item
        if args.target_lo <= p["p_hat"] <= args.target_hi
    ]
    calibrated.sort(key=lambda p: abs(p["p_hat"] - 0.5))

    # Aggregate API usage from the OpenAI client. Anthropic / dry-run
    # clients don't expose usage_summary; guard via getattr.
    usage = None
    if hasattr(client, "usage_summary"):
        kwargs = {}
        if (args.input_usd_per_million is not None
                and args.output_usd_per_million is not None):
            kwargs["input_usd_per_million"] = args.input_usd_per_million
            kwargs["output_usd_per_million"] = args.output_usd_per_million
        usage = client.usage_summary(**kwargs)

    metric_kwargs = dict(
        n_per_item=len(per_item),
        n_blocked=len(blocked_items),
        n_calibrated=len(calibrated),
        yield_pct=100.0 * len(calibrated) / max(len(selected), 1),
    )
    if usage:
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                metric_kwargs[f"usage_{k}"] = v
    tracker.log_metrics(**metric_kwargs)

    return {
        "model": args.model,
        "provider": args.provider,
        "temperature": 0.7,
        "n_candidates": args.n_candidates,
        "skip_first": args.skip_first,
        "n_reps": args.n_reps,
        "target_lo": args.target_lo,
        "target_hi": args.target_hi,
        "max_concurrent": args.max_concurrent,
        "per_item": per_item,
        "blocked_items": blocked_items,
        "n_blocked": len(blocked_items),
        "calibrated_subset": calibrated,
        "n_calibrated": len(calibrated),
        "interpretation": (
            "p̂ per item is the no-stimulus accuracy at temp=0.7 over "
            f"{args.n_reps} reps. SE ≈ sqrt(0.25/{args.n_reps}) "
            f"≈ {(0.25/args.n_reps)**0.5:.3f}. The calibrated_subset is "
            "the input pool for the follow-on concavity probe."
        ),
        "tracker_dir": str(tracker_dir),
        "usage": usage,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", required=True, type=Path)
    ap.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-candidates", type=int, default=50)
    ap.add_argument("--skip-first", type=int, default=0,
                    help="Skip the first N items in the filtered subset (resume support)")
    ap.add_argument("--difficulty", default="hard",
                    help="Filter bank by difficulty. 'hard' (default, backward "
                         "compatible), 'easy', 'medium', a comma-separated "
                         "list (e.g. 'easy,medium'), or 'all' to skip the filter.")
    ap.add_argument("--n-reps", type=int, default=50)
    ap.add_argument("--target-lo", type=float, default=0.40)
    ap.add_argument("--target-hi", type=float, default=0.60)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-concurrent", type=int, default=50,
                    help="Max in-flight API calls across all candidates × reps. "
                         "OpenAIClient handles 429 backoff and the per-rep code "
                         "self-heals from sustained rate-limit by sleeping and "
                         "retrying. Default 50 is a tested floor for gpt-5.4-nano; "
                         "raise cautiously and watch for 'BLOCKED' rate-limit lines.")
    ap.add_argument("--tracker-dir", type=Path, default=None,
                    help="ExperimentTracker output dir (default: "
                         "<output>.tracker/bank_<sha[:12]>/). Holds run "
                         "metadata, per-item cache, and stage timing. A "
                         "killed run resumes from cache on next invocation. "
                         "Different banks get separate subdirs (old data "
                         "preserved). Delete the dir to force a full re-run.")
    ap.add_argument("--candidates-per-batch", type=int, default=20,
                    help="Dispatch this many candidates concurrently per "
                         "batch. Each candidate spawns n_reps coroutines so "
                         "max coroutine count = batch × n_reps. Bigger = "
                         "more wall-clock parallelism but higher peak memory.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--input-usd-per-million", type=float, default=None,
                    help="Pricing for input tokens, USD per 1M. When set "
                         "with --output-usd-per-million, the run logs an "
                         "estimated_usd metric. Without these, only token "
                         "totals are logged. Pricing is operator-supplied "
                         "rather than hardcoded because OpenAI changes it.")
    ap.add_argument("--output-usd-per-million", type=float, default=None,
                    help="Pricing for output tokens (includes reasoning "
                         "tokens, which bill at the output rate), USD "
                         "per 1M. See --input-usd-per-million.")
    args = ap.parse_args()

    result = asyncio.run(run_probe(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    print(f"\nMini-calibration written to {args.output}", file=sys.stderr)
    print(
        f"  {result['n_calibrated']} of {args.n_candidates} items in "
        f"[{args.target_lo:.2f}, {args.target_hi:.2f}]",
        file=sys.stderr,
    )
    if result["calibrated_subset"]:
        print("  closest to p̂=0.5:", file=sys.stderr)
        for p in result["calibrated_subset"][:5]:
            print(f"    {p['item_id']}: p̂={p['p_hat']:.2f}", file=sys.stderr)
    if result.get("usage"):
        u = result["usage"]
        print(
            f"\n  API usage ({u.get('n_calls', 0)} calls): "
            f"prompt={u.get('prompt_tokens', 0):,}  "
            f"completion={u.get('completion_tokens', 0):,}  "
            f"reasoning={u.get('reasoning_tokens', 0):,}",
            file=sys.stderr,
        )
        if "estimated_usd" in u:
            print(f"  estimated cost: ${u['estimated_usd']:.4f}",
                  file=sys.stderr)
        else:
            print(
                "  cost: not estimated (pass --input-usd-per-million and "
                "--output-usd-per-million to enable)",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
