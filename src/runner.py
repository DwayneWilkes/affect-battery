"""Experiment runner for the Affect Battery."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from .conditioning.prompts import Condition, FEEDBACK_SETS
from .conditioning.protocol import ConditioningProtocol, Message, build_transfer_messages
from .conditioning.tasks import get_arithmetic_problems, get_transfer_tasks, MathProblem
from .models import ModelClient, NonRetryableAPIError  # noqa: F401 -- public API
from .scoring.accuracy import extract_numeric_answer, score_factual_qa
from .util import CHECKSUM_KEY, checksum_of_payload, enum_value, model_slug

log = logging.getLogger(__name__)


# Spec Requirement (conditioning-protocol::Transfer phase message format): this
# exact string MUST be the system prompt for cross-session transfer, identical
# across all conditions. Keep as a module-level constant so any drift is visible.
CROSS_SESSION_SYSTEM_PROMPT = "Answer the questions to the best of your ability."


class ExperimentType(str, Enum):
    # Values updated per design.md D6: use paper-section-aligned identifiers
    # (exp1a/exp1b/exp2/exp3a/exp3b/exp3c) as the canonical discriminator.
    # Enum names preserved for backward-compat code references.
    TRANSFER_WITHIN = "exp1a"
    TRANSFER_CROSS = "exp1b"
    PERSISTENCE = "exp2"
    AROUSAL_PERFORMANCE = "exp3a"
    COGNITIVE_SCOPE = "exp3b"
    CONSERVATIVE_SHIFT = "exp3c"


@dataclass
class ExperimentConfig:
    model_name: str
    condition: Condition
    experiment_type: ExperimentType
    num_runs: int = 50
    temperature: float = 0.7
    transfer_task: str = "factual_qa"
    num_conditioning_turns: int = 5
    num_transfer_questions: int = 5
    seed: int | None = None
    # Persistence-specific
    neutral_turns: int = 0
    # Base-model (non-instruct) inference path: when True, runner builds a
    # few-shot scaffold via build_base_model_prompt and calls /v1/completions
    # via VLLMCompletionClient.complete_text instead of the chat API.
    is_base_model: bool = False
    # Stimulus bank identity: bank_id names the bank, stimulus_bank_hash is
    # the SHA-256 over the canonicalized items list (computed by the bank
    # loader). The hash participates in cache-identity so mid-curation bank
    # edits invalidate cached results even when the bank_id is unchanged.
    # stimulus_bank_hash defaults to '' so the runner doesn't need to load
    # a bank itself; the CLI populates the hash from the loaded bank.
    stimulus_bank: str = "arithmetic_easy_v1"
    stimulus_bank_hash: str = ""
    # Optional: path to a transfer-question bank YAML. When set, runner
    # loads transfer questions from the bank (with answer_aliases) instead
    # of the hardcoded factual_qa pool. Used to swap in TriviaQA hard subset
    # for Tier-1 ceiling-break on frontier models.
    transfer_bank: str = ""
    # SHA-256 over the transfer-bank file content. Participates in cache
    # identity so re-piloting with a different transfer bank invalidates
    # cached results (which would otherwise silently return stale data
    # from the prior bank — a subtle correctness bug we hit when adding
    # the alias-aware bank loader).
    transfer_bank_hash: str = ""


# ---------------------------------------------------------------------------
# Discriminated-union result schema (design.md D6)
# ---------------------------------------------------------------------------
# Base RunResult carries experiment-agnostic fields (config, model, condition,
# experiment_type, run_number, start_time, end_time, checksum). The `body`
# field is an experiment-specific dataclass keyed on experiment_type. Cross-
# experiment analyses read base fields + match on experiment_type to unwrap
# the body; per-experiment analyses read result.body directly.

_VALID_EXPERIMENT_TYPES = frozenset(
    ["exp1a", "exp1b", "exp2", "exp3a", "exp3b", "exp3c"]
)


@dataclass
class Exp1aBody:
    """Within-session transfer (paper §3.2.1 Exp 1a)."""
    conditioning_responses: list[str]
    conditioning_correct: list[bool]
    transfer_responses: list[str]
    transfer_questions: list[str]
    transfer_expected: list[str]
    transfer_correct: list[float] = field(default_factory=list)
    transfer_expected_aliases: list[list[str]] = field(default_factory=list)


@dataclass
class Exp1bBody:
    """Cross-session falsification (paper §3.2.2 Exp 1b). Phase 1 session-1
    conditioning + transfer + Phase 2 session-2 fresh-context re-test.
    session_1_seed / session_2_seed are recorded separately per spec."""
    conditioning_responses: list[str]
    conditioning_correct: list[bool]
    transfer_responses: list[str]
    transfer_questions: list[str]
    transfer_expected: list[str]
    transfer_correct: list[float] = field(default_factory=list)
    transfer_expected_aliases: list[list[str]] = field(default_factory=list)
    session_1_seed: int = 0
    session_2_seed: int = 0


@dataclass
class Exp2Body:
    """Persistence / recovery dynamics (paper §3.3 Exp 2). N-value specifies
    number of neutral turns between conditioning and recovery measurement.
    turn_accuracies is per-turn accuracy across the N recovery turns."""
    n_value: int
    turn_accuracies: list[float]


@dataclass
class Exp3aBody:
    """Nonlinear arousal-performance (paper §3.4.1 Exp 3a). intensity_level is
    the pre-registered 1..7 level from the primary_valence_axis."""
    intensity_level: int
    transfer_responses: list[str]
    transfer_expected: list[str]


@dataclass
class Exp3bBody:
    """Cognitive scope (paper §3.4.2 Exp 3b). 10 generations per prompt per
    condition for semantic-diversity computation."""
    prompt_id: str
    generations: list[str]
    per_generation_seeds: list[int]


@dataclass
class Exp3cBody:
    """Conservative shift (paper §3.4.3 Exp 3c). Factual QA with difficulty
    stratification + confidence elicitation + refusal tracking.

    `expected_aliases` carries the bank's per-item alias list so the
    analyzer can match 'U.S.' to 'United States' the way exp1a does.
    Missing/empty aliases reduce to substring match against `expected`.
    """
    difficulty: str
    question: str
    response: str
    expected: str
    stated_confidence: int | None = None
    refused: bool = False
    expected_aliases: list[str] = field(default_factory=list)


# Union of all body types (for type annotations / analysis dispatch)
ExperimentBody = Exp1aBody | Exp1bBody | Exp2Body | Exp3aBody | Exp3bBody | Exp3cBody


@dataclass
class RunResult:
    config: dict
    run_number: int
    # Base fields promoted per D6: experiment_type is the discriminator;
    # model + condition are top-level for cross-experiment analyses that
    # read them without unwrapping.
    experiment_type: str = "exp1a"
    model: str = ""
    condition: str = ""
    # Legacy Exp 1a fields retained at top level for backward compatibility
    # during the D6 migration. New code SHOULD read result.body.<field>;
    # legacy callsites keep working via __post_init__ sync (see below).
    conditioning_responses: list[str] = field(default_factory=list)
    conditioning_correct: list[bool] = field(default_factory=list)
    transfer_responses: list[str] = field(default_factory=list)
    transfer_questions: list[str] = field(default_factory=list)
    transfer_expected: list[str] = field(default_factory=list)
    transfer_correct: list[float] = field(default_factory=list)
    transfer_expected_aliases: list[list[str]] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    checksum: str = ""
    body: ExperimentBody | None = None

    def __post_init__(self) -> None:
        # Reject unknown experiment_type values (per D6 Literal contract).
        if self.experiment_type not in _VALID_EXPERIMENT_TYPES:
            raise ValueError(
                f"Invalid experiment_type {self.experiment_type!r}; "
                f"must be one of {sorted(_VALID_EXPERIMENT_TYPES)}"
            )
        # Bi-directional sync for Exp 1a during migration: if caller passed
        # legacy top-level conditioning/transfer fields WITHOUT a body,
        # synthesize an Exp1aBody. If caller passed body but not top-level
        # fields, mirror body fields to top-level for legacy reads.
        # Defensively copy lists when synthesizing a body or mirroring
        # body->top-level so mutations on top-level fields don't alias
        # the body. Reject inputs where caller passes BOTH a body and
        # legacy top-level fields whose values disagree.
        if self.body is None and self.experiment_type == "exp1a":
            self.body = Exp1aBody(
                conditioning_responses=list(self.conditioning_responses),
                conditioning_correct=list(self.conditioning_correct),
                transfer_responses=list(self.transfer_responses),
                transfer_questions=list(self.transfer_questions),
                transfer_expected=list(self.transfer_expected),
                transfer_correct=list(self.transfer_correct),
                transfer_expected_aliases=[list(a) for a in self.transfer_expected_aliases],
            )
        elif self.body is None and self.experiment_type == "exp1b":
            # Phase 2 fresh-session re-test: top-level conditioning/transfer
            # fields carry session-1 conditioning + session-2 transfer (per
            # the cross-session branch in run_single). Seeds default to 0 and
            # are populated by run_single when config is TRANSFER_CROSS.
            self.body = Exp1bBody(
                conditioning_responses=list(self.conditioning_responses),
                conditioning_correct=list(self.conditioning_correct),
                transfer_responses=list(self.transfer_responses),
                transfer_questions=list(self.transfer_questions),
                transfer_expected=list(self.transfer_expected),
                transfer_correct=list(self.transfer_correct),
                transfer_expected_aliases=[list(a) for a in self.transfer_expected_aliases],
            )
        elif isinstance(self.body, Exp1aBody):
            # Body was passed; sync top-level for legacy readers, but reject
            # inconsistent input rather than silently coercing.
            for field_name in (
                "conditioning_responses", "conditioning_correct",
                "transfer_responses", "transfer_questions",
                "transfer_expected", "transfer_correct",
                "transfer_expected_aliases",
            ):
                top = getattr(self, field_name)
                body_v = getattr(self.body, field_name)
                if top and list(top) != list(body_v):
                    raise ValueError(
                        f"RunResult.{field_name} provided at top-level AND "
                        f"on body, but values differ. Pass one or the other."
                    )
                if not top:
                    setattr(self, field_name, list(body_v))

    def compute_checksum(self) -> str:
        self.checksum = checksum_of_payload(asdict(self))
        return self.checksum


_BASE_STOP = ["Human:", "\n\nHuman:"]


async def _run_single_base(
    config: ExperimentConfig,
    client: ModelClient,
    run_number: int,
) -> RunResult:
    """Base-model inference path: few-shot scaffold + /v1/completions.

    The scaffold is built incrementally between turns so every model call
    sees the full conditioning history up to that point.
    """
    from .conditioning.protocol import build_base_model_prompt

    start = time.time()
    seed = (config.seed or 0) + run_number
    problems = get_arithmetic_problems(config.num_conditioning_turns, seed=seed)
    transfer_qs = get_transfer_tasks(
        config.transfer_task, config.num_transfer_questions, seed=seed,
        bank_path=config.transfer_bank or None,
    )

    protocol = ConditioningProtocol(
        condition=config.condition,
        num_conditioning_turns=config.num_conditioning_turns,
    )

    # Prime: few-shot examples + conversation header, but NO per-turn question
    # lines yet (build_base_model_prompt appends them, but we want to drive
    # turn-by-turn so each call sees the accumulated history).
    scaffold = build_base_model_prompt(protocol, [], [])

    conditioning_responses: list[str] = []
    conditioning_correct: list[bool] = []

    if config.condition != Condition.NO_CONDITIONING:
        feedback_set = FEEDBACK_SETS[config.condition]
        max_turns = min(len(problems), len(feedback_set.turns))
        for i in range(max_turns):
            problem = problems[i]
            scaffold += f"\nHuman: {problem.question}\nAssistant:"
            response = await client.complete_text(  # type: ignore[attr-defined]
                scaffold, temperature=config.temperature, stop=_BASE_STOP,
            )
            conditioning_responses.append(response)
            extracted = extract_numeric_answer(response)
            is_correct = (
                extracted is not None
                and abs(extracted - problem.answer) < 0.01
            )
            conditioning_correct.append(is_correct)
            turn = feedback_set.turns[i]
            feedback = turn.correct if is_correct else turn.incorrect
            scaffold += f"{response}\nHuman: {feedback}"

    # Transfer phase: within-session keeps the conditioning scaffold.
    # Cross-session: reset to a neutral priming.
    if config.experiment_type == ExperimentType.TRANSFER_CROSS:
        scaffold = (
            CROSS_SESSION_SYSTEM_PROMPT
            + "\n\n### Conversation:"
        )

    transfer_responses: list[str] = []
    for q in transfer_qs:
        scaffold += f"\nHuman: {q.question}\nAssistant:"
        response = await client.complete_text(  # type: ignore[attr-defined]
            scaffold, temperature=config.temperature, stop=_BASE_STOP,
        )
        transfer_responses.append(response)
        scaffold += f"{response}"

    transfer_expected = [q.expected_answer for q in transfer_qs]
    transfer_aliases = [list(q.expected_aliases) for q in transfer_qs]
    transfer_correct = [
        score_factual_qa(r, e, aliases=a)
        for r, e, a in zip(transfer_responses, transfer_expected, transfer_aliases)
    ]
    cfg_dict = asdict(config)
    result = RunResult(
        config=cfg_dict,
        run_number=run_number,
        conditioning_responses=conditioning_responses,
        conditioning_correct=conditioning_correct,
        transfer_responses=transfer_responses,
        transfer_questions=[q.question for q in transfer_qs],
        transfer_expected=transfer_expected,
        transfer_correct=transfer_correct,
        transfer_expected_aliases=transfer_aliases,
        start_time=start,
        end_time=time.time(),
    )
    result.compute_checksum()
    return result


async def run_conditioning_phase(
    config: ExperimentConfig,
    client: ModelClient,
    seed: int,
) -> tuple[list[dict], list[str], list[bool]]:
    """Run the 5-turn affective-conditioning phase and return
    (messages, conditioning_responses, conditioning_correct).

    Shared by run_single (Exp 1a/1b) and the Exp 3b/3c runners so the
    affective conditioning protocol is identical across experiments.
    Condition-stratified metrics in 3b/3c require this protocol to fire
    before the generation/QA phase.
    """
    problems = get_arithmetic_problems(config.num_conditioning_turns, seed=seed)
    protocol = ConditioningProtocol(
        condition=config.condition,
        num_conditioning_turns=config.num_conditioning_turns,
    )
    conditioning_responses: list[str] = []
    conditioning_correct: list[bool] = []
    messages: list[dict] = [{"role": "system", "content": protocol.system_prompt}]

    if config.condition != Condition.NO_CONDITIONING:
        feedback_set = FEEDBACK_SETS[config.condition]
        max_turns = min(len(problems), len(feedback_set.turns))
        for i in range(max_turns):
            problem = problems[i]
            messages.append({"role": "user", "content": problem.question})
            # Conditioning answers are short numerics — 50 tokens caps
            # the model's verbosity without truncating valid answers,
            # cutting output cost on tiers where the model would
            # otherwise emit prose ("The answer is 47, computed as...").
            response = await client.complete(
                messages, temperature=config.temperature, max_tokens=50,
            )
            messages.append({"role": "assistant", "content": response})
            conditioning_responses.append(response)

            extracted = extract_numeric_answer(response)
            is_correct = extracted is not None and abs(extracted - problem.answer) < 0.01
            conditioning_correct.append(is_correct)

            turn = feedback_set.turns[i]
            feedback = turn.correct if is_correct else turn.incorrect
            messages.append({"role": "user", "content": feedback})

    return messages, conditioning_responses, conditioning_correct


async def run_single(
    config: ExperimentConfig,
    client: ModelClient,
    run_number: int,
) -> RunResult:
    """Run a single conditioning + transfer experiment."""
    if config.is_base_model:
        return await _run_single_base(config, client, run_number)
    start = time.time()

    seed = (config.seed or 0) + run_number
    transfer_qs = get_transfer_tasks(
        config.transfer_task, config.num_transfer_questions, seed=seed,
        bank_path=config.transfer_bank or None,
    )

    # Phase 1: Conditioning (extracted to a shared helper for Exp 3b/3c).
    messages, conditioning_responses, conditioning_correct = await run_conditioning_phase(
        config, client, seed
    )

    # Phase 1.5: Neutral buffer turns (for persistence experiments).
    # Per-turn accuracy is captured for Exp 2's recovery-curve analysis.
    neutral_turn_accuracies: list[float] = []
    for _ in range(config.neutral_turns):
        buffer_problems = get_arithmetic_problems(1, seed=seed + 1000 + _)
        messages.append({"role": "user", "content": buffer_problems[0].question})
        # Neutral buffer turns are arithmetic; 50 tokens is enough.
        response = await client.complete(
            messages, temperature=config.temperature, max_tokens=50,
        )
        messages.append({"role": "assistant", "content": response})
        extracted = extract_numeric_answer(response)
        is_correct = extracted is not None and abs(extracted - buffer_problems[0].answer) < 0.01
        neutral_turn_accuracies.append(1.0 if is_correct else 0.0)
    
    # Phase 2: Transfer
    if config.experiment_type == ExperimentType.TRANSFER_CROSS:
        messages = [{"role": "system", "content": CROSS_SESSION_SYSTEM_PROMPT}]
    
    transfer_responses = []
    for q in transfer_qs:
        messages.append({"role": "user", "content": q.question})
        # Transfer answers may include hedging prose (exp3c-style
        # responses can run 100-200 tokens); 256 caps the worst case
        # without truncating reasonable hedge-rich answers.
        response = await client.complete(
            messages, temperature=config.temperature, max_tokens=256,
        )
        messages.append({"role": "assistant", "content": response})
        transfer_responses.append(response)

    transfer_expected = [q.expected_answer for q in transfer_qs]
    transfer_aliases = [list(q.expected_aliases) for q in transfer_qs]
    transfer_correct = [
        score_factual_qa(r, e, aliases=a)
        for r, e, a in zip(transfer_responses, transfer_expected, transfer_aliases)
    ]
    result = RunResult(
        config=asdict(config),
        run_number=run_number,
        experiment_type=config.experiment_type.value,
        model=config.model_name,
        condition=config.condition.value,
        conditioning_responses=conditioning_responses,
        conditioning_correct=conditioning_correct,
        transfer_responses=transfer_responses,
        transfer_questions=[q.question for q in transfer_qs],
        transfer_expected=transfer_expected,
        transfer_correct=transfer_correct,
        transfer_expected_aliases=transfer_aliases,
        start_time=start,
        end_time=time.time(),
    )
    # For PERSISTENCE (Exp 2), attach an Exp2Body recording per-turn
    # accuracy across the neutral_turns recovery phase + the n_value
    # sweep step. Done before checksum so the body participates in
    # reproducibility hashing.
    if config.experiment_type == ExperimentType.PERSISTENCE:
        result.body = Exp2Body(
            n_value=config.neutral_turns,
            turn_accuracies=neutral_turn_accuracies,
        )
    # For TRANSFER_CROSS, record session_1_seed (conditioning phase) and
    # session_2_seed (fresh-session re-test) on the Exp1bBody. Phase 2 uses
    # `seed + 1`-derived offset so the two sessions sample distinct draws
    # while remaining deterministic.
    if config.experiment_type == ExperimentType.TRANSFER_CROSS and isinstance(result.body, Exp1bBody):
        result.body.session_1_seed = seed
        result.body.session_2_seed = seed + 10_000
    result.compute_checksum()
    return result


class _TokenBucket:
    """Simple asyncio token bucket.

    tokens refill at `rate_per_second` continuously; at most `capacity`
    tokens accumulate. acquire() returns immediately if a token is
    available, otherwise awaits until one is.
    """

    def __init__(self, rate_per_second: float, capacity: int | None = None):
        self._rate = float(rate_per_second)
        self._capacity = float(capacity) if capacity is not None else self._rate
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, n: int = 1) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                needed = n - self._tokens
                wait = needed / self._rate
            await asyncio.sleep(wait)


@dataclass
class BatchBudget:
    """Optional cap on API call count for a single run_batch invocation.

    cost_per_call is only used for pre-flight cost estimates; the budget
    itself is counted in API calls, not dollars.
    """
    max_api_calls: int | None = None
    cost_per_call: float | None = None


class BatchBudgetExceeded(Exception):
    """Raised by the budget-aware client proxy when a call would push the
    API call counter past max_api_calls. Caught by run_batch to halt
    cleanly and emit batch_budget_exceeded."""


class _BudgetedClient(ModelClient):
    """Proxy around a ModelClient that counts calls, enforces BatchBudget,
    and optionally throttles via a token bucket."""

    def __init__(
        self,
        wrapped: ModelClient,
        budget: BatchBudget | None,
        rate_limiter: _TokenBucket | None = None,
    ):
        self._wrapped = wrapped
        self._budget = budget
        self._rate_limiter = rate_limiter
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._wrapped.model_name

    async def _check_and_gate(self) -> None:
        if (self._budget is not None
                and self._budget.max_api_calls is not None
                and self.call_count >= self._budget.max_api_calls):
            raise BatchBudgetExceeded(
                f"max_api_calls={self._budget.max_api_calls} reached"
            )
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        self.call_count += 1

    async def complete(self, messages, temperature=0.7, max_tokens=1024):
        await self._check_and_gate()
        return await self._wrapped.complete(messages, temperature, max_tokens)

    async def complete_text(self, prompt, temperature=0.7, max_tokens=1024, stop=None):
        """Proxy for base-model path. Applies the same budget + rate-limit
        gates as complete()."""
        await self._check_and_gate()
        return await self._wrapped.complete_text(# type: ignore[attr-defined]
            prompt, temperature=temperature, max_tokens=max_tokens, stop=stop,
        )


class EventEmitter:
    """Append-only JSONL event sink.

    Each event is one line of JSON with a UTC ISO-8601 timestamp, an
    `event` name, and type-specific fields. The file is opened per-write
    to keep semantics simple (batches emit ~hundreds of events, not
    millions, so per-write open is cheap).
    """

    def __init__(self, path: Path):
        self._path = Path(path)

    def emit(self, event: str, **fields: Any) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        with self._path.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")


def _cached_run_path(output_dir: Path, config: ExperimentConfig, run_number: int) -> Path:
    """Derive the path save_result writes for a (config, run_number).

    Layout: <output_dir>/<condition>/[<n_prefix>_]<NNNN>.json. The
    n_prefix is `n<neutral_turns>_` for exp2's N-sweep so different
    neutral_turns runs at the same (condition, run_number) don't
    collide on disk. Other experiments (neutral_turns=0) get the
    plain `<NNNN>.json` filename.

    Other matrix axes are encoded by the caller's directory hierarchy:
    model lives in the pilot directory name (and manifest), experiment
    type lives in the parent of `output_dir`. This puts each cell of
    the experimental matrix in its own leaf directory so per-condition
    operations (`ls`, `du`, `wc`, `jq`) are natural.
    """
    cond = enum_value(config.condition)
    n = getattr(config, "neutral_turns", 0) or 0
    if n > 0:
        return Path(output_dir) / cond / f"n{n}_{run_number:04d}.json"
    return Path(output_dir) / cond / f"{run_number:04d}.json"


def is_valid_cached_result(
    path: Path,
    expected_stimulus_bank_hash: str | None = None,
    expected_transfer_bank_hash: str | None = None,
) -> bool:
    """Return True if `path` is a schema-valid, checksum-valid result file.

    Optional bank-hash arguments participate in cache identity:
      - `expected_stimulus_bank_hash` rejects the cache if the cached
        result was produced under a different conditioning (arithmetic)
        bank.
      - `expected_transfer_bank_hash` rejects the cache if the cached
        result was produced under a different transfer-question bank
        (e.g. legacy hardcoded pool vs. TriviaQA hard).

    Omitting either argument skips that check (back-compat for callers
    that don't track the corresponding bank).
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    for key in ("config", "run_number", "conditioning_responses",
                "conditioning_correct", "transfer_responses",
                "transfer_questions", "transfer_expected",
                "start_time", "end_time", CHECKSUM_KEY):
        if key not in data:
            return False
    stored = data.get(CHECKSUM_KEY, "")
    if not stored:
        return False
    if checksum_of_payload(data) != stored:
        return False
    if expected_stimulus_bank_hash is not None:
        cached_hash = data.get("config", {}).get("stimulus_bank_hash", "")
        if cached_hash != expected_stimulus_bank_hash:
            return False
    if expected_transfer_bank_hash is not None:
        cached_hash = data.get("config", {}).get("transfer_bank_hash", "")
        if cached_hash != expected_transfer_bank_hash:
            return False
    return True


async def run_batch(
    config: ExperimentConfig,
    client: ModelClient,
    max_concurrent: int = 5,
    output_dir: Path | None = None,
    circuit_breaker_threshold: int = 5,
    budget: BatchBudget | None = None,
    rate_limit_rps: float | None = None,
    cancel_event: asyncio.Event | None = None,
) -> AsyncIterator[RunResult]:
    """Run a batch of experiments with concurrency control.

    When `output_dir` is provided:
    - Existing valid result files are loaded and yielded without calling the API.
    - New results are saved as they complete.
    - Lifecycle events (run_started, run_completed, run_skipped_cached,
      run_failed, batch_circuit_open) are appended to `output_dir/events.jsonl`.

    The circuit breaker halts dispatching after `circuit_breaker_threshold`
    consecutive NonRetryableAPIError failures. A successful run or a cached
    skip resets the counter.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    emitter: EventEmitter | None = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        emitter = EventEmitter(output_dir / "events.jsonl")

    cached_runs: list[int] = []
    missing_runs: list[int] = []
    for i in range(config.num_runs):
        if output_dir is not None:
            path = _cached_run_path(output_dir, config, i)
            if is_valid_cached_result(
                path,
                expected_transfer_bank_hash=config.transfer_bank_hash or None,
            ):
                cached_runs.append(i)
                continue
        missing_runs.append(i)

    # Pre-flight event: fires once per run_batch before any API call.
    if emitter:
        calls_per_run = config.num_conditioning_turns + config.num_transfer_questions
        expected_calls = len(missing_runs) * calls_per_run
        preflight: dict[str, Any] = {
            "runs_total": config.num_runs,
            "runs_cached": len(cached_runs),
            "runs_to_execute": len(missing_runs),
            "expected_api_calls": expected_calls,
        }
        if budget is not None and budget.cost_per_call is not None:
            preflight["estimated_cost_usd"] = expected_calls * budget.cost_per_call
        emitter.emit("batch_preflight", **preflight)
        cost_str = (
            f" est ${preflight['estimated_cost_usd']:.2f}"
            if "estimated_cost_usd" in preflight else ""
        )
        log.info(
            "[preflight] %d runs total, %d cached, %d to execute (%d API calls%s)",
            preflight["runs_total"], preflight["runs_cached"],
            preflight["runs_to_execute"], preflight["expected_api_calls"],
            cost_str,
        )

    # Wrap the client so call-level budget enforcement + rate limiting work
    # regardless of which ModelClient subclass the caller passed in.
    rate_limiter = _TokenBucket(rate_limit_rps) if rate_limit_rps else None
    budgeted_client = _BudgetedClient(client, budget, rate_limiter)

    consecutive_failures = 0
    circuit_open = False
    budget_exceeded = False

    if output_dir is not None:
        for run_num in cached_runs:
            path = _cached_run_path(output_dir, config, run_num)
            data = load_result(path)
            run_name = path.stem
            if emitter:
                emitter.emit("run_skipped_cached", run_name=run_name,
                             path=str(path))
            result = RunResult(**{k: v for k, v in data.items() if k != CHECKSUM_KEY})
            result.checksum = data.get(CHECKSUM_KEY, "")
            yield result

    async def run_with_semaphore(run_num: int) -> RunResult | None:
        run_name = None
        if output_dir is not None:
            run_name = _cached_run_path(output_dir, config, run_num).stem
        if emitter:
            emitter.emit("run_started", run_name=run_name, run_number=run_num)
        async with semaphore:
            log.info(f"Run {run_num}/{config.num_runs} | {config.condition.value} | {config.model_name}")
            started = time.time()
            try:
                result = await run_single(config, budgeted_client, run_num)
            except NonRetryableAPIError as e:
                if emitter:
                    emitter.emit(
                        "run_failed", run_name=run_name, run_number=run_num,
                        error=str(e), error_class="NonRetryableAPIError",
                        status_code=e.status_code,
                    )
                raise
        if output_dir is not None:
            save_result(result, output_dir)
        if emitter:
            emitter.emit("run_completed", run_name=run_name,
                         run_number=run_num, elapsed_s=time.time() - started)
        return result

    # asyncio.wait with FIRST_COMPLETED so early break on cancel/circuit/budget
    # doesn't leave wrapper coroutines pending (as_completed would).
    tasks = [asyncio.create_task(run_with_semaphore(i)) for i in missing_runs]
    shutdown_signalled = False
    pending: set[asyncio.Task] = set(tasks)

    def _cancel_pending() -> None:
        for tt in pending:
            tt.cancel()

    def _signal_shutdown() -> None:
        nonlocal shutdown_signalled
        if shutdown_signalled:
            return
        shutdown_signalled = True
        if emitter:
            emitter.emit(
                "batch_shutdown_signal",
                runs_completed=(len(cached_runs) + budgeted_client.call_count),
            )
        _cancel_pending()

    try:
        while pending:
            if cancel_event is not None and cancel_event.is_set():
                _signal_shutdown()
                break
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            should_stop = False
            for t in done:
                try:
                    result = t.result()
                except NonRetryableAPIError as e:
                    consecutive_failures += 1
                    if consecutive_failures >= circuit_breaker_threshold and not circuit_open:
                        circuit_open = True
                        if emitter:
                            emitter.emit(
                                "batch_circuit_open",
                                consecutive_failures=consecutive_failures,
                                threshold=circuit_breaker_threshold,
                                reason=str(e),
                                error_class="NonRetryableAPIError",
                            )
                        _cancel_pending()
                        should_stop = True
                        break
                    continue
                except BatchBudgetExceeded as e:
                    if not budget_exceeded:
                        budget_exceeded = True
                        if emitter:
                            emitter.emit(
                                "batch_budget_exceeded",
                                reason=str(e),
                                max_api_calls=budget.max_api_calls if budget else None,
                                api_calls_made=budgeted_client.call_count,
                            )
                        _cancel_pending()
                    should_stop = True
                    break
                except asyncio.CancelledError:
                    continue
                consecutive_failures = 0
                if result is not None:
                    yield result
                if cancel_event is not None and cancel_event.is_set():
                    _signal_shutdown()
                    should_stop = True
                    break
            if should_stop:
                break
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except (NonRetryableAPIError, BatchBudgetExceeded, asyncio.CancelledError):
                pass
        if emitter:
            emitter.emit(
                "batch_completed",
                runs_yielded_cached=len(cached_runs),
                runs_executed=budgeted_client.call_count > 0,
                api_calls_made=budgeted_client.call_count,
                circuit_open=circuit_open,
                budget_exceeded=budget_exceeded,
            )


_REQUIRED_CONFIG_KEYS = ("model_name", "condition", "experiment_type")


def _validate_result(result: RunResult) -> None:
    """Raise ValueError if the RunResult is missing required fields or has
    mismatched array lengths. Called from save_result before writing."""
    cfg = result.config or {}
    for key in _REQUIRED_CONFIG_KEYS:
        if not cfg.get(key):
            raise ValueError(
                f"save_result: result.config missing required key '{key}'"
            )
    tq = len(result.transfer_questions)
    tr = len(result.transfer_responses)
    te = len(result.transfer_expected)
    if not (tq == tr == te):
        raise ValueError(
            f"save_result: transfer arrays have mismatched length "
            f"(questions={tq}, responses={tr}, expected={te})"
        )


def save_result(result: RunResult, output_dir: Path):
    """Validate and write a single result JSON.

    Layout: <output_dir>/<condition>/[n<N>_]<NNNN>.json. The n<N>
    prefix is included when config.neutral_turns > 0 so exp2's
    multi-N sweep can coexist in the same condition dir without
    collision.
    """
    _validate_result(result)
    cfg = result.config
    cond = enum_value(cfg["condition"])
    cond_dir = output_dir / cond
    cond_dir.mkdir(parents=True, exist_ok=True)
    n_turns = cfg.get("neutral_turns", 0) or 0
    if n_turns > 0:
        filename = f"n{n_turns}_{result.run_number:04d}.json"
    else:
        filename = f"{result.run_number:04d}.json"
    path = cond_dir / filename
    payload = asdict(result)
    payload["config"] = {k: enum_value(v) for k, v in payload["config"].items()}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_result(path: Path, verify: bool = False) -> dict:
    """Load a saved result JSON. Pass verify=True to recompute the checksum
    and log a warning on mismatch (audit / tamper detection path)."""
    data = json.loads(Path(path).read_text())
    if verify:
        stored = data.get(CHECKSUM_KEY, "")
        recomputed = checksum_of_payload(data)
        if stored and recomputed != stored:
            log.warning(
                "Checksum mismatch in %s: stored=%s, recomputed=%s.",
                path, stored, recomputed,
            )
    return data


def load_results(results_dir: Path, verify: bool = False) -> list[dict]:
    """Load every JSON result file in a directory in filename sort order.
    Checksum verification is off by default because batch-loading 15k results
    doesn't need per-file tamper detection; pass verify=True for audits."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return [load_result(p, verify=verify) for p in sorted(results_dir.rglob("*.json"))]
