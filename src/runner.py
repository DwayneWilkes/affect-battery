"""Experiment runner for the Affect Battery."""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import AsyncIterator

from .conditioning.prompts import Condition, FEEDBACK_SETS
from .conditioning.protocol import ConditioningProtocol, Message, build_conditioning_messages, build_transfer_messages
from .conditioning.tasks import get_arithmetic_problems, get_transfer_tasks, MathProblem
from .models import ModelClient
from .scoring.accuracy import extract_numeric_answer

log = logging.getLogger(__name__)


class ExperimentType(str, Enum):
    TRANSFER_WITHIN = "transfer_within"
    TRANSFER_CROSS = "transfer_cross"
    PERSISTENCE = "persistence"
    AROUSAL_PERFORMANCE = "arousal_performance"
    COGNITIVE_SCOPE = "cognitive_scope"
    CONSERVATIVE_SHIFT = "conservative_shift"


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


@dataclass
class RunResult:
    config: dict
    run_number: int
    conditioning_responses: list[str]
    conditioning_correct: list[bool]
    transfer_responses: list[str]
    transfer_questions: list[str]
    transfer_expected: list[str]
    start_time: float
    end_time: float
    checksum: str = ""
    
    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum, excluding the checksum field itself."""
        data = asdict(self)
        data.pop("checksum", None)
        content = json.dumps(data, sort_keys=True, default=str)
        self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.checksum


async def run_single(
    config: ExperimentConfig,
    client: ModelClient,
    run_number: int,
) -> RunResult:
    """Run a single conditioning + transfer experiment."""
    start = time.time()
    
    # Generate problems and transfer questions
    seed = (config.seed or 0) + run_number
    problems = get_arithmetic_problems(config.num_conditioning_turns, seed=seed)
    transfer_qs = get_transfer_tasks(config.transfer_task, config.num_transfer_questions, seed=seed)
    
    # Phase 1: Conditioning
    protocol = ConditioningProtocol(
        condition=config.condition,
        num_conditioning_turns=config.num_conditioning_turns,
    )
    
    conditioning_responses = []
    conditioning_correct = []
    messages = [{"role": "system", "content": protocol.system_prompt}]
    
    if config.condition != Condition.NO_CONDITIONING:
        # Use structured FEEDBACK_SETS (per-turn) rather than the single-string
        # FEEDBACK_TEMPLATES legacy facade (task 8.5). Each turn picks its own
        # feedback text, isolating per-turn variation as a first-class axis.
        feedback_set = FEEDBACK_SETS.get(config.condition)

        for i, problem in enumerate(problems):
            # Ask the question
            messages.append({"role": "user", "content": problem.question})
            response = await client.complete(messages, temperature=config.temperature)
            messages.append({"role": "assistant", "content": response})
            conditioning_responses.append(response)

            # Check correctness
            extracted = extract_numeric_answer(response)
            is_correct = extracted is not None and abs(extracted - problem.answer) < 0.01
            conditioning_correct.append(is_correct)

            # Give feedback using this turn's entry in the FeedbackSet.
            # The correct/incorrect split is already encoded per-turn
            # (STRONG_POSITIVE / STRONG_NEGATIVE turns set correct == incorrect
            # so the call site does not need to special-case them).
            if feedback_set and i < len(feedback_set.turns):
                turn = feedback_set.turns[i]
                feedback = turn.correct if is_correct else turn.incorrect
                messages.append({"role": "user", "content": feedback})
    
    # Phase 1.5: Neutral buffer turns (for persistence experiments)
    for _ in range(config.neutral_turns):
        buffer_problems = get_arithmetic_problems(1, seed=seed + 1000 + _)
        messages.append({"role": "user", "content": buffer_problems[0].question})
        response = await client.complete(messages, temperature=config.temperature)
        messages.append({"role": "assistant", "content": response})
    
    # Phase 2: Transfer
    if config.experiment_type == ExperimentType.TRANSFER_CROSS:
        # New conversation for cross-session transfer
        messages = [{"role": "system", "content": "Answer the following questions to the best of your ability."}]
    
    transfer_responses = []
    for q in transfer_qs:
        messages.append({"role": "user", "content": q.question})
        response = await client.complete(messages, temperature=config.temperature)
        messages.append({"role": "assistant", "content": response})
        transfer_responses.append(response)
    
    result = RunResult(
        config=asdict(config),
        run_number=run_number,
        conditioning_responses=conditioning_responses,
        conditioning_correct=conditioning_correct,
        transfer_responses=transfer_responses,
        transfer_questions=[q.question for q in transfer_qs],
        transfer_expected=[q.expected_answer for q in transfer_qs],
        start_time=start,
        end_time=time.time(),
    )
    result.compute_checksum()
    return result


async def run_batch(
    config: ExperimentConfig,
    client: ModelClient,
    max_concurrent: int = 5,
) -> AsyncIterator[RunResult]:
    """Run a batch of experiments with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(run_num: int) -> RunResult:
        async with semaphore:
            log.info(f"Run {run_num}/{config.num_runs} | {config.condition.value} | {config.model_name}")
            return await run_single(config, client, run_num)
    
    tasks = [run_with_semaphore(i) for i in range(config.num_runs)]
    for coro in asyncio.as_completed(tasks):
        yield await coro


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
    """Save a single result as JSON. Validates required config keys and
    transfer-array length consistency per spec Requirement: Result JSON schema."""
    _validate_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    name = f"{cfg['model_name'].split('/')[-1]}_{cfg['condition']}_{cfg['experiment_type']}_{result.run_number:04d}.json"
    path = output_dir / name
    path.write_text(json.dumps(asdict(result), indent=2, default=str))
    return path


def load_result(path: Path) -> dict:
    """Load a saved result JSON and verify the stored checksum.

    Logs a warning if the recomputed checksum does not match the stored one
    (indicates post-write tampering). Returns the loaded dict either way so
    downstream analyses can choose to proceed or quarantine.
    """
    data = json.loads(Path(path).read_text())
    stored = data.get("checksum", "")
    # Recompute using the same algorithm as RunResult.compute_checksum.
    without_checksum = {k: v for k, v in data.items() if k != "checksum"}
    content = json.dumps(without_checksum, sort_keys=True, default=str)
    recomputed = hashlib.sha256(content.encode()).hexdigest()[:16]
    if stored and recomputed != stored:
        log.warning(
            "Checksum mismatch in %s: stored=%s, recomputed=%s. "
            "File may have been tampered with.",
            path, stored, recomputed,
        )
    return data


def load_results(results_dir: Path) -> list[dict]:
    """Load every JSON result file in a directory. Each file is checksum-
    verified via load_result. Returns a list of result dicts in filename
    sort order. Empty directory -> empty list."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return [load_result(p) for p in sorted(results_dir.glob("*.json"))]
