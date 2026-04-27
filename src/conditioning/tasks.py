"""Task pools for conditioning and transfer phases."""

from dataclasses import dataclass, field
from pathlib import Path
import random


@dataclass
class MathProblem:
    question: str
    answer: float


@dataclass
class TransferQuestion:
    question: str
    expected_answer: str
    task_type: str
    difficulty: str = "medium"
    expected_aliases: list[str] = field(default_factory=list)


def get_arithmetic_problems(n: int, seed: int | None = None) -> list[MathProblem]:
    """Generate n arithmetic problems for the conditioning phase."""
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            answer = a + b
        elif op == "-":
            answer = a - b
        else:
            answer = a * b
        problems.append(MathProblem(
            question=f"What is {a} {op} {b}?",
            answer=float(answer),
        ))
    return problems


# Hardcoded pilot pools. Production should use established benchmarks.
_FACTUAL_QA: list[TransferQuestion] = [
    TransferQuestion("What is the capital of Australia?", "Canberra", "factual_qa", "easy"),
    TransferQuestion("Who wrote 'Pride and Prejudice'?", "Jane Austen", "factual_qa", "easy"),
    TransferQuestion("What element has the atomic number 79?", "Gold", "factual_qa", "medium"),
    TransferQuestion("In what year did the Berlin Wall fall?", "1989", "factual_qa", "medium"),
    TransferQuestion("What is the smallest prime number greater than 50?", "53", "factual_qa", "hard"),
    TransferQuestion("Who was the first person to circumnavigate the globe?", "Ferdinand Magellan", "factual_qa", "hard"),
]

_LOGIC: list[TransferQuestion] = [
    TransferQuestion(
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "No", "logic", "medium",
    ),
    TransferQuestion(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "$0.05", "logic", "medium",
    ),
    TransferQuestion(
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "5 minutes", "logic", "medium",
    ),
]

_CREATIVE: list[TransferQuestion] = [
    TransferQuestion(
        "Write a short paragraph (3-4 sentences) continuing this story: 'The last train had already left the station when...'",
        "", "creative", "medium",
    ),
    TransferQuestion(
        "List as many uses for a paperclip as you can think of.",
        "", "creative", "medium",
    ),
]


def _load_bank_pool(bank_path: str | Path) -> list[TransferQuestion]:
    """Load a transfer-question bank YAML into a list of TransferQuestion.

    Bank schema (per ingestion script): each item carries `id`, `question`,
    `expected`, `answer_aliases`, `difficulty`. We project these into the
    TransferQuestion shape so the caller is bank-agnostic.
    """
    import yaml

    data = yaml.safe_load(Path(bank_path).read_text())
    items = data.get("items", [])
    out: list[TransferQuestion] = []
    for it in items:
        out.append(TransferQuestion(
            question=str(it["question"]),
            expected_answer=str(it["expected"]),
            task_type="factual_qa",
            difficulty=str(it.get("difficulty", "medium")),
            expected_aliases=list(it.get("answer_aliases") or []),
        ))
    return out


def get_transfer_tasks(
    task_type: str,
    n: int,
    seed: int | None = None,
    bank_path: str | Path | None = None,
) -> list[TransferQuestion]:
    """Get n transfer task questions of the given type.

    When `bank_path` is provided, the sampling pool comes from the YAML
    bank at that path (alias-aware items). Otherwise the hardcoded
    pilot pools are used. Sampling is deterministic given `seed`.
    """
    if bank_path is not None:
        pool = _load_bank_pool(bank_path)
    else:
        pools = {
            "factual_qa": _FACTUAL_QA,
            "logic": _LOGIC,
            "creative": _CREATIVE,
        }
        pool = pools.get(task_type, _FACTUAL_QA)
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))
