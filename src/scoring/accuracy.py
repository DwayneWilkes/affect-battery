"""Score response correctness for the Affect Battery."""

import re


def extract_numeric_answer(text: str) -> float | None:
    """Extract a numeric answer from model output.

    Priority order per spec (scoring-pipeline Requirement: Numeric answer
    extraction):
        1. Explicit answer markers: "the answer is X", "answer: X",
           "total: X", "total = X", "result is X", "result: X",
           "equals X"
        2. Boxed answers: \\boxed{X}
        3. Equals sign: "= X"
        4. Last number in text (fallback)

    Within each priority tier, the LAST match in the text wins. Models
    emitting chain-of-thought responses produce intermediate '=' lines
    (e.g. "6 * 75 = 450 ... Total = 770"); picking the first match
    extracts an intermediate value rather than the final answer.
    """
    priority_patterns = [
        r'(?:the answer is|answer:|total:|total\s*=|result is|result:|equals)\s*\**\s*(-?[\d,]+\.?\d*)',
        r'\\boxed\{(-?[\d,]+\.?\d*)\}',
        r'(?:=)\s*\**\s*(-?[\d,]+\.?\d*)',
    ]
    for pattern in priority_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return float(matches[-1].replace(",", ""))

    # Fall back to last number in text.
    # Use word-boundary negative sign (preceded by space/start, not digit/letter)
    # to avoid parsing hyphens in ranges like "3-5" as negative numbers.
    numbers = re.findall(r'(?:^|(?<=\s))-?[\d,]+\.?\d*', text)
    if not numbers:
        # Try without negative sign as final fallback
        numbers = re.findall(r'[\d,]+\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].strip().replace(",", ""))
        except ValueError:
            pass
    
    return None


def score_arithmetic(response: str, expected: float) -> bool:
    """Check if the response contains the correct numeric answer."""
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return False
    return abs(extracted - expected) < 0.01


def score_factual_qa(
    response: str,
    expected_answer: str,
    aliases: list[str] | None = None,
) -> float:
    """Score factual QA with substring matching against the canonical
    expected answer and any provided aliases. Returns 0.0 or 1.0.

    Aliases handle benchmark items like ('United States', ['U.S.', 'USA',
    'America']) where the model may emit any surface form. Matching is
    case-insensitive and substring-based to tolerate framing prose
    ('The answer is USA.' matches alias 'USA'). The numeric-fallback
    branch is preserved so numeric expected values still match via
    extract_numeric_answer.
    """
    response_lower = response.lower().strip()
    expected_lower = expected_answer.lower().strip()

    candidates: list[str] = []
    if expected_lower:
        candidates.append(expected_lower)
    if aliases:
        candidates.extend(a.lower().strip() for a in aliases if a and a.strip())

    if not candidates:
        return 0.0

    for c in candidates:
        if c and c in response_lower:
            return 1.0

    # Numeric-match fallback: applies when the canonical expected is numeric.
    try:
        expected_num = float(expected_lower)
        extracted = extract_numeric_answer(response)
        if extracted is not None and abs(extracted - expected_num) < 0.01:
            return 1.0
    except ValueError:
        pass

    return 0.0
