"""Score response correctness for the Affect Battery."""

import re

# LaTeX digit-group separators: the `{,}` idiom and the thin-space kerns
# (\, \! \; \:). Anchored between two digits so ordinary braces, commas
# and spacing macros elsewhere in the response are left alone.
_LATEX_DIGIT_GROUP_SEPARATOR = re.compile(
    r'(?<=\d)(?:\{\s*\\?,\s*\}|\\[,!;:])(?=\d)'
)

# A currency marker sitting between `=` and its number, in the escaped
# math form (\$) or bare ($). Without this the equals tier sees no digit
# after the marker and the answer falls through to a weaker tier. The
# marker must be glued to the digits: a `$` separated from them by a
# space is an inline-math delimiter, not currency.
_CURRENCY_AFTER_EQUALS = re.compile(r'(=\s*\**\s*)(?:\\?\$)+(?=\d)')

# A number token must start with a digit (after an optional sign) so a
# stray comma never reaches float().
_NUMBER = r'-?\d[\d,]*\.?\d*'

# Absolute tolerance for arithmetic correctness. GSM-Hard expecteds run
# to nine significant figures, so this is tighter than it looks.
ARITHMETIC_TOLERANCE = 0.01


def _normalise(text: str) -> str:
    """Strip LaTeX digit grouping and post-equals currency markers."""
    text = _LATEX_DIGIT_GROUP_SEPARATOR.sub("", text)
    return _CURRENCY_AFTER_EQUALS.sub(r'\1', text)


def _to_float(token: str) -> float | None:
    """Parse an extracted token, returning None rather than raising."""
    try:
        return float(token.strip().replace(",", ""))
    except ValueError:
        return None


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

    The text is normalised before any tier runs: LaTeX digit-group
    separators are removed so "22{,}463{,}288" and "8\\,111\\,382" read
    as single numbers, and a currency marker following an equals sign is
    dropped so "= \\$14" reaches the equals tier.
    """
    normalised = _normalise(text)

    priority_patterns = [
        rf'(?:the answer is|answer:|total:|total\s*=|result is|result:|equals)\s*\**\s*({_NUMBER})',
        rf'\\boxed\{{({_NUMBER})\}}',
        rf'(?:=)\s*\**\s*({_NUMBER})',
    ]
    for pattern in priority_patterns:
        matches = re.findall(pattern, normalised, re.IGNORECASE)
        if matches:
            value = _to_float(matches[-1])
            if value is not None:
                return value

    # Fall back to last number in text.
    # Use word-boundary negative sign (preceded by space/start, not digit/letter)
    # to avoid parsing hyphens in ranges like "3-5" as negative numbers.
    numbers = re.findall(rf'(?:^|(?<=\s)){_NUMBER}', normalised)
    if not numbers:
        # Try without negative sign as final fallback
        numbers = re.findall(r'\d[\d,]*\.?\d*', normalised)
    if numbers:
        return _to_float(numbers[-1])

    return None


def score_arithmetic_binary(response: str, expected: float | str | None) -> int:
    """Binary correctness at ARITHMETIC_TOLERANCE against `expected`.

    Returns 0 when the response carries no number or when `expected` is
    not numeric. Single scorer for the exp3a runner and the analysis and
    probe scripts, which all compare a str-valued bank `expected`.
    """
    extracted = extract_numeric_answer(response)
    if extracted is None:
        return 0
    try:
        target = float(expected)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return int(abs(extracted - target) < ARITHMETIC_TOLERANCE)


def score_arithmetic(response: str, expected: float) -> bool:
    """Check if the response contains the correct numeric answer."""
    return bool(score_arithmetic_binary(response, expected))


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
        if extracted is not None and abs(extracted - expected_num) < ARITHMETIC_TOLERANCE:
            return 1.0
    except ValueError:
        pass

    return 0.0
