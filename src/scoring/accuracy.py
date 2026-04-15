"""Score response correctness for the Affect Battery."""

import re


def extract_numeric_answer(text: str) -> float | None:
    """Extract a numeric answer from model output.
    
    Handles: "the answer is 42", "= 42", "\\boxed{42}", "42.", 
    and falls back to the last number in the text.
    """
    # Try explicit answer patterns first
    patterns = [
        r'(?:the answer is|answer:)\s*(-?[\d,]+\.?\d*)',
        r'(?:=)\s*(-?[\d,]+\.?\d*)',
        r'\\boxed\{(-?[\d,]+\.?\d*)\}',
        r'(?:equals|result is)\s*(-?[\d,]+\.?\d*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    
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


def score_factual_qa(response: str, expected_answer: str) -> float:
    """Score factual QA with fuzzy matching. Returns 0.0 or 1.0."""
    response_lower = response.lower().strip()
    expected_lower = expected_answer.lower().strip()

    if not expected_lower:
        return 0.0

    if expected_lower in response_lower:
        return 1.0
    
    # Check for numeric match
    try:
        expected_num = float(expected_lower)
        extracted = extract_numeric_answer(response)
        if extracted is not None and abs(extracted - expected_num) < 0.01:
            return 1.0
    except ValueError:
        pass
    
    return 0.0
