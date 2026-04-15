"""Extract stated confidence ratings from model responses."""

import re


def extract_confidence(text: str) -> int | None:
    """Extract a self-rated confidence score (1-10) from a response.
    
    Handles: "my confidence is 7", "confidence: 8/10", "I'd rate my 
    confidence at 6", "confidence level: 9", etc.
    """
    patterns = [
        r'confidence[:\s]+(\d{1,2})(?:\s*/\s*10)?',
        r'confidence (?:is|at|of|level[:\s]*)\s*(\d{1,2})',
        r"I(?:'d| would) (?:rate|say|give)\s+(?:my\s+)?confidence\s+(?:as |at |a )?(\d{1,2})",
        r'(\d{1,2})\s*/\s*10\s*(?:confidence)?',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 1 <= val <= 10:
                return val
    
    return None
