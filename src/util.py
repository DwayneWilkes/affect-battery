"""Shared helpers used by runner and tracking.

Kept small on purpose: one helper per genuine cross-module need.
"""

import hashlib
import json
from typing import Any


CHECKSUM_KEY = "checksum"
CHECKSUM_HEX_DIGITS = 16


def enum_value(x: Any) -> Any:
    """Return `.value` on an Enum, pass strings through unchanged.

    Needed because Python 3.12 f-string / json.dumps(default=str) on a
    str-mixin Enum produces the repr form ('Condition.STRONG_POSITIVE')
    instead of the value ('strong_positive').
    """
    return getattr(x, "value", x)


def model_slug(model_name: str) -> str:
    """Strip any 'org/' prefix so filenames do not embed path separators."""
    return str(model_name).rsplit("/", 1)[-1]


def checksum_of_payload(data: dict) -> str:
    """SHA-256 hex prefix of a JSON-serialised dict, with `checksum` removed.

    Used for both computing a result's stored checksum and re-deriving it on
    load for tamper detection. Serialisation is stable via sort_keys.
    """
    without = {k: v for k, v in data.items() if k != CHECKSUM_KEY}
    content = json.dumps(without, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:CHECKSUM_HEX_DIGITS]
