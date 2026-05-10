"""Generalized live dashboard for in-flight experiment runs.

Importing this package puts the repo root on `sys.path` so submodules
can resolve `src.lib.*`. The check is idempotent."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
