"""Generalized live dashboard for in-flight experiment runs.

The package's submodules import from `src.lib.*`, which requires the
repo root on `sys.path`. The entry point (`dashboard.py`) handles that
when invoked as a script; for any other code path that imports a
submodule directly (tests, ad-hoc REPL), this `__init__` is the
one-shot place that adds it. Idempotent: re-add is a no-op."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
