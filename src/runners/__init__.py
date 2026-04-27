"""Per-experiment runners (design.md D5 + D7).

Each experiment has its own module with a `run_<name>(config)` entry point.
The `RUNNERS` dispatch dict maps experiment_type strings (see
`src.runner.ExperimentType`) to the corresponding runner function.

Exp 1a currently delegates to the existing `src.runner.run_batch`;
other experiments are stubs that raise NotImplementedError until their
implementation tasks land (Phases 4-7).
"""

from .exp1a import run_exp1a
from .exp1b import run_exp1b
from .exp2 import run_exp2
from .exp3a import run_exp3a
from .exp3b import run_exp3b
from .exp3c import run_exp3c


RUNNERS = {
    "exp1a": run_exp1a,
    "exp1b": run_exp1b,
    "exp2": run_exp2,
    "exp3a": run_exp3a,
    "exp3b": run_exp3b,
    "exp3c": run_exp3c,
}


__all__ = [
    "run_exp1a",
    "run_exp1b",
    "run_exp2",
    "run_exp3a",
    "run_exp3b",
    "run_exp3c",
    "RUNNERS",
]
