#!/usr/bin/env python3
"""Run the H3b Phase 1A 20-pass single-turn calibrated replication.

Pre-registration: docs/preregistrations/h3b_2026-05-07.md

Each of N passes writes to its own subdirectory under the output base, so
parallel passes never collide on filename. The pre-reg locks the
(item_id, level) cell as the unit of analysis, not pass order, so corpora
aggregated across pass_*/level_M/ directories are equivalent to a
sequential run.

Usage:
  python scripts/pilots/run_h3b_phase1a.py \\
    --prereg-commit DwayneWilkes/affect-battery@<squash-sha> \\
    [--output-base results/h3b_2026-05-07] \\
    [--max-parallel 20] [--n-passes 20] [--seed 42] [--dry-run]

Environment:
  OPENAI_API_KEY must be set (skipped under --dry-run).
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUTPUT_BASE = "results/h3b_2026-05-07"
DEFAULT_BANK = "configs/banks/h3b_calibrated_v2.yaml"
DEFAULT_RUNNER_CONFIG = "configs/exp3a_runner_h3b_2026-05-07.yaml"
PREREG_PATH = "docs/preregistrations/h3b_2026-05-07.md"
TEMPERATURE = "0.7"
MODEL = "gpt-5.4-nano"

CELL_BASENAME_RE = re.compile(r"^\d+\.json$")


class WrapperError(Exception):
    """User-facing usage / config error. Caller maps to exit 1 + stderr."""


class _ShutdownRequested(BaseException):
    """Internal: raised from a signal handler to break out of any wait.
    Inherits from BaseException so generic `except Exception` clauses
    in helper code don't swallow it; only `main` catches it."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code


class _ArgParser(argparse.ArgumentParser):
    """Argparse subclass that exits with code 1 (not the default 2) on
    argument errors, matching the rest of the wrapper's "exit 1 = user
    error" convention."""

    def error(self, message: str):
        self.print_usage(sys.stderr)
        sys.stderr.write(f"ERROR: {message}\n")
        sys.exit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    p = _ArgParser(
        prog="run_h3b_phase1a.py",
        description=(
            "Run the H3b Phase 1A N-pass single-turn calibrated replication. "
            f"Pre-registration: {PREREG_PATH}."
        ),
    )
    p.add_argument(
        "--prereg-commit",
        required=True,
        help="Pre-registration commit reference, e.g. owner/repo@<sha>.",
    )
    p.add_argument(
        "--output-base",
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output root (default: {DEFAULT_OUTPUT_BASE}).",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=20,
        help="Max concurrent passes in flight (default: 20).",
    )
    p.add_argument(
        "--n-passes",
        type=int,
        default=20,
        help="Number of passes (default: 20).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampler seed (default: 42).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Pass --dry-run through to affect-battery (canned responses, no "
            "API calls; bypasses pre-reg/power gates). For wiring/E2E sanity."
        ),
    )
    p.add_argument(
        "--power-report-path",
        type=Path,
        default=None,
        help=(
            "Path to the power/precision report JSON pinned by the prereg. "
            "Required for non-dry-run invocations unless --skip-power-gate "
            "is set. Passed through to affect-battery."
        ),
    )
    p.add_argument(
        "--power-report-sha",
        default=None,
        help=(
            "SHA-256 of the power-report file. The runner verifies the "
            "file's hash matches; mismatch aborts the run. Passed through "
            "to affect-battery."
        ),
    )
    p.add_argument(
        "--skip-power-gate",
        default=None,
        help=(
            "Bypass the power-gate with a written rationale (passed through "
            "verbatim to affect-battery). For smoke runs where the power "
            "report is not yet pinned. Use sparingly; rationale appears in "
            "the run's audit log."
        ),
    )
    return p


def validate_args(args: argparse.Namespace) -> None:
    """Range checks beyond what argparse type=int catches. Bank and
    runner-config existence checks are here because the file paths are
    hardcoded to the prereg-pinned artifacts; argparse can't validate
    them."""
    if args.n_passes <= 0:
        raise WrapperError(
            f"--n-passes must be a positive integer (got {args.n_passes})"
        )
    if args.max_parallel <= 0:
        raise WrapperError(
            f"--max-parallel must be a positive integer (got {args.max_parallel})"
        )
    if args.seed < 0:
        raise WrapperError(
            f"--seed must be a non-negative integer (got {args.seed})"
        )
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        raise WrapperError(
            "OPENAI_API_KEY env var must be set "
            "(skip this check by adding --dry-run for offline E2E sanity)"
        )
    bank_path = Path(DEFAULT_BANK)
    if not bank_path.is_file():
        raise WrapperError(
            f"bank YAML not found at {bank_path} (run from repo root)"
        )
    runner_path = Path(DEFAULT_RUNNER_CONFIG)
    if not runner_path.is_file():
        raise WrapperError(
            f"runner config not found at {runner_path} (run from repo root)"
        )


def count_bank_items(bank_path: Path) -> int:
    """Count items by matching `^- id:` lines. Avoids requiring a YAML
    parser dep; the parent emitter (build_calibrated_bank.py) writes one
    `- id:` per item at column 0."""
    return len(re.findall(r"^- id:", bank_path.read_text(), re.MULTILINE))


def count_intensity_levels(runner_config: Path) -> int:
    """Count integers in the runner config's `intensity_levels` field.

    Handles both YAML forms:
      inline   `intensity_levels: [1, 2, 3]`
      block    `intensity_levels:\\n  - 1\\n  - 2\\n  - 3`
    The block form's continuation lines are at deeper indent than the
    key line, so we accept the key line plus any subsequent lines that
    are either indented or start with `-` until we hit a sibling key
    (column-0 non-comment line)."""
    text = runner_config.read_text()
    lines = text.splitlines()
    in_block = False
    captured: list[str] = []
    for line in lines:
        if not in_block:
            if re.match(r"^intensity_levels:", line):
                in_block = True
                captured.append(line)
                # Inline form: the value is on the same line, done after this.
                if re.search(r"\[.*\]", line):
                    break
            continue
        # In block: continuation lines are indented (or empty); a column-0
        # non-comment line ends the block.
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        captured.append(line)
    if not captured:
        return 0
    return len(re.findall(r"\d+", "\n".join(captured)))


def find_cell_files(directory: Path) -> list[Path]:
    """Return cell files (numeric basename) under any `level_*/neutral/`.

    Output layout for exp3a is `<dir>/data/level_<N>/neutral/<NN>.json`.
    The "neutral" segment is `affect-battery run`'s default --condition
    string (cli.py condition_value). exp3a rejects --condition per
    prereg §3.4.1, so every cell lands under that subdirectory. The
    numeric-basename filter excludes per-pass `manifest.yaml` and any
    operator-dropped scratch JSONs."""
    if not directory.is_dir():
        return []
    cells: list[Path] = []
    for level_dir in directory.glob("**/level_*"):
        neutral = level_dir / "neutral"
        if not neutral.is_dir():
            continue
        for entry in neutral.iterdir():
            if entry.is_file() and CELL_BASENAME_RE.match(entry.name):
                cells.append(entry)
    return cells


def count_cell_files(directory: Path) -> int:
    return len(find_cell_files(directory))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class PassRunner:
    """Owns dispatch/drain/cleanup for a single wrapper invocation.

    Failure-fast guarantee: the first pass to exit non-zero triggers
    `terminate_inflight` regardless of which loop sees it (bounded-
    concurrency dispatch or final unbounded drain). A `cleanup_triggered`
    flag prevents redundant teardown calls.

    Race-free pgroup signaling: each dispatch uses `start_new_session=True`,
    which calls `os.setsid()` in the child between fork and exec.
    `subprocess.Popen` synchronizes via an internal pipe so it does not
    return until the child's exec has succeeded — by then setsid has run
    and the pgid is established. This is structurally different from
    bash's `setsid prog &` where the parent sees `$!` immediately and
    can race the child's setsid call."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        n_items: int,
        n_levels: int,
    ) -> None:
        self.args = args
        self.output_base = Path(args.output_base)
        self.bank = Path(DEFAULT_BANK)
        self.runner_config = Path(DEFAULT_RUNNER_CONFIG)
        self.n_items = n_items
        self.n_levels = n_levels
        self.expected_per_pass = n_items * n_levels
        self.in_flight: dict[int, tuple[subprocess.Popen, int]] = {}
        self.failed_passes: list[int] = []
        self.cleanup_triggered = False
        self.manifest_path = self.output_base / "run_manifest.txt"

    def write_manifest_header(self) -> None:
        lines = [
            f"H3b Phase 1A: single-turn calibrated replication "
            f"({self.args.n_passes} within-subjects passes)",
            f"prereg:                {PREREG_PATH}",
            f"prereg_commit:         {self.args.prereg_commit}",
            f"bank:                  {self.bank}",
            f"runner_config:         {self.runner_config}",
            f"n_bank_items:          {self.n_items}",
            f"n_levels:              {self.n_levels}",
            f"expected_cells/pass:   {self.expected_per_pass}",
            f"n_passes:              {self.args.n_passes}",
            f"max_parallel:          {self.args.max_parallel}",
            f"seed:                  {self.args.seed}",
            f"started_utc:           {utc_now_iso()}",
        ]
        self.manifest_path.write_text("\n".join(lines) + "\n")

    def append_manifest(self, line: str) -> None:
        with self.manifest_path.open("a") as f:
            f.write(line + "\n")

    def pass_dir(self, pass_num: int) -> Path:
        return self.output_base / f"pass_{pass_num:02d}"

    def existing_cells(self, pass_num: int) -> int:
        return count_cell_files(self.pass_dir(pass_num))

    def build_command(self, pass_num: int) -> list[str]:
        # --num-runs is the per-level item count for the within_subjects
        # sampler; for "every item appears at every level" to hold, it
        # must equal the bank's item count. Derive from self.n_items
        # (already counted from the bank YAML on startup) instead of
        # hardcoding, so a recalibrated bank with a different item count
        # doesn't silently undercount.
        cmd = [
            "affect-battery", "run",
            "--experiment", "exp3a",
            "--provider", "openai",
            "--model", MODEL,
            "--transfer-bank", str(self.bank),
            "--num-runs", str(self.n_items),
            "--seed", str(self.args.seed),
            "--temperature", TEMPERATURE,
            "--runner-config", str(self.runner_config),
            "--pre-registration-github-commit", self.args.prereg_commit,
            "--output-dir", str(self.pass_dir(pass_num)),
        ]
        if self.args.dry_run:
            cmd.append("--dry-run")
        if self.args.power_report_path is not None:
            cmd += ["--power-report-path", str(self.args.power_report_path)]
        if self.args.power_report_sha is not None:
            cmd += ["--power-report-sha", self.args.power_report_sha]
        if self.args.skip_power_gate is not None:
            cmd += ["--skip-power-gate", self.args.skip_power_gate]
        return cmd

    def maybe_skip(self, pass_num: int) -> bool:
        """Return True if the pass should be skipped (already complete).

        Partial passes are re-dispatched and rely on the runner's per-cell
        cache to resume; the wrapper does not delete the partial state."""
        existing = self.existing_cells(pass_num)
        if existing >= self.expected_per_pass:
            print(
                f"pass {pass_num}: {existing}/{self.expected_per_pass} cells "
                f"complete, skipping"
            )
            return True
        if existing > 0:
            print(
                f"pass {pass_num}: {existing}/{self.expected_per_pass} cells "
                f"partial, re-dispatching (runner cache resumes)"
            )
        else:
            print(f"pass {pass_num}: dispatching to {self.pass_dir(pass_num)}")
        return False

    def dispatch(self, pass_num: int) -> None:
        # Known minor race: a signal arriving between `Popen` returning
        # and the IN_FLIGHT assignment below would let _ShutdownRequested
        # propagate before the proc is registered, orphaning that one
        # pass. The window is between two adjacent Python bytecodes
        # (microseconds) — the obvious mitigation (pthread_sigmask
        # SIG_BLOCK) breaks signal delivery to children because the
        # blocked mask is inherited across fork() and Python preserves
        # it across exec(). preexec_fn could reset the child mask but
        # has its own deadlock caveats. Documented as a limitation;
        # operationally negligible for a few-hours pilot.
        proc = subprocess.Popen(
            self.build_command(pass_num),
            start_new_session=True,
        )
        self.in_flight[proc.pid] = (proc, pass_num)

    def terminate_inflight(self) -> None:
        """SIGTERM the pgroup of every in-flight pass. Idempotent: a
        process that exited between the lookup and the kill yields
        ProcessLookupError which we swallow."""
        for pid in list(self.in_flight):
            try:
                pgid = os.getpgid(pid)
            except ProcessLookupError:
                continue
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def account_status(self, pass_num: int, rc: int) -> None:
        if rc == 0:
            return
        self.failed_passes.append(pass_num)
        if not self.cleanup_triggered:
            print(
                f"ERROR: pass {pass_num} failed (exit {rc}); "
                f"terminating remaining in-flight passes",
                file=sys.stderr,
            )
            self.terminate_inflight()
            self.cleanup_triggered = True

    def reap_one(self) -> None:
        """Block until any in-flight child exits, then account it.

        `os.wait` returns `(pid, raw_status)`. `os.waitstatus_to_exitcode`
        normalises: positive int = exit code, negative int = -signum (we
        convert to bash's 128+signum convention so account_status sees
        a uniform non-zero rc on signal-killed children)."""
        try:
            pid, raw_status = os.wait()
        except ChildProcessError:
            return  # nothing left to wait for
        if pid not in self.in_flight:
            return  # foreign child (shouldn't happen)
        _proc, pass_num = self.in_flight.pop(pid)
        rc = os.waitstatus_to_exitcode(raw_status)
        if rc < 0:
            rc = 128 + (-rc)
        self.account_status(pass_num, rc)

    def drain(self) -> None:
        while self.in_flight:
            self.reap_one()

    def run(self) -> int:
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.write_manifest_header()

        # Dispatch + drain phases share a try/finally: if Popen raises
        # mid-loop (e.g., affect-battery missing from PATH, fork fails
        # under resource pressure) after some passes have already been
        # dispatched, the prior in-flight passes still need to be
        # terminated and reaped. Without this, the exception propagates
        # leaving orphaned children. _ShutdownRequested has its own
        # path through main(), so let it through unmodified.
        try:
            for pass_num in range(1, self.args.n_passes + 1):
                if self.maybe_skip(pass_num):
                    continue
                self.dispatch(pass_num)
                if len(self.in_flight) >= self.args.max_parallel:
                    self.reap_one()
                    if self.cleanup_triggered:
                        break
            self.drain()
        except _ShutdownRequested:
            raise
        except BaseException:
            self.terminate_inflight()
            try:
                self.drain()
            except BaseException:
                pass  # best-effort; the original exception is what matters
            raise

        self.append_manifest(f"completed_utc:         {utc_now_iso()}")
        self.append_manifest(f"failed_passes:         {len(self.failed_passes)}")

        if self.failed_passes:
            print()
            print(
                f"FAILURE: {len(self.failed_passes)} pass(es) failed: "
                f"{' '.join(str(p) for p in self.failed_passes)}",
                file=sys.stderr,
            )
            print(f"Manifest: {self.manifest_path}", file=sys.stderr)
            return 1

        print()
        print(f"All {self.args.n_passes} passes complete. "
              f"Manifest: {self.manifest_path}")
        print("Cell-count check:")
        total = sum(
            self.existing_cells(p) for p in range(1, self.args.n_passes + 1)
        )
        expected = self.args.n_passes * self.expected_per_pass
        print(
            f"  total result JSONs: {total} (expected {expected} = "
            f"{self.args.n_passes} passes × {self.n_items} items × "
            f"{self.n_levels} levels)"
        )
        if total != expected:
            sys.stderr.write(
                f"ERROR: cell-count mismatch — got {total}, expected {expected}\n"
                f"       Pre-registered analysis assumes {expected} cells; "
                f"missing\n"
                f"       cells will silently bias the (item_id, level) corpus. "
                f"Inspect\n"
                f"       per-pass cell counts under {self.output_base}/pass_*/ "
                f"before any\n"
                f"       analysis or amendment.\n"
            )
            self.append_manifest(
                f"cell_count_check:      FAIL ({total}/{expected})"
            )
            return 1
        self.append_manifest(f"cell_count_check:      OK ({total}/{expected})")
        return 0


def install_signal_handlers(runner: PassRunner) -> None:
    """SIGINT → exit 130, SIGTERM → exit 143 (bash 128+signum convention).

    The handler raises `_ShutdownRequested` rather than calling sys.exit
    directly. That gives `main` one well-defined catch site that can
    invoke cleanup on its own terms (terminate + drain) before exiting.
    Inheriting from BaseException avoids being swallowed by helper-code
    `except Exception` clauses."""
    def make(exit_code: int):
        def handler(signum, frame):
            raise _ShutdownRequested(exit_code)
        return handler
    signal.signal(signal.SIGINT, make(130))
    signal.signal(signal.SIGTERM, make(143))


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_args(args)
    except WrapperError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    bank_path = Path(DEFAULT_BANK)
    runner_config_path = Path(DEFAULT_RUNNER_CONFIG)
    n_items = count_bank_items(bank_path)
    n_levels = count_intensity_levels(runner_config_path)
    if n_items < 1 or n_levels < 1:
        print(
            f"ERROR: could not derive item or level counts "
            f"(items={n_items}, levels={n_levels})",
            file=sys.stderr,
        )
        return 1

    runner = PassRunner(args, n_items=n_items, n_levels=n_levels)
    install_signal_handlers(runner)

    try:
        return runner.run()
    except _ShutdownRequested as e:
        runner.terminate_inflight()
        # Drain quietly — children are SIGTERM'd, just collect their exits.
        try:
            runner.drain()
        except _ShutdownRequested:
            pass  # second signal during shutdown; just exit
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
