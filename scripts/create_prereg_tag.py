"""Create a Git pre-registration tag at the current commit and print
the CLI invocation that uses it.

GitHub-as-pre-registration is a legitimate pre-registration vehicle for
projects where the methodology lives in the repo: the specs/ directory,
configs/osf_prereg_v1.yaml, and the ingestion scripts collectively
encode the planned analysis. A signed Git tag at a specific commit
provides timestamping + immutability comparable to OSF.

This script:
  1. Verifies the working tree is clean (no uncommitted changes).
  2. Verifies the current branch is pushed to origin (the commit must
     be reachable on a public ref for the pre-reg to be verifiable).
  3. Reads the GitHub remote owner/repo from the origin URL.
  4. Creates an annotated tag at HEAD with the supplied date + study
     name. Optionally signed with GPG when --sign is passed.
  5. Pushes the tag.
  6. Prints the exact `affect-battery run --pre-registration-github-commit`
     invocation that cites this tag.

Usage:
    python -m scripts.create_prereg_tag \\
        --tag prereg-affect-battery-2026-04-26 \\
        --message "Affect Battery study, full pre-registration" \\
        --sign

Once the tag is pushed, runs that cite it produce result files whose
`config.pre_registration_github_commit` field locks the methodology
to that exact commit. Reviewers can `git show <tag>` to see the
methodology that was pre-registered.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date


GITHUB_REMOTE_PATTERNS = [
    re.compile(r"^git@github\.com:(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?$"),
    re.compile(r"^https://github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?$"),
]


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def _check_clean_tree() -> None:
    """Refuse to tag if the working tree has uncommitted changes."""
    proc = _run(["git", "status", "--porcelain"])
    if proc.returncode != 0:
        sys.exit(f"git status failed: {proc.stderr}")
    if proc.stdout.strip():
        sys.exit(
            "error: working tree has uncommitted changes. The pre-reg "
            "tag must point to a clean commit. Stage and commit, or "
            "stash, before tagging.\n\n" + proc.stdout
        )


def _check_branch_pushed() -> str:
    """Verify the current commit is reachable from origin. Returns the
    current commit SHA."""
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if not head:
        sys.exit("error: cannot resolve HEAD")
    # Find a remote ref that contains HEAD.
    contained = _run(
        ["git", "branch", "-r", "--contains", head]
    ).stdout.strip()
    if not contained:
        sys.exit(
            f"error: HEAD ({head[:12]}) is not on any origin branch. "
            "Push your branch before creating a pre-reg tag.\n"
            "  git push origin <your-branch>"
        )
    return head


def _resolve_owner_repo() -> tuple[str, str]:
    proc = _run(["git", "remote", "get-url", "origin"])
    if proc.returncode != 0:
        sys.exit(f"git remote get-url origin failed: {proc.stderr}")
    url = proc.stdout.strip()
    for pat in GITHUB_REMOTE_PATTERNS:
        m = pat.match(url)
        if m:
            return m.group("owner"), m.group("repo")
    sys.exit(
        f"error: origin URL {url!r} doesn't look like a GitHub remote. "
        "GitHub-as-pre-reg requires a github.com remote."
    )


def create_tag(
    tag: str,
    message: str,
    sign: bool = False,
    skip_push: bool = False,
) -> str:
    """Create + push the pre-reg tag. Returns the GitHub commit ref."""
    _check_clean_tree()
    head = _check_branch_pushed()
    owner, repo = _resolve_owner_repo()

    # Refuse to overwrite an existing tag — pre-reg integrity is the
    # whole point.
    existing = _run(["git", "tag", "--list", tag]).stdout.strip()
    if existing:
        sys.exit(
            f"error: tag {tag!r} already exists. Pre-reg tags MUST NOT "
            "be moved or overwritten. Pick a new tag name (e.g., "
            "include the date) and re-run."
        )

    cmd = ["git", "tag", "-a", tag, "-m", message]
    if sign:
        cmd.insert(2, "-s")
    proc = _run(cmd)
    if proc.returncode != 0:
        sys.exit(f"git tag failed: {proc.stderr}")

    if not skip_push:
        push_proc = _run(["git", "push", "origin", tag])
        if push_proc.returncode != 0:
            sys.exit(f"git push origin {tag} failed: {push_proc.stderr}")

    ref = f"{owner}/{repo}@{head}"
    return ref


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="create_prereg_tag")
    p.add_argument(
        "--tag",
        default=f"prereg-{date.today().isoformat()}",
        help="Annotated tag name (default: prereg-YYYY-MM-DD).",
    )
    p.add_argument(
        "--message",
        default="Affect Battery study pre-registration",
        help="Tag annotation message.",
    )
    p.add_argument(
        "--sign", action="store_true",
        help="Sign the tag with GPG (requires gpg configured).",
    )
    p.add_argument(
        "--skip-push", action="store_true",
        help="Create the tag locally but don't push to origin.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ref = create_tag(args.tag, args.message, args.sign, args.skip_push)
    print()
    print(f"Pre-reg tag created: {args.tag}")
    print(f"Commit reference: {ref}")
    print()
    print("Use the following flag in your run invocation:")
    print(f"  --pre-registration-github-commit {ref}")
    print()
    print("Reviewers can verify the pre-reg methodology with:")
    print(f"  git show {args.tag}")
