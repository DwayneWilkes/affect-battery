"""Every file the README names must exist in the repository.

A dead reference in a public README is a small lie that compounds: a reader who
cannot find a cited file stops trusting the ones they can find. This module
collects every relative link target and every file-like code span in README.md
and checks each against ``git ls-files``.

If a check here fires on an honest edit, loosen it and say why in the commit
body rather than deleting it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:")
_CODE_SPAN_RE = re.compile(r"`([^`]+)`")
_FILE_TOKEN_RE = re.compile(r"^[\w./-]+\.(md|py|yaml|yml|json|pdf|sh|toml)$")
_TOKEN_EXCLUSIONS = ("*", "<", ",")


def strip_fenced(text: str) -> str:
    """Drop the lines between triple-backtick fences, the fence lines included."""
    kept = []
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        kept.append(line)
    return "\n".join(kept)


def relative_link_targets(text: str) -> list[str]:
    """Markdown link targets that are neither absolute URIs nor bare fragments."""
    targets = []
    for match in _LINK_RE.finditer(text):
        target = match.group(1)
        if _SCHEME_RE.match(target) or target.startswith("#"):
            continue
        targets.append(target.split("#", 1)[0])
    return targets


def file_tokens(text: str) -> list[str]:
    """Inline code spans that name a file."""
    tokens = []
    for match in _CODE_SPAN_RE.finditer(text):
        token = match.group(1)
        if any(bad in token for bad in _TOKEN_EXCLUSIONS):
            continue
        if _FILE_TOKEN_RE.match(token):
            tokens.append(token)
    return tokens


def resolves(target: str, tracked: frozenset[str]) -> bool:
    """A target resolves when it is tracked, or, ending in ``/``, prefixes one."""
    if target in tracked:
        return True
    if target.endswith("/"):
        return any(path.startswith(target) for path in tracked)
    return False


def check_references(text: str, tracked: frozenset[str]) -> list[str]:
    """Every reference in ``text`` that does not resolve."""
    stripped = strip_fenced(text)
    references = relative_link_targets(stripped) + file_tokens(stripped)
    return [ref for ref in references if not resolves(ref, tracked)]


def tracked_files() -> frozenset[str]:
    """The tracked paths at the repository root. Raises on failure; never skips."""
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(completed.stdout.splitlines())


TRACKED = {"docs/preregistrations/h3b_2026-05-07.md", "README.md"}


def test_collectors_ignore_fences_and_non_references():
    text = (
        "See [the guide](docs/guide.md) and `src/cli.py`.\n"
        "```bash\n[hidden](docs/hidden.md) and `src/hidden.py`\n```\n"
        "Not files: `results/pilot`, `a*.py`, `path.yaml::symbol`, "
        "[abs](https://example.com), [frag](#section).\n"
        "Fragment stripped: [x](docs/guide.md#heading).\n"
    )
    assert relative_link_targets(strip_fenced(text)) == [
        "docs/guide.md",
        "docs/guide.md",
    ]
    assert file_tokens(strip_fenced(text)) == ["src/cli.py"]


def test_resolves_exact_match_and_directory_prefix():
    assert resolves("docs/preregistrations/h3b_2026-05-07.md", frozenset(TRACKED))
    assert resolves("docs/preregistrations/", frozenset(TRACKED))
    assert not resolves("docs/preregistrations/missing.md", frozenset(TRACKED))


def test_check_references_names_what_does_not_resolve():
    text = "`The_Affect_Battery.pdf` and `results/probes/missing.json` and `README.md`"
    assert check_references(text, frozenset(TRACKED)) == [
        "The_Affect_Battery.pdf",
        "results/probes/missing.json",
    ]


def test_tracked_files_raises_rather_than_skipping(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(FileNotFoundError):
        tracked_files()


def test_every_readme_reference_resolves():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    tracked = tracked_files()
    stripped = strip_fenced(readme)
    collected = relative_link_targets(stripped) + file_tokens(stripped)
    assert collected, "the README names no files; the guard would pass vacuously"
    unresolved = check_references(readme, tracked)
    assert not unresolved, f"README names files that are not tracked: {unresolved}"
