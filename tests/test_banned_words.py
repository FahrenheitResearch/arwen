"""Two words the project does not use stay out of its first-party text.

The 2.7.5 sweep took them out of the prose, the comments, the docstrings
and the identifiers of every first-party tree, one reading each.  What
it could not take them out of is named in ``tests/banned_word_allowlist.tsv``
with the reason, and every reason is a pin: a file whose BYTES are an
anchor (a frozen kernel digest, a fixture receipt, a prepared-tree
run-control document), a dated record, or an emitted receipt key that
readers of earlier receipts depend on.

The sweep missed occurrences twice before it had a gate -- the second
pass found twelve in eight files that the first had not opened -- so
the gate is the sweep's own walk:

* an occurrence anywhere outside the allowlist fails, naming the line;
* an allowlisted path must exist and must carry EXACTLY the count the
  allowlist records, so a carve-out cannot outlive its subject and a new
  occurrence inside a carved-out file cannot hide behind the old ones.

This file spells neither word, so it passes its own scan.
"""
from __future__ import annotations

import functools
from pathlib import Path
import re

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = REPOSITORY_ROOT / "tests" / "banned_word_allowlist.tsv"

#: The two words, in every inflection and spelling the tree has used,
#: assembled so that this module does not carry them.
BANNED = re.compile("(?i)" + "hon" + "est" + "|" + "load" + "[- ]bearing")

SCANNED_ROOTS = ("gpuwm", "docs", "configs", "tests", "tools", "tilestream")
ROOT_FILES = ("README.md", "AGENTS.md", "CONTRIBUTING.md", "RELEASE_CHECKLIST.md",
              "SPEEDRUN.md", "MP28_HANDOFF.md", "MP28_PORT_SPEC.md",
              "PROVENANCE.md", "pyproject.toml", "MPAS-SEAM-CONTRACT.md",
              "PERF-SURVEY.md", "HANDOFF.md", "CHANGELOG.md")
#: Somebody else's words, or build products: never scanned.
SKIPPED_DIRECTORY_NAMES = {
    "vendor", "arwen-ui-vendor", "crates-io", "target", "node_modules",
    "__pycache__", ".git", ".pytest_cache", "mutants.out",
}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".rs", ".cu", ".cuh", ".txt",
                 ".sh", ".yaml", ".yml", ".cfg", ".h", ".c", ".ps1", ".bat",
                 ".rst", ".ini", ".csv", ".tsv", ".html", ".css", ".js",
                 ".jsonl", ".wps", ".input", ".namelist", ".F90", ".f90",
                 ".F", ".cmd", ".mk", ".in", ""}


def first_party_text_files() -> list[Path]:
    found: list[Path] = []
    for name in ROOT_FILES:
        path = REPOSITORY_ROOT / name
        if path.is_file():
            found.append(path)
    for root_name in SCANNED_ROOTS:
        root = REPOSITORY_ROOT / root_name
        if not root.is_dir():
            continue
        stack = [root]
        while stack:
            directory = stack.pop()
            for entry in sorted(directory.iterdir()):
                if entry.is_dir():
                    if entry.name not in SKIPPED_DIRECTORY_NAMES:
                        stack.append(entry)
                elif entry.suffix in TEXT_SUFFIXES:
                    found.append(entry)
    return found


def occurrences(path: Path) -> list[tuple[int, str]]:
    """(line number, line) for every occurrence in one file; [] if binary."""

    try:
        raw = path.read_bytes()
    except OSError:
        return []
    if b"\x00" in raw[:4096]:
        return []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return []
    if not BANNED.search(text):
        return []
    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        for _ in BANNED.finditer(line):
            out.append((number, line.strip()[:160]))
    return out


@functools.lru_cache(maxsize=1)
def scan() -> dict[str, list[tuple[int, str]]]:
    """relative path -> occurrences, for every first-party file carrying one.

    Cached: one walk of the tree serves every case in this module.
    """

    here = Path(__file__).resolve()
    found: dict[str, list[tuple[int, str]]] = {}
    for path in first_party_text_files():
        if path.resolve() == here or path.resolve() == ALLOWLIST.resolve():
            continue
        hits = occurrences(path)
        if hits:
            found[path.relative_to(REPOSITORY_ROOT).as_posix()] = hits
    return found


def read_allowlist() -> list[tuple[str, int, str]]:
    """(path or directory prefix, recorded count, reason) per row."""

    rows = []
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, count, reason = line.split("\t", 2)
        rows.append((path, int(count), reason.strip()))
    return rows


def _covered_by(relative: str, entry: str) -> bool:
    return relative == entry or (entry.endswith("/") and relative.startswith(entry))


def test_no_first_party_text_carries_either_word_outside_the_allowlist():
    found = scan()
    entries = [path for path, _, _ in read_allowlist()]
    stray = []
    for relative, hits in sorted(found.items()):
        if any(_covered_by(relative, entry) for entry in entries):
            continue
        for number, line in hits:
            stray.append(f"{relative}:{number}: {line}")
    assert not stray, (
        "first-party text carries a word the project does not use; reword "
        "it (one reading each: accurate/accuracy, essential), or if the "
        "file's bytes are a pin, add it to tests/banned_word_allowlist.tsv "
        "with the count and the pin that holds it:\n  " + "\n  ".join(stray))


@pytest.mark.parametrize("entry,count,reason",
                         read_allowlist(),
                         ids=[row[0] for row in read_allowlist()])
def test_every_allowlisted_path_still_carries_exactly_what_it_recorded(
        entry, count, reason):
    """A carve-out names its pin, exists, and has not grown or emptied."""

    assert reason, f"{entry} is allowlisted with no reason"
    target = REPOSITORY_ROOT / entry
    assert target.exists(), (
        f"{entry} is allowlisted and does not exist; delete the row")
    found = scan()
    carried = sum(len(hits) for relative, hits in found.items()
                  if _covered_by(relative, entry))
    assert carried == count, (
        f"{entry} carries {carried} occurrence(s) and the allowlist "
        f"records {count}: a new one has appeared (reword it) or the pin "
        "has lost its subject (re-record the count, or delete the row)")


def test_the_allowlist_names_no_path_twice():
    entries = [path for path, _, _ in read_allowlist()]
    assert len(entries) == len(set(entries)), entries
