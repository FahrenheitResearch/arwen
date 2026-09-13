"""Move every declaration of the release version in one step.

THE BREAKAGE THIS PREVENTS: the release number is declared in four files
that nothing ties together at edit time: ``pyproject.toml`` (the project
version and the ``gpuwm-data==`` pin), ``gpuwm-data/gpuwm_data/VERSION``
(the data package, which the engine requires at exactly its own version),
and the terminal crate's ``Cargo.toml`` and ``Cargo.lock``. Opening 2.7.4
by hand moved the first two and missed the terminal, so the release contract
set failed on ``test_terminal_and_python_release_versions_agree`` and the
candidate was not cuttable until a second commit. This tool rewrites all four
together, opens the changelog section for the new number, and ``--check``
reports any file that disagrees with ``pyproject.toml``.

Usage:
  python tools/release/bump_version.py --to 2.7.5    # rewrite, open the changelog section
  python tools/release/bump_version.py --check       # exit 1 naming any file out of step
Exit 0 = every declaration agrees (or was moved); 1 = a declaration disagrees; 2 = usage.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VERSION = re.compile(r"^\d+\.\d+\.\d+$")

#: Relative path, the pattern whose group 1 is the version, how many matches the file must carry.
DECLARATIONS = (
    ("pyproject.toml", re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.M), 1),
    ("pyproject.toml", re.compile(r'"gpuwm-data==(\d+\.\d+\.\d+)"'), 1),
    ("gpuwm-data/gpuwm_data/VERSION", re.compile(r"^(\d+\.\d+\.\d+)\s*$"), 1),
    ("tools/arwen-tui/Cargo.toml", re.compile(r'^version = "(\d+\.\d+\.\d+)"$', re.M), 1),
    ("tools/arwen-tui/Cargo.lock", re.compile(r'name = "arwen-tui"\nversion = "(\d+\.\d+\.\d+)"'), 1),
)

CHANGELOG = "CHANGELOG.md"


def declared(root: Path) -> list[tuple[str, str]]:
    """Every (file, version) declaration in the tree, in DECLARATIONS order."""
    found = []
    for relative, pattern, count in DECLARATIONS:
        text = (root / relative).read_text(encoding="utf-8")
        matches = pattern.findall(text)
        if len(matches) != count:
            raise SystemExit(f"bump_version: {relative} carries {len(matches)} version declarations, expected {count}")
        found.extend((relative, m) for m in matches)
    return found


def disagreements(root: Path) -> list[str]:
    found = declared(root)
    lead = found[0][1]
    return [f"{relative} says {version}, pyproject.toml says {lead}" for relative, version in found if version != lead]


def _rewrite(path: Path, pattern: re.Pattern, to: str) -> None:
    text = path.read_text(encoding="utf-8", newline="")

    def sub(match: re.Match) -> str:
        whole = match.group(0)
        start = match.start(1) - match.start(0)
        end = match.end(1) - match.start(0)
        return whole[:start] + to + whole[end:]

    path.write_text(pattern.sub(sub, text), encoding="utf-8", newline="")


def open_changelog(root: Path, to: str) -> bool:
    """Insert ``## <to> (unreleased)`` at the head unless a section for it exists."""
    path = root / CHANGELOG
    text = path.read_text(encoding="utf-8", newline="")
    if re.search(rf"^## {re.escape(to)}\b", text, re.M):
        return False
    newline = "\r\n" if "\r\n" in text else "\n"
    section = f"## {to} (unreleased){newline}{newline}New:{newline}{newline}Fixed:{newline}{newline}"
    head = re.match(r"# Changelog\r?\n\r?\n", text)
    if not head:
        raise SystemExit("bump_version: CHANGELOG.md does not start with '# Changelog'")
    path.write_text(text[: head.end()] + section + text[head.end():], encoding="utf-8", newline="")
    return True


def bump(root: Path, to: str) -> list[str]:
    if not VERSION.match(to):
        raise SystemExit(f"bump_version: --to wants MAJOR.MINOR.PATCH, got {to!r}")
    declared(root)  # refuses a tree whose declarations are not where this tool expects them
    touched = []
    for relative, pattern, _ in DECLARATIONS:
        _rewrite(root / relative, pattern, to)
        if relative not in touched:
            touched.append(relative)
    if open_changelog(root, to):
        touched.append(CHANGELOG)
    return touched


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--to", help="the version every declaration moves to")
    parser.add_argument("--check", action="store_true", help="report declarations that disagree with pyproject.toml")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    if bool(args.to) == args.check:
        parser.print_usage(sys.stderr)
        return 2
    root = args.root.resolve()
    if args.check:
        wrong = disagreements(root)
        for line in wrong:
            print("bump_version: " + line, file=sys.stderr)
        if wrong:
            return 1
        print(f"bump_version: every declaration says {declared(root)[0][1]}")
        return 0
    touched = bump(root, args.to)
    for relative in touched:
        print(f"bump_version: {relative} -> {args.to}")
    return 1 if disagreements(root) else 0


if __name__ == "__main__":
    raise SystemExit(main())
