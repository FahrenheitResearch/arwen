"""List the CHANGELOG rows a release section gained on the integration line after it was condensed.

The 2.8.0 section was condensed from integrate/2.8, and ``BASE`` (below) is the last integrate/2.8
commit whose rows are folded into it.  Rows merged into integrate/2.8 after that commit are not in the
condensed section, so at the cut they are folded in by hand: this lists them, under the heading and
area they were written under, so the refresh takes minutes.  Each one goes into the condensed section
in a user's words, and word for word under "Rows added after the section was condensed" in
docs/dev/changelog-2.8.0-full.md; then ``BASE`` moves to the commit folded in.  It reads the release's section of CHANGELOG.md at ``BASE`` and at ``REF`` and prints

- every row at REF that is not at BASE: a new row, or a row reworded since;
- every row at BASE that is not at REF, so a reworded row is seen from both sides;
- the word count of this tree's section, and what it would be with the new rows as written, against
  the limit tests/test_release_notes_are_public_facing.py holds the newest section to (that release's
  entry in its RELEASE_WORD_ALLOWANCE, 7,000 words for 2.8.0, and MAX_WORDS for every other release).

Each new row is printed as the Weather Library rename (tools/dev/rename_weather_library.py) would
word it, since integrate/2.8 still carries the old name; ``--raw`` prints it as written.

    python tools/dev/changelog_new_rows.py                      rows added on integrate/2.8 since BASE
    python tools/dev/changelog_new_rows.py COMMIT               since another base
    python tools/dev/changelog_new_rows.py --ref BRANCH         on another line
    python tools/dev/changelog_new_rows.py --json               the same, as JSON

Run it before the condensed section is merged into integrate/2.8.  After that merge the section at
REF is the condensed one: pass the merge commit as the base, and the rows added after it are listed.

Exit 0 when it read both sections, 2 when a ref or the section cannot be read.  Standard library only.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
#: The last integrate/2.8 commit whose 2.8.0 rows are folded into the condensed section: the section
#: was condensed from f3842587c, 953d69f3d's one new row (boundary cadence) was folded in after, and
#: the 66 rows 6eba59eee added after that (chained preparation, the terrain time step, ICON-D2's model
#: levels and the rest) and e9574164c's one (the card probe on a machine with several cards) were folded
#: in at the 7,000-word rebuild, and the 31 d5b0b1a6b added after that (the one-minute ASOS reader, the
#: fixed-order LETKF, the saturation bound, the radial-velocity dispersion gate, the renderer's memory
#: sizing and the rest) were folded in at 6,598 words, and the 14 1a9b24134 added or reworded after that
#: (container memory limits for the tile planner and mesh build, the delayed nest's single build,
#: preparation steps in the terminal, transport auto and the rest) were folded in at the trim to
#: 6,590 words, and 6dce23fbe's one (the pinned host store under a memory limit) at 6,589, and the four
#: 846e37d7b added after that (the mp=28 cold start's droplet number, preparation steps on the run page on
#: the staged and plain HRRR and GFS routes, and transport auto's plan note) at 6,735, and the four
#: f43ef142a added or reworded after that (the preparation priced before the card on every door, the
#: Thompson rain and ice starting numbers, and auto's and RW-WPS's priced memory reading) at the trim
#: to 6,430, and the twelve b47973260 added or reworded after that (the adaptive clock default, the
#: Thompson + MYNN + RUC suites, ECMWF and GEM retention, the early CUDA preparation refusal, the
#: memory rows, the download retries, the single-domain HRRR steps, the renderer's Times fixes and the
#: Thompson numbers in the stock WRF files) at 6,755.
BASE = "b0fa06816"
REF = "integrate/2.8"
VERSION = "2.8.0"
CHANGELOG = "CHANGELOG.md"
LIMIT_TEST = "tests/test_release_notes_are_public_facing.py"

_GROUP = re.compile(r"^([A-Z][A-Za-z ]*):\s*$")
_AREA = re.compile(r"^\*\*(.+?)\*\*\s*$")


@dataclass(frozen=True)
class Row:
    group: str   # New, Changed defaults, Fixed, Known limits ...
    area: str    # the bold heading above it, or ""
    text: str

    @property
    def key(self) -> str:
        return " ".join(self.text.split())

    @property
    def words(self) -> int:
        return len(self.text.split())


def section(text: str, version: str) -> str | None:
    """``## <version>`` and its body up to the next ``## `` heading, or None; the heading counts as
    words, as the release-notes test counts them."""

    head = re.search(rf"^## {re.escape(version)}\b.*$", text, re.M)
    if head is None:
        return None
    following = re.search(r"^## ", text[head.end():], re.M)
    return text[head.start():head.end() + following.start()] if following else text[head.start():]


def rows(body: str) -> list[Row]:
    """Every ``- `` row of a section, with its continuation lines, under its group and area."""

    found: list[Row] = []
    group = area = ""
    current: list[str] | None = None

    def close() -> None:
        nonlocal current
        if current:
            found.append(Row(group, area, "\n".join(current).rstrip()))
        current = None

    for line in body.splitlines():
        if line.startswith("- "):
            close()
            current = [line]
        elif current is not None and line.strip() and line[:1].isspace():
            current.append(line)
        else:
            close()
            if _GROUP.match(line):
                group, area = _GROUP.match(line).group(1), ""
            elif _AREA.match(line):
                area = _AREA.match(line).group(1)
    close()
    return found


def git_text(root: Path, ref: str, path: str) -> str:
    done = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=str(root), capture_output=True)
    if done.returncode != 0:
        raise LookupError(f"cannot read {path} at {ref}: {done.stderr.decode('utf-8', 'replace').strip()}")
    return done.stdout.decode("utf-8")


def resolve(root: Path, ref: str) -> str:
    done = subprocess.run(["git", "rev-parse", "--short=12", f"{ref}^{{commit}}"], cwd=str(root),
                          capture_output=True, text=True)
    if done.returncode != 0:
        raise LookupError(f"no commit {ref}: {done.stderr.strip()}")
    return done.stdout.strip()


def word_limit(root: Path, version: str = VERSION) -> int | None:
    """The word limit the release-notes test holds ``version``'s section to: its entry in
    ``RELEASE_WORD_ALLOWANCE`` when it has one, ``MAX_WORDS`` otherwise; None without the test."""

    try:
        text = (root / LIMIT_TEST).read_text(encoding="utf-8")
    except OSError:
        return None
    allowance = re.search(r"^RELEASE_WORD_ALLOWANCE\b[^=\n]*=\s*(\{[^}]*\})", text, re.M)
    if allowance is not None:
        try:
            given = ast.literal_eval(allowance.group(1))
        except (SyntaxError, ValueError):
            given = {}
        if isinstance(given, dict) and isinstance(given.get(version), int):
            return given[version]
    found = re.search(r"^MAX_WORDS\s*=\s*(\d+)", text, re.M)
    return int(found.group(1)) if found else None


def _renamer():
    """``text -> text`` worded as the Weather Library rename words release notes, or None without the tool."""

    path = Path(__file__).resolve().parent / "rename_weather_library.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("rename_weather_library", path)
    tool = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, tool)
    spec.loader.exec_module(tool)
    notes = tool.Scope("notes", True, True)
    return lambda text: tool.rename_text(text, notes, tool.NOTES_EDITS, None, strict=False).text


def compare(root: Path, base: str, ref: str, version: str, *, raw: bool = False) -> dict:
    before_body = section(git_text(root, base, CHANGELOG), version)
    after_body = section(git_text(root, ref, CHANGELOG), version)
    for body, where in ((before_body, base), (after_body, ref)):
        if body is None:
            raise LookupError(f"{CHANGELOG} at {where} has no ## {version} section")
    before, after = rows(before_body), rows(after_body)
    known_before = {row.key for row in before}
    known_after = {row.key for row in after}
    added = [row for row in after if row.key not in known_before]
    removed = [row for row in before if row.key not in known_after]
    rename = None if raw else _renamer()
    if rename is not None:
        added = [Row(row.group, row.area, rename(row.text)) for row in added]
    try:
        here = section((root / CHANGELOG).read_text(encoding="utf-8"), version)
    except OSError:
        here = None
    tree_words = len(here.split()) if here is not None else None
    added_words = sum(row.words for row in added)
    return {
        "version": version,
        "base": base, "base_commit": resolve(root, base),
        "ref": ref, "ref_commit": resolve(root, ref),
        "reworded_as_weather_library": rename is not None,
        "added": [asdict(row) | {"words": row.words} for row in added],
        "removed": [asdict(row) | {"words": row.words} for row in removed],
        "tree_section_words": tree_words,
        "tree_section_words_with_added": None if tree_words is None else tree_words + added_words,
        "word_limit": word_limit(root, version),
    }


def _where(row: dict) -> str:
    return row["group"] + (f" / {row['area']}" if row["area"] else "")


def report(found: dict) -> str:
    lines = [f"{CHANGELOG} ## {found['version']}: {found['ref']} at {found['ref_commit']} "
             f"against the base {found['base']} ({found['base_commit']})."]
    if not found["added"] and not found["removed"]:
        lines.append(f"No row was added, reworded or removed on {found['ref']} since the base.")
    if found["added"]:
        note = ", worded as the Weather Library rename words it" if found["reworded_as_weather_library"] else ""
        lines.append("")
        lines.append(f"{len(found['added'])} rows at {found['ref']} that the base does not have "
                     f"(new or reworded{note}):")
        where = None
        for row in found["added"]:
            if _where(row) != where:
                where = _where(row)
                lines += ["", f"[{where}]"]
            lines.append(row["text"])
    if found["removed"]:
        lines.append("")
        lines.append(f"{len(found['removed'])} rows of the base that {found['ref']} no longer has "
                     "(removed, or reworded above):")
        for row in found["removed"]:
            lines += ["", f"[{_where(row)}]", row["text"]]
    if found["tree_section_words"] is not None:
        limit = found["word_limit"]
        lines.append("")
        lines.append(f"This tree's ## {found['version']} section: {found['tree_section_words']} words; "
                     f"with the new rows as written, {found['tree_section_words_with_added']}"
                     + (f" (limit {limit})." if limit else "."))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the CHANGELOG rows a release section gained on the "
                                                 "integration line since it was condensed.")
    parser.add_argument("base", nargs="?", default=BASE, help=f"the commit the section was condensed from "
                                                              f"(default {BASE})")
    parser.add_argument("--ref", default=REF, help=f"the line whose new rows to list (default {REF})")
    parser.add_argument("--version", default=VERSION, help=f"the release section (default {VERSION})")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout (default: this one)")
    parser.add_argument("--raw", action="store_true", help="print rows as written, without the rename's wording")
    parser.add_argument("--json", action="store_true", help="print JSON")
    arguments = parser.parse_args(argv)
    try:
        found = compare(arguments.root.resolve(), arguments.base, arguments.ref, arguments.version,
                        raw=arguments.raw)
    except LookupError as error:
        print(f"changelog_new_rows: {error}", file=sys.stderr)
        return 2
    print(json.dumps(found, indent=1) if arguments.json else report(found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
