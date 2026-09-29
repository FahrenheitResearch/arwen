"""The word rules for everything the page shows.

Every string in ``gui/copy/*.json`` and every string literal in the page's
scripts must:

- carry no marketing word (:data:`BANNED`): a page says what a button
  does, not how it feels;
- carry no long or medium dash, so the page reads the same as the
  project's other text;
- carry no percentage claim about quality ("95 percent accurate"): a
  number on screen is a measurement the engine reported, never a
  promise written into the copy.

``python -m gpuwm.gui.copy_lint [FILE ...]`` checks the named files, or
every copy file and page script when none is named; exit 1 lists each
problem.  Standard library only.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Iterator

GUI = Path(__file__).resolve().parent
#: Marketing words, as patterns over lower-case text, kept as patterns so
#: this module does not itself spell every word it looks for.
BANNED = (
    r"\bseamless", r"\bpowerful\b", r"\bblazing", r"\bstunning\b", r"\beffortless", r"\bmagic",
    r"\bsupercharg", r"\brevolutionary\b", r"\bcutting.edge\b", r"\bnext.gen", r"\bunleash",
    r"\blightning.fast\b", r"\bamazing\b", r"\bincredibl", r"\bbeautiful", r"\bworld.class\b",
    r"\bgame.chang", r"\bstate.of.the.art\b", r"\bbreathtaking\b", r"\bawesome\b", r"\bepic\b",
    r"\bultimate\b", r"\bbreakthrough", r"\bunprecedented\b", r"\bdelve", r"\brobust\b",
    r"\bleverag", r"\bempower",
)
DASHES = ("\u2014", "\u2013")
_CLAIM = re.compile(r"\b\d{2,3}(\.\d+)?\s*(%|percent)\s+(accura|correct|reliab|confiden|sure|certain)", re.I)
#: Fields of the copy files that are notes for whoever edits them, not words a page shows.
NOTE_FIELDS = frozenset({"about", "schema"})


def lint_text(text: str, *, where: str = "") -> list[str]:
    problems = []
    low = text.lower()
    for pattern in BANNED:
        found = re.search(pattern, low)
        if found:
            problems.append(f"{where}: marketing word {found.group(0)!r} in {text!r}")
    for dash in DASHES:
        if dash in text:
            problems.append(f"{where}: a long or medium dash in {text!r}; use a comma, colon or full stop")
    if _CLAIM.search(text):
        problems.append(f"{where}: a percentage claim in {text!r}")
    return problems


def _strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in NOTE_FIELDS:
                continue
            yield from _strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def lint_copy(path: Path) -> list[str]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for where, text in _strings(document, ""):
        problems += lint_text(text, where=f"{Path(path).name}:{where}")
    return problems


_JS_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'|`((?:[^`\\]|\\.)*)`', re.S)


def js_strings(source: str) -> Iterator[str]:
    """The literal text of every string in a script (template literals without their ${...} parts)."""

    body = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    body = "\n".join(re.sub(r"(^|\s)//.*$", "", line) for line in body.splitlines())
    for match in _JS_STRING.finditer(body):
        text = next(group for group in match.groups() if group is not None)
        text = re.sub(r"\$\{[^}]*\}", " ", text)
        yield text.encode("utf-8").decode("unicode_escape", "ignore") if "\\u" in text else text


def lint_js(path: Path) -> list[str]:
    source = Path(path).read_text(encoding="utf-8")
    problems = []
    for text in js_strings(source):
        problems += lint_text(text, where=Path(path).name)
    for dash in DASHES:
        if dash in source:
            problems.append(f"{Path(path).name}: a long or medium dash in the source")
    return problems


def page_files() -> list[Path]:
    return sorted((GUI / "copy").glob("*.json")) + sorted((GUI / "static").rglob("*.js")) \
        + sorted((GUI / "static").rglob("*.html"))


def lint_file(path: Path) -> list[str]:
    path = Path(path)
    if path.suffix == ".json":
        return lint_copy(path)
    if path.suffix == ".js":
        return lint_js(path)
    return lint_text(path.read_text(encoding="utf-8"), where=path.name)


def lint_all(paths: Iterable[Path] | None = None) -> list[str]:
    problems = []
    for path in (list(paths) if paths else page_files()):
        problems += lint_file(Path(path))
    return problems


def main(argv: list[str] | None = None) -> int:
    paths = [Path(arg) for arg in (sys.argv[1:] if argv is None else argv)]
    problems = lint_all(paths or None)
    for problem in problems:
        print(problem)
    if not problems:
        print(f"copy lint: {len(paths) or len(page_files())} file(s), no problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BANNED", "js_strings", "lint_all", "lint_copy", "lint_file", "lint_js", "lint_text", "page_files"]
