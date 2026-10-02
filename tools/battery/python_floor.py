"""The oldest Python the package declares, and the syntax that Python cannot read.

THE BREAKAGE THIS PREVENTS.  The public CI's syntax job compiles every Python
file of the tree under Python 3.11, the floor ``requires-python`` declares, and
it failed on 2.7.6, 2.7.7 and 2.8.0: five vendored copies of the libc crate's
``etc/libc-util.py`` reuse the enclosing quote inside an f-string replacement
field, which only Python 3.12 (PEP 701) reads.  Nothing in this tree ran under
3.11, so nothing saw it before a tag did.

``python tools/battery/python_floor.py`` prints ``version=X.Y`` for the job to
install, so the job follows pyproject.toml instead of repeating it.
:func:`newer_syntax` is what tests/test_public_ci_contract.py runs over the
same files on whatever interpreter the battery has: on the floor itself it is
``compile``, and on a newer one it is ``ast.parse(feature_version=floor)``
plus a token scan for the PEP 701 forms, which ``feature_version`` does not
gate (measured on 3.13: ``f"{", ".join(x)}"`` parses with
``feature_version=(3, 11)``).
"""
from __future__ import annotations

import ast
import io
import pathlib
import re
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: What the syntax job compiles, as .github/workflows/ci.yml names it.
SYNTAX_JOB_ROOTS = ("gpuwm", "tilestream", "tools", "tests", "conftest.py")


def declared_floor(root: pathlib.Path = ROOT) -> tuple[int, int]:
    """``(major, minor)`` of the ``>=`` clause of ``[project].requires-python``.

    A specifier without a ``>=`` lower bound is refused: the syntax job would
    not know which grammar the package promises to be read by.
    """
    import tomllib

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    spec = data["project"]["requires-python"]
    for clause in spec.split(","):
        match = re.fullmatch(r"\s*>=\s*(\d+)\.(\d+)(?:\.\d+)?\s*", clause)
        if match:
            return int(match.group(1)), int(match.group(2))
    raise ValueError(f"requires-python {spec!r} declares no >= lower bound")


def _pep701_forms(source: str) -> list[tuple[int, str]]:
    """F-string forms that need Python 3.12, found by the 3.12+ tokenizer.

    Before 3.12 an f-string's expression part could not hold the quote that
    delimits any enclosing f-string, a backslash, a comment, or (inside a
    single-quoted f-string) a line break.
    """
    found: list[tuple[int, str]] = []
    enclosing: list[tuple[str, bool]] = []      # (quote, triple) per open f-string

    def closes_an_enclosing(delimiter: str) -> bool:
        # Pre-3.12 the tokenizer ended an f-string at its own delimiter
        # before the expression part was read.
        return any(delimiter.startswith(quote * 3) if triple else delimiter[0] == quote
                   for quote, triple in enclosing)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return found
    for token in tokens:
        kind, text, (line, _column) = token.type, token.string, token.start
        if kind == tokenize.FSTRING_START:
            quote = text.lstrip("rRbBfFuU")
            if enclosing and closes_an_enclosing(quote):
                found.append((line, "an f-string nested in one that uses the same quote"))
            enclosing.append((quote[0], len(quote) == 3))
            continue
        if kind == tokenize.FSTRING_END:
            if enclosing:
                enclosing.pop()
            continue
        if not enclosing:
            continue
        if kind == tokenize.FSTRING_MIDDLE:
            if len(enclosing) > 1 and "\\" in text:
                found.append((line, "a backslash inside an f-string's expression part"))
            continue
        if kind == tokenize.STRING:
            if closes_an_enclosing(text.lstrip("rRbBuU")):
                found.append((line, "the enclosing f-string's quote reused inside its expression part"))
            if "\\" in text:
                found.append((line, "a backslash inside an f-string's expression part"))
        elif kind == tokenize.COMMENT:
            found.append((line, "a comment inside an f-string's expression part"))
        elif kind == tokenize.NL and not all(triple for _, triple in enclosing):
            found.append((line, "a line break inside a single-quoted f-string's expression part"))
    return found


def newer_syntax(source: str, floor: tuple[int, int], name: str = "<source>") -> list[str]:
    """Why ``floor`` cannot read ``source``; empty when it can."""
    if sys.version_info[:2] == floor:
        try:
            compile(source, name, "exec", dont_inherit=True)
        except SyntaxError as error:
            return [f"{name}:{error.lineno}: {error.msg}"]
        return []
    if sys.version_info[:2] < floor:
        raise RuntimeError(f"this interpreter predates the declared floor {floor}")
    problems: list[str] = []
    try:
        ast.parse(source, filename=name, feature_version=floor)
    except SyntaxError as error:
        problems.append(f"{name}:{error.lineno}: {error.msg}")
    if floor < (3, 12):
        problems += [f"{name}:{line}: {what}" for line, what in _pep701_forms(source)]
    return problems


def syntax_job_files(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    """Every .py file the syntax job compiles."""
    files: list[pathlib.Path] = []
    for entry in SYNTAX_JOB_ROOTS:
        path = root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def main() -> int:
    major, minor = declared_floor()
    print(f"version={major}.{minor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
