"""Fail when a declared knob claims an implementation it cannot show.

The physics registry is the answer to "which of WRF's knobs does this model
have?".  That answer is only worth reading if a claim cannot be made by
assertion, so implementation is proven rather than asserted: a parameter
counts as implemented exactly when it cites ``consuming_read``, a
repo-relative file that actually reads it.

This gate re-resolves every citation against the tree:

* the cited file must exist and must parse, so an unreadable citation fails
  rather than passing unexamined;
* it must reference the knob in code -- prose does not count, and both of
  Python's prose channels are removed before the search: ``#`` comments and
  module/class/function docstrings;
* the knob must be a real GPUWM configuration field, so a citation cannot
  point at prose that merely mentions the name; and
* a parameter marked ``implemented: false`` must say why, and must not also
  carry a citation or a default that would leak into resolved settings.

Only those two channels are removed.  A string literal that is not a
docstring stays, because a read can legitimately go through one --
``getattr(cfg, "moist_cq", True)`` in ``gpuwm/core/acoustic.py`` is such a
read -- and blanking every string would fail real code for containing a
string.

Two properties matter beyond correctness.  A claim cannot outlive the code
that justified it -- delete the consuming read and this fails.  And each
scheme's owner controls only their own rows: a declaration with no citation
asserts nothing and passes, so turning a knob on means citing the read that
makes it true, never editing someone else's entry.

``tests/test_parameter_claim_gate.py`` is the negative control: it builds a
scratch tree whose cited module is nothing but a docstring naming the knob
and asserts this gate rejects it.

Usage
-----
    python tools/check_parameter_claims.py [<repo root>]
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import io
import re
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm.config import RunConfig  # noqa: E402
from gpuwm.physics_registry import physics_registry  # noqa: E402

#: Knobs GPUWM reads from the experiment schema rather than from RunConfig.
EXPERIMENT_SCHEMA_PARAMETERS = frozenset({"p_top", "blend_width", "co2_vmr"})

#: Nodes whose first statement may be a docstring.
_DOCSTRING_OWNERS = (
    ast.Module,
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


class UnparseableCitation(Exception):
    """The cited file is not source this gate can read a proof out of."""


def configuration_fields() -> frozenset[str]:
    return frozenset(
        {field.name for field in dataclasses.fields(RunConfig)}
        | set(EXPERIMENT_SCHEMA_PARAMETERS)
    )


def _blank_span(
    lines: list[str],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """Overwrite one source span with spaces, preserving every other offset.

    Blanking rather than deleting keeps the remaining spans addressable by
    their original (row, column), so several spans can be removed in any
    order without recomputing offsets.
    """

    start_row, start_column = start
    end_row, end_column = end
    for row in range(start_row, end_row + 1):
        line = lines[row - 1]
        first = start_column if row == start_row else 0
        last = min(end_column if row == end_row else len(line), len(line))
        if last > first:
            lines[row - 1] = line[:first] + " " * (last - first) + line[last:]


def _character_column(line: str, byte_column: int) -> int:
    """``ast`` reports columns as UTF-8 byte offsets; ``str`` indexes characters."""

    return len(line.encode("utf-8")[:byte_column].decode("utf-8", errors="ignore"))


def _python_code_without_prose(text: str) -> str:
    """Return ``text`` with comments and docstrings blanked out.

    A docstring is prose, so a knob named only in one proves nothing.  The
    docstring spans come from ``ast`` (which identifies them exactly, for
    both quote styles and at module, class and function scope) and the
    comment spans from ``tokenize`` (which, unlike splitting on ``#``, will
    not truncate a line at a ``#`` that lives inside a string).
    """

    lines = text.split("\n")
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as error:  # ValueError: embedded NUL
        raise UnparseableCitation(str(error)) from error

    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        head = body[0]
        if not isinstance(head, ast.Expr):
            continue
        literal = head.value
        if not (
            isinstance(literal, ast.Constant) and isinstance(literal.value, str)
        ):
            continue
        end_lineno = literal.end_lineno or literal.lineno
        end_col_offset = literal.end_col_offset or 0
        _blank_span(
            lines,
            (
                literal.lineno,
                _character_column(lines[literal.lineno - 1], literal.col_offset),
            ),
            (
                end_lineno,
                _character_column(lines[end_lineno - 1], end_col_offset),
            ),
        )

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as error:
        raise UnparseableCitation(str(error)) from error
    for token in tokens:
        if token.type == tokenize.COMMENT:
            _blank_span(lines, token.start, token.end)

    return "\n".join(lines)


def _code_without_prose(text: str, suffix: str) -> str:
    if suffix in {".py", ".pyi"}:
        return _python_code_without_prose(text)
    stripped = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return "\n".join(line.split("//", 1)[0] for line in stripped.splitlines())


def check(root: Path) -> list[str]:
    registry = physics_registry()
    parameters = registry.get("parameters", {})
    selectors: set[str] = set()
    for component in registry.get("components", {}).values():
        selectors |= set(component.get("selector_keys", ()))
    fields = configuration_fields()
    failures: list[str] = []

    for name, spec in sorted(parameters.items()):
        if not isinstance(spec, dict):
            failures.append(f"{name}: declaration is not an object")
            continue

        citation = spec.get("consuming_read")
        declared_unimplemented = spec.get("implemented") is False

        if declared_unimplemented:
            reason = spec.get("unimplemented_reason")
            if not (isinstance(reason, str) and reason.strip()):
                failures.append(
                    f"{name}: implemented=false must say why it is not implemented")
            if citation is not None:
                failures.append(
                    f"{name}: implemented=false must not cite a consuming read")
            if "default" in spec:
                failures.append(
                    f"{name}: implemented=false must not carry a default, which "
                    "would seed a resolved setting")
            continue

        if citation is None:
            # A declaration with no citation claims nothing. That is the state
            # every other declarer's rows are in, and it is not an error.
            continue

        if not isinstance(citation, str) or not citation.strip():
            failures.append(f"{name}: consuming_read must be a repo-relative path")
            continue
        if name in selectors:
            failures.append(
                f"{name}: selector keys are declared on their component, not as "
                "parameters")
        if name not in fields:
            failures.append(
                f"{name}: claims a consuming read but is not a GPUWM "
                "configuration field, so nothing can carry the value")
            continue

        path = root / citation
        if not path.is_file():
            failures.append(f"{name}: consuming_read {citation!r} does not exist")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            failures.append(f"{name}: consuming_read {citation!r} unreadable: {error}")
            continue
        try:
            code = _code_without_prose(text, path.suffix)
        except UnparseableCitation as error:
            failures.append(
                f"{name}: consuming_read {citation!r} does not parse, so no "
                f"read can be proven out of it: {error}")
            continue
        pattern = rf"\b{re.escape(name)}\b"
        if not re.search(pattern, code):
            if re.search(pattern, text):
                failures.append(
                    f"{name}: consuming_read {citation!r} names it only in a "
                    "comment or docstring; prose is not a read, so cite the "
                    "file whose code consumes the value")
            else:
                failures.append(
                    f"{name}: consuming_read {citation!r} does not reference it in "
                    "code; the claim has outlived the read that justified it")

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    failures = check(args.root.resolve())
    parameters = physics_registry().get("parameters", {})
    claimed = sum(
        1
        for spec in parameters.values()
        if isinstance(spec, dict)
        and spec.get("implemented") is not False
        and isinstance(spec.get("consuming_read"), str)
    )

    if failures:
        print(f"unproven knob claims ({len(failures)}):")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nA knob is implemented when a citation proves GPUWM reads it.\n"
            "Cite the file that reads it, or declare implemented=false with a "
            "reason."
        )
        return 1

    print(
        f"every knob claim resolves ({claimed} of {len(parameters)} declared "
        "parameters cite a consuming read)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
