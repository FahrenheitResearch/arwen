"""Enumerate every device-code compile site in the tree.

A route is one construction path from Python source text to a compiled CUDA
module.  The receipt's per-route claims are only as complete as this
inventory, so the inventory is derived -- never typed -- from an AST walk over
``git ls-files``, and it records the option tuple each caller supplies rather
than a single global flags string.  There is no global flags string: the
sites below pass different tuples, and one of them (``cp.ReductionKernel``)
supplies no options at all.

Usage::

    python -m tools.ftz_receipt.route_inventory --check
    python -m tools.ftz_receipt.route_inventory --out <path>

``--check`` regenerates in memory and fails on any drift from the committed
``receipt/route_inventory.json``.
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

SCHEMA_ID = "gpuwm.ftz-route-inventory/v1"

#: Attribute/name tails that construct or invoke a device-code compile.  The
#: walk matches on the tail so ``cp.RawModule``, ``cupy.RawModule`` and a
#: bare ``RawModule`` imported by name are all one kind.
CONSTRUCTOR_KINDS = {
    "RawModule": "cupy.RawModule",
    "RawKernel": "cupy.RawKernel",
    "ElementwiseKernel": "cupy.ElementwiseKernel",
    "ReductionKernel": "cupy.ReductionKernel",
    "compile_using_nvrtc": "cupy.cuda.compiler.compile_using_nvrtc",
}

#: Where the caller-supplied option tuple lives per kind: (positional index or
#: None, keyword name or None).  ``ReductionKernel``/``ElementwiseKernel``
#: take no compile options from the caller at all.
OPTION_ARG = {
    "cupy.RawModule": (None, "options"),
    "cupy.RawKernel": (None, "options"),
    "cupy.cuda.compiler.compile_using_nvrtc": (1, "options"),
    "cupy.ElementwiseKernel": (None, None),
    "cupy.ReductionKernel": (None, None),
}


def repo_root(start: Path | None = None) -> Path:
    """Return the git top level containing ``start`` (default: this file)."""
    base = (start or Path(__file__).resolve().parent)
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         cwd=str(base), capture_output=True, text=True,
                         check=True)
    return Path(out.stdout.strip())


def tracked_python_files(root: Path) -> list[str]:
    """Every tracked ``*.py`` path, POSIX-relative, in git's own order."""
    out = subprocess.run(["git", "ls-files", "--", "*.py"], cwd=str(root),
                         capture_output=True, text=True, check=True)
    return sorted(p for p in out.stdout.splitlines() if p.strip())


def _callee_tail(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return "<expr>"


def _option_argument(call: ast.Call, kind: str) -> ast.AST | None:
    index, keyword = OPTION_ARG[kind]
    if keyword is not None:
        for kw in call.keywords:
            if kw.arg == keyword:
                return kw.value
    if index is not None and len(call.args) > index:
        return call.args[index]
    return None


def _literal_options(node: ast.AST | None,
                     text: str) -> tuple[list[str | None] | None,
                                         list[str] | None]:
    """Split an options argument into per-element literals and expressions.

    A dynamic element (``f"-arch=compute_{arch}"`` at
    ``gpuwm/core/rrtmg_lw.py``) becomes ``None`` in the value list while its
    source text is kept, so a consumer can bind the static elements and
    supply the dynamic one the same way the production site does.  Returns
    ``(None, None)`` when the argument is not a sequence display at all.
    """
    if node is None:
        return None, None
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None, None
    values: list[str | None] = []
    expressions: list[str] = []
    for element in node.elts:
        expressions.append(ast.get_source_segment(text, element) or "")
        try:
            value = ast.literal_eval(element)
        except (ValueError, TypeError, SyntaxError):
            values.append(None)
            continue
        values.append(value if isinstance(value, str) else None)
    return values, expressions


def _surface(path: str) -> str:
    """Which surface a site lives on: the shipped package, tests, or tools."""
    head = path.split("/", 1)[0]
    if head == "gpuwm":
        return "package"
    if head in ("tests", "tools"):
        return head
    return "other"


def _enclosing(tree: ast.AST) -> dict[int, str]:
    """Map every line to the innermost def/class name that contains it."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            for line in range(node.lineno, (end or node.lineno) + 1):
                owner[line] = node.name
    return owner


def scan_source(path: str, text: str) -> list[dict]:
    """Return one record per compile site in ``text``."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    owner = _enclosing(tree)
    lines = text.splitlines()
    records: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        tail = _callee_tail(node.func)
        kind = CONSTRUCTOR_KINDS.get(tail or "")
        if kind is None:
            continue
        option_node = _option_argument(node, kind)
        options, option_elements = _literal_options(option_node, text)
        if option_node is None:
            option_expr = None
        else:
            option_expr = ast.get_source_segment(text, option_node)
        records.append({
            "file": path,
            "line": node.lineno,
            "column": node.col_offset,
            "surface": _surface(path),
            "constructor": _dotted(node.func),
            "constructor_kind": kind,
            "enclosing": owner.get(node.lineno),
            "takes_caller_options": OPTION_ARG[kind] != (None, None),
            "options_argument_present": option_node is not None,
            "options_expression": option_expr,
            "options": options,
            "options_element_expressions": option_elements,
            "options_are_literal": options is not None and all(
                item is not None for item in options),
            "text": lines[node.lineno - 1].strip() if node.lineno <= len(lines)
                    else "",
        })
    records.sort(key=lambda r: (r["file"], r["line"], r["column"]))
    return records


def build_inventory(root: Path) -> dict:
    """Walk the tracked Python tree and return the inventory document."""
    records: list[dict] = []
    for rel in tracked_python_files(root):
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        records.extend(scan_source(rel, text))
    records.sort(key=lambda r: (r["file"], r["line"], r["column"]))
    by_kind: dict[str, int] = {}
    by_surface: dict[str, dict[str, int]] = {}
    for record in records:
        by_kind[record["constructor_kind"]] = (
            by_kind.get(record["constructor_kind"], 0) + 1)
        bucket = by_surface.setdefault(record["surface"], {})
        bucket[record["constructor_kind"]] = (
            bucket.get(record["constructor_kind"], 0) + 1)
    distinct = sorted({
        json.dumps(r["options"]) for r in records
        if r["options_are_literal"]})
    return {
        "schema": SCHEMA_ID,
        "generator": "tools/ftz_receipt/route_inventory.py",
        "scope": "git ls-files -- *.py",
        "constructor_kinds": sorted(set(CONSTRUCTOR_KINDS.values())),
        "site_count": len(records),
        "site_count_by_kind": dict(sorted(by_kind.items())),
        "site_count_by_surface": {
            surface: dict(sorted(counts.items()))
            for surface, counts in sorted(by_surface.items())},
        "distinct_literal_option_tuples": [
            json.loads(item) for item in distinct],
        "sites": records,
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def default_output(root: Path) -> Path:
    return root / "tools" / "ftz_receipt" / "receipt" / "route_inventory.json"


def load_inventory(root: Path | None = None) -> dict:
    """Read the committed inventory."""
    root = root or repo_root()
    return json.loads(default_output(root).read_text(encoding="utf-8"))


def find_site(inventory: dict, *, file: str, constructor_kind: str,
              enclosing: str | None = None) -> dict:
    """Return the one site matching the triple, or raise.

    Consumers address a production site by what it IS rather than by a line
    number, so an unrelated edit above it does not silently repoint them at
    a different compile.
    """
    matches = [site for site in inventory["sites"]
               if site["file"] == file
               and site["constructor_kind"] == constructor_kind
               and (enclosing is None or site["enclosing"] == enclosing)]
    if len(matches) != 1:
        raise LookupError(
            f"expected exactly one {constructor_kind} site in {file}"
            + (f" within {enclosing}" if enclosing else "")
            + f"; found {len(matches)}")
    return matches[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None,
                        help="repository root (default: this file's git top "
                             "level)")
    parser.add_argument("--out", default=None,
                        help="output path (default: the committed receipt)")
    parser.add_argument("--check", action="store_true",
                        help="compare against the committed receipt and exit "
                             "non-zero on drift")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else repo_root()
    document = build_inventory(root)
    text = render(document)
    out = Path(args.out) if args.out else default_output(root)
    if args.check:
        if not out.exists():
            print(f"route inventory missing: {out}", file=sys.stderr)
            return 1
        current = out.read_text(encoding="utf-8")
        if current != text:
            print(f"route inventory out of date: {out}", file=sys.stderr)
            return 1
        print(f"route inventory current: {document['site_count']} sites")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out} ({document['site_count']} sites)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
