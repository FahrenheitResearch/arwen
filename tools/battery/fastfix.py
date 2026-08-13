"""Pick the tests a change can actually break, and price them.

    python tools/battery/fastfix.py --base v2.1.0
    python tools/battery/fastfix.py --base <sha> --head <sha> --format pytest

This is Lane A2 of the fast-fix lane (see ``tools/battery/FASTFIX.md``): the
step that turns "these files changed" into "run these suites", so a defect
can be fixed and published without waiting on the full battery.

THE ONE DESIGN DECISION, AND THE MEASUREMENT BEHIND IT
-----------------------------------------------------
Selection is by **direct imports only**.  A test file is selected when its
own source imports a module that a touched file defines.  No transitive
closure.

That is not a shortcut, it is the measured answer.  The 2026-08-13
test-estate audit prototyped both and reported:

* transitive closure: mean closure 186 modules, and touching
  ``gpuwm/core/moist.py``, ``gpuwm/fetch.py``, ``gpuwm/experiment.py`` or
  ``tilestream/driver.py`` selects **441 of 591 test files (75%)**.  A
  selector that selects three quarters of the estate is not a selector.
* depth 0: mean closure 5.6 modules, 1% to 12% of the estate per touched
  product file.

and then validated both against a real defect rather than trusting either.
Commit ``6e9c690f0`` touched ``gpuwm/core/moist.py`` and
``gpuwm/core/dycore.py`` and broke two things.  Depth 0 selects 77 files and
**hits ``tests/test_pd_advection.py``**, the test that catches it.  Depth 1
selects 186 and depth 2 selects 276 **for no additional catch**.  Three
times the cost, same answer.  ``tests/test_fastfix_selector.py`` pins that
commit as a regression so the setting cannot drift on a hunch.

THE LIMIT, WHICH IS WHY THE ALWAYS LIST EXISTS
----------------------------------------------
The same validation found what this cannot see.  ``6e9c690f0`` also left
``tools/ftz_receipt/receipt/route_inventory.json`` stale, and
``tests/test_ftz_route_inventory.py`` is a MISS at **every** depth, because
it reads files rather than importing modules.  Citation checkers,
line-ending gates, release-note gates, case-token leakage and receipt
regenerators have no import edge to anything.

They are therefore never selected.  ``tools/battery/always_files.txt`` runs
unconditionally in every lane, and a non-Python change (a ``.cu``, a
``.toml``, a receipt) selects the ALWAYS list *and nothing else*, because
import analysis has nothing to say about those files and pretending
otherwise would be worse than admitting it.

OUTPUT
------
The selected files, **sorted cheapest first** so a red arrives early, each
with the reason it was selected, and an estimated wall time.  Durations come
from ``tools/battery/durations.json`` when it is present -- see
``--record`` -- and files with no measurement are shown as estimates so the
total is never quietly wrong.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ALWAYS_LIST = REPO_ROOT / "tools" / "battery" / "always_files.txt"
DURATIONS = REPO_ROOT / "tools" / "battery" / "durations.json"

#: Trees whose ``.py`` files are product code a test can import.
PRODUCT_TREES = ("gpuwm", "tools", "tilestream")

#: Trees that hold collected test modules.
TEST_TREES = ("tests", "tilestream")

#: What a file with no measured duration is assumed to cost, in seconds.
#: Deliberately not zero: an unmeasured file must not make an estimate look
#: cheaper than a measured one.
DEFAULT_SECONDS = 3.0


# ---------------------------------------------------------------------------
# manifest parsing
# ---------------------------------------------------------------------------


def read_manifest(path: pathlib.Path) -> list[str]:
    """Entries from a battery list file, comments and blanks removed.

    This is the parse every consumer of these files must use.  The audit's
    own membership check did a ``grep -F`` instead and reported
    ``tests/test_stage1_manifest.py`` as ON the stage-1 list when the string
    occurred only inside a comment -- which inverted the finding until it
    was caught.  Comment lines are prose about entries, not entries.
    """

    if not path.is_file():
        return []
    entries = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


# ---------------------------------------------------------------------------
# the import index
# ---------------------------------------------------------------------------


def _module_name(path: pathlib.Path) -> str:
    """The dotted module name a product file is importable as."""

    rel = path.relative_to(REPO_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _direct_imports(tree: ast.Module) -> set[str]:
    """Every module name this source imports, at any nesting depth.

    "Direct" is about the IMPORT GRAPH, not about lexical position: an
    import inside a test function body is still this file importing that
    module, and lazy imports inside test bodies are common here.  What is
    excluded is the module that module in turn imports -- the transitive
    step the measurement above rejected.
    """

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative import; not a product edge
                continue
            module = node.module or ""
            if not module:
                continue
            names.add(module)
            # `from gpuwm.core import advection` -> gpuwm.core.advection
            for alias in node.names:
                names.add(f"{module}.{alias.name}")
    return names


def _parse(path: pathlib.Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None


def build_index() -> tuple[dict[str, str], dict[str, set[str]]]:
    """``(module name -> product path, product path -> {test paths})``."""

    modules: dict[str, str] = {}
    for tree_name in PRODUCT_TREES:
        for path in sorted((REPO_ROOT / tree_name).rglob("*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "/rescued-tools/" in rel or rel.startswith("tilestream/test_"):
                continue
            modules[_module_name(path)] = rel

    importers: dict[str, set[str]] = {}
    for tree_name in TEST_TREES:
        for path in sorted((REPO_ROOT / tree_name).rglob("test_*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "/rescued-tools/" in rel:
                continue
            tree = _parse(path)
            if tree is None:
                continue
            for name in _direct_imports(tree):
                product = modules.get(name)
                if product is not None:
                    importers.setdefault(product, set()).add(rel)
    return modules, importers


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def changed_files(base: str, head: str = "HEAD") -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base}..{head}"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _is_test_path(rel: str) -> bool:
    return (rel.startswith("tests/") or rel.startswith("tilestream/")) \
        and pathlib.PurePosixPath(rel).name.startswith("test_") \
        and rel.endswith(".py")


def select(touched: list[str],
           importers: dict[str, set[str]] | None = None,
           ) -> dict[str, list[str]]:
    """``{test file: [reasons]}`` for a list of repository-relative paths."""

    if importers is None:
        _modules, importers = build_index()

    selected: dict[str, list[str]] = {}

    def add(test_path: str, reason: str) -> None:
        reasons = selected.setdefault(test_path, [])
        if reason not in reasons:
            reasons.append(reason)

    for entry in read_manifest(ALWAYS_LIST):
        add(entry, "always (repo-scanning gate)")

    for rel in touched:
        if _is_test_path(rel):
            if (REPO_ROOT / rel).is_file():
                add(rel, "its own file was touched")
            continue
        if not rel.endswith(".py"):
            # A .cu, .toml, .json, .md or .txt has no import edge.  Saying
            # so is the honest answer; the ALWAYS list is the coverage.
            continue
        for test_path in sorted(importers.get(rel, ())):
            add(test_path, f"imports {rel}")

    return {k: selected[k] for k in sorted(selected)}


# ---------------------------------------------------------------------------
# durations
# ---------------------------------------------------------------------------


def load_durations() -> dict[str, float]:
    if not DURATIONS.is_file():
        return {}
    try:
        payload = json.loads(DURATIONS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: float(v) for k, v in payload.get("seconds", {}).items()}


def record_durations(paths: list[str], python: str | None = None) -> None:
    """Measure each file and update the tracked receipt.

    G11 of the audit: ``test_health_field_census.py`` documents its own cost
    as "about 65 s" and measures 541 s, because **nothing in the project
    measures test-suite cost** and the fast/slow split was maintained from
    stale prose.  This is the measurement, kept in the tree so the next
    reader inherits a number instead of a sentence.
    """

    payload = {"seconds": {}}
    if DURATIONS.is_file():
        try:
            payload = json.loads(DURATIONS.read_text(encoding="utf-8"))
            payload.setdefault("seconds", {})
        except (OSError, ValueError):
            payload = {"seconds": {}}

    exe = python or sys.executable
    for rel in paths:
        started = time.time()
        subprocess.run([exe, "-m", "pytest", rel, "-q", "--no-header"],
                       cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        payload["seconds"][rel] = round(time.time() - started, 2)
        print(f"  {payload['seconds'][rel]:8.2f} s  {rel}", flush=True)

    payload["seconds"] = dict(sorted(payload["seconds"].items()))
    DURATIONS.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# front door
# ---------------------------------------------------------------------------


def _ordered(selected: dict[str, list[str]],
             durations: dict[str, float]) -> list[tuple[str, float, bool]]:
    """Cheapest first, so a red arrives fast."""

    rows = [(path, durations.get(path, DEFAULT_SECONDS), path in durations)
            for path in selected]
    return sorted(rows, key=lambda row: (row[1], row[0]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fastfix",
        description="Select the test files a change can break, cheapest first.")
    parser.add_argument("--base", required=True,
                        help="the ref the fix branched from (a tag, usually)")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--format", choices=("report", "pytest", "json"),
                        default="report")
    parser.add_argument(
        "--record", action="store_true",
        help="measure the selection and update tools/battery/durations.json")
    args = parser.parse_args(argv)

    touched = changed_files(args.base, args.head)
    selected = select(touched)
    durations = load_durations()
    rows = _ordered(selected, durations)

    if args.format == "pytest":
        print(" ".join(path for path, _s, _m in rows))
        return 0
    if args.format == "json":
        print(json.dumps(
            {"base": args.base, "head": args.head, "touched": touched,
             "selected": {p: selected[p] for p, _s, _m in rows}},
            indent=2))
        return 0

    print(f"fastfix: {args.base}..{args.head}")
    print(f"  {len(touched)} file(s) touched, "
          f"{sum(1 for t in touched if t.endswith('.py'))} of them Python")
    always = len(read_manifest(ALWAYS_LIST))
    print(f"  {len(rows)} test file(s) selected "
          f"({always} always + {len(rows) - always} by direct import)")
    print()
    total = 0.0
    measured_total = 0.0
    for path, seconds, measured in rows:
        total += seconds
        if measured:
            measured_total += seconds
        mark = " " if measured else "~"
        print(f"  {seconds:7.1f}s{mark} {path}")
        for reason in selected[path]:
            print(f"           {reason}")
    unmeasured = sum(1 for _p, _s, m in rows if not m)
    print()
    print(f"  estimated wall: {total:.0f} s "
          f"({measured_total:.0f} s measured, {unmeasured} file(s) "
          f"estimated at {DEFAULT_SECONDS:g} s and marked ~)")

    if args.record:
        print("\nmeasuring:")
        record_durations([p for p, _s, _m in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
