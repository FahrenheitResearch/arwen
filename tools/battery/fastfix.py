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
import re
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


class UnreadableTestFile(RuntimeError):
    """A test file the index could not parse, named rather than dropped."""


def _synthetic_product_paths(name: str) -> tuple[str, ...]:
    """The paths a dotted module WOULD occupy, for a module that is gone.

    The inverse of :func:`_module_name`, used only for import names that
    resolve to no file on disk.  Returns an empty tuple for anything
    outside the product trees, so third-party and standard-library imports
    never enter the index.

    BOTH spellings are returned because the name alone cannot distinguish
    them: ``gpuwm.core.advection`` is ``gpuwm/core/advection.py`` if it was
    a module and ``gpuwm/core/advection/__init__.py`` if it was a package,
    and the file is gone, so there is nothing left to look at.  Keying both
    is safe in the direction that matters -- a key for a path that never
    existed is never touched by anything and costs one dict entry, while
    missing the real one is the silent no-selection this fixes.
    """

    parts = name.split(".")
    if not parts or parts[0] not in PRODUCT_TREES:
        return ()
    stem = "/".join(parts)
    return (f"{stem}.py", f"{stem}/__init__.py")


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
    unreadable: list[str] = []
    for tree_name in TEST_TREES:
        for path in sorted((REPO_ROOT / tree_name).rglob("test_*.py")):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if "/rescued-tools/" in rel:
                continue
            tree = _parse(path)
            if tree is None:
                # A TEST FILE THAT DOES NOT PARSE IS NOT A TEST FILE THAT
                # DOES NOT MATTER.  Skipping it silently removes every
                # import edge it owns, so a lane that edits a module only
                # this file imports selects NOTHING, runs nothing, and
                # reports a green fast-fix leg -- and the file was
                # unparseable, which means it was mid-edit or broken, which
                # is precisely when its edges matter most.  The failure is
                # invisible in both directions: no error from the selector,
                # and no test in the output to notice the absence of.
                unreadable.append(rel)
                continue
            for name in _direct_imports(tree):
                product = modules.get(name)
                if product is None:
                    # A NAME THAT RESOLVES TO NO FILE ON DISK IS THE
                    # DELETION CASE, and it is the one the selector used to
                    # answer with silence.  `build_index` walks the tree as
                    # it is NOW, so a product module deleted in the range
                    # under test has no entry, `importers.get(rel)` returns
                    # empty, and the lane runs nothing -- while the tests
                    # that still import the deleted module are exactly the
                    # ones about to fail.  Deleting a module is when import
                    # analysis has the most to say, and it said nothing.
                    #
                    # The edge is recoverable without consulting git,
                    # because the TEST still carries the import statement:
                    # invert _module_name back into the path the module
                    # would occupy and key the edge there, so a later
                    # `importers.get("gpuwm/foo/bar.py")` finds its
                    # importers whether or not that file still exists.
                    # Only names under a product tree are synthesised;
                    # third-party imports are not this index's business.
                    for candidate in _synthetic_product_paths(name):
                        importers.setdefault(candidate, set()).add(rel)
                    continue
                importers.setdefault(product, set()).add(rel)
    if unreadable:
        raise UnreadableTestFile(
            "fastfix cannot parse "
            + ", ".join(unreadable)
            + " -- these files own import edges the selector needs, and "
              "dropping them silently is how a lane runs zero tests and "
              "reports green. Fix the syntax error (or delete the file) "
              "rather than letting the index be built without it.")
    return modules, importers


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def changed_files(base: str, head: str = "HEAD") -> list[str]:
    # --no-renames ON PURPOSE.  Rename detection is on by default, and it
    # reports a renamed module under its NEW path only.  For test selection
    # that is the wrong summary: the tests that import the OLD name are the
    # ones a rename breaks, and they are reachable only from the old path.
    # Reported as a delete plus an add, both paths enter the selector and
    # the deletion half is answered by the synthetic-path edges above.
    out = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", f"{base}..{head}"],
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
            # A .cu, .toml, .json, .md or .txt has no import edge, so import
            # analysis genuinely has nothing to say about it.  But "no import
            # edge" is not "no gate": some gates READ these files.  Saying
            # only "the ALWAYS list is the coverage" was measured wrong on
            # 2026-08-29 -- a WSM6 kernel was switched off
            # (`pracw = 0.0f * fminf(...)`) and the whole 172-file stage-1 leg
            # ran twice, pristine and injected, with ZERO differing node ids.
            # The gate that catches it reads .cu bytes, so it is now on the
            # ALWAYS list, and the extension below selects it BY NAME so the
            # reason a lane runs it is legible rather than incidental.
            for suffix, gates in _NON_PYTHON_GATES.items():
                if rel.endswith(suffix):
                    for gate in gates:
                        if (REPO_ROOT / gate).is_file():
                            add(gate, f"reads {suffix} bytes; {rel} touched")
            continue
        for test_path in sorted(importers.get(rel, ())):
            add(test_path, f"imports {rel}")

    return {k: selected[k] for k in sorted(selected)}


# ---------------------------------------------------------------------------
# durations
# ---------------------------------------------------------------------------


#: Gates that READ a file type rather than importing it, keyed by suffix.
#: The selector cannot reach these by import edge -- that is the whole reason
#: they exist -- so they are named here and are also on ``always_files.txt``.
#: Belt and braces on purpose: the ALWAYS list makes them run, this table
#: makes the REASON they ran appear in the selector's output, which is what a
#: person iterating on a kernel actually needs to see.
_NON_PYTHON_GATES: dict[str, tuple[str, ...]] = {
    ".cu": (
        "tests/test_kernel_source_freeze_per_module.py",
        "tests/test_cuda_libm_table_copies.py",
    ),
    ".rs": ("tests/test_bridge_source_rev_stamp.py",),
}


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


#: ``  1.23s call     tests/test_x.py::test_y`` -- pytest's ``--durations``
#: report line.  Setup, call and teardown are three lines per test.
_DURATION_LINE = re.compile(
    r"^\s*(?P<seconds>[0-9.]+)s\s+(?:setup|call|teardown)\s+"
    r"(?P<node>\S+\.py)::")


def durations_from_batched_report(text: str) -> dict[str, float]:
    """Per-file seconds from ONE pytest run's ``--durations=0`` report.

    THE ACCIDENTAL COST THIS REMOVES, measured on this box 2026-08-29.
    ``record_durations`` above spawns one pytest per file.  The battery runs
    the files in a SINGLE invocation, and an invocation costs about 2.45 s of
    interpreter start, plugin load and conftest import before a test runs:
    172 stage-1 files as 172 processes measured 2,252 s against 1,441.68 s for
    the same files in one invocation -- 421 s, 29% of the leg, spent on
    nothing.

    Every row written by ``record_durations`` therefore carries ~2.45 s the
    battery never pays.  That is a rounding error on a 300 s file and it is
    50-100% on a 2 s one, and 100 of the 172 stage-1 files are under 5 s --
    which is exactly the end where ``_ordered``'s cheapest-first decision is
    made.  A selector that sorts by a systematically wrong number puts the
    red later than it needs to be.

    This reads the report of a run that already happened, so it costs nothing
    and it measures the arrangement the battery actually uses.  Sum over
    setup, call and teardown, because a fixture is part of what a file costs.

    WHAT IT DOES NOT MEASURE, stated rather than hidden: pytest attributes
    only those three phases, so COLLECTION -- where a module's import cost
    lands -- is not in the report and a batched row understates an
    import-heavy file.  It is still the better of the two numbers for the
    cheapest-first ordering, because the term it omits is smaller and more
    uniform across files than the ~2.45 s the per-process spelling adds to
    every row.
    """

    seconds: dict[str, float] = {}
    for line in text.splitlines():
        match = _DURATION_LINE.match(line)
        if match is None:
            continue
        path = match.group("node").replace("\\", "/")
        seconds[path] = round(seconds.get(path, 0.0)
                              + float(match.group("seconds")), 2)
    return seconds


def record_durations_batched(report: pathlib.Path) -> int:
    """Update the tracked receipt from a batched report, and say what moved."""

    raw = report.read_bytes()
    # A PowerShell redirect writes UTF-16 on this box; decode by BOM
    # rather than assuming, so a report captured either way parses.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    measured = durations_from_batched_report(text)
    if not measured:
        print(f"no --durations lines in {report}; nothing recorded",
              file=sys.stderr)
        return 1

    payload = {"seconds": {}}
    if DURATIONS.is_file():
        try:
            payload = json.loads(DURATIONS.read_text(encoding="utf-8"))
            payload.setdefault("seconds", {})
        except (OSError, ValueError):
            payload = {"seconds": {}}

    for path, value in sorted(measured.items()):
        was = payload["seconds"].get(path)
        payload["seconds"][path] = value
        if was is None:
            print(f"  {value:8.2f} s  {path}  (new)")
        elif abs(was - value) >= 0.05:
            print(f"  {value:8.2f} s  {path}  (was {was:.2f} s, "
                  f"{value - was:+.2f})")
    stale = sorted(set(payload["seconds"]) - set(measured))
    payload["seconds"] = dict(sorted(payload["seconds"].items()))
    payload["_stale_rows"] = (
        "Rows this batched run did not refresh, still carrying whatever "
        "method and date _measured describes.  A file lands here either "
        "because the run did not include it, or because every one of its "
        "tests was under pytest's 0.005 s reporting floor and so emitted no "
        "duration line at all -- the second case is a row that is already "
        "as cheap as the receipt can express: "
        + (", ".join(stale) if stale else "none") + "."
    )
    payload["_how"] = (
        "One batched pytest invocation with --durations=0, summed per file "
        "over setup+call+teardown; see fastfix.py "
        "durations_from_batched_report.  This is the arrangement the battery "
        "runs.  The earlier spelling of this file measured one pytest process "
        "per file, which added about 2.45 s of interpreter and conftest start "
        "to every row -- 421 s across the 172-file stage-1 leg, and 50-100% "
        "of the recorded cost of the 100 files that are under 5 s."
    )
    DURATIONS.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + chr(10),
        encoding="utf-8", newline=chr(10))
    print()
    print(f"  {len(measured)} file(s) recorded from {report}")
    return 0


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
    parser.add_argument("--base",
                        help="the ref the fix branched from (a tag, usually); "
                             "required for a selection, unused by "
                             "--record-from")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--format", choices=("report", "pytest", "json"),
                        default="report")
    parser.add_argument(
        "--record", action="store_true",
        help="measure the selection and update tools/battery/durations.json")
    parser.add_argument(
        "--record-from", metavar="REPORT", type=pathlib.Path,
        help="update tools/battery/durations.json from the --durations=0 "
             "report of a BATCHED run, which is the arrangement the battery "
             "uses; --record spawns one process per file and inflates every "
             "row by the interpreter start the battery does not pay")
    args = parser.parse_args(argv)

    if args.record_from is not None:
        return record_durations_batched(args.record_from)
    if not args.base:
        parser.error("--base is required to select tests")

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
