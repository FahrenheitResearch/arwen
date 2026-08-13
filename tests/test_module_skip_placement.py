"""A module-level skip may not sit below a test.

THE INCIDENT

``tests/test_first_products.py`` carried ``pytest.importorskip("wrf")`` at
line 568, two thirds of the way down, under a banner reading "The real
renderer".  Read as prose that is local: the section below needs the science
core.  Read as Python it is not local at all -- it executes during module
IMPORT, so it skipped the 21 tests defined ABOVE it as well.  Those 21 are
the entire early-render / time-to-first-plot front door, and the file had
been on ``tools/battery/stage1_files.txt`` since 1.8.7, added there *because*
a regression in it is silent.  It collected zero tests on every cut from the
day it landed.  A stage-1 entry that reports green while running nothing is
worse than an absent one: an absent one is at least visibly absent.

The 2026-08-13 test-estate audit found that one by hand.  An AST sweep of the
whole tree found two more of exactly the same shape -- the Noah-MP SFLX and
WATER device gates, each with two source-scan checks written above a
``cp = pytest.importorskip("cupy")``, one of them under a docstring
promising "That test needs no GPU and runs everywhere".  Four more tests that
had never run anywhere.

So it is a class, not an incident, and a class needs a gate rather than three
fixes.

WHAT IS FORBIDDEN, AND WHAT IS NOT

Only the ORDER is.  A module-level skip at the top of a file is fine and
common: it says "this whole module needs X", which is true and legible.  What
this refuses is a module-level skip that appears after the first test
definition, because that spelling always means something the author did not
write -- it silently takes the tests above it too.

The remedies are ordinary:

* move the skip to the top, if the whole module really does need the
  dependency;
* turn it into a fixture and let the tests that need it request it, which is
  what ``tests/test_first_products.py`` now does -- a fixture skips exactly
  its requesters;
* split the module, which is what the two Noah-MP gates now do, because
  ``tests/conftest.py`` marks a module ``gpu`` in its ENTIRETY when cupy is
  imported at module scope, and that rule is correct and must not be worked
  around.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Trees that hold collected test modules.  ``tilestream`` is included
#: because its ``test_*`` modules are real suites (the tiles gate list runs
#: them as ``python -m``), and the defect class is about file shape, not
#: about which runner happens to invoke the file.
_TREES = ("tests", "tilestream")

#: A declared parts bin, by the tree's own README.  Not shipped, not run.
_EXCLUDED_PARTS = ("rescued-tools",)

#: Calls on the ``pytest`` module that end collection of the whole module
#: when they run at module scope.
_SKIPPING_CALLS = frozenset({"importorskip", "skip", "exit", "fail"})


def _module_skip_label(node: ast.stmt) -> str | None:
    """A label when this top-level statement can skip the whole module."""

    call = None
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
    elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
            node.value, ast.Call):
        call = node.value
    if call is not None:
        func = call.func
        if (isinstance(func, ast.Attribute)
                and func.attr in _SKIPPING_CALLS
                and isinstance(func.value, ast.Name)
                and func.value.id == "pytest"):
            return f"pytest.{func.attr}(...)"

    # ``pytestmark = ...`` rebinding below a test is the same trap wearing a
    # different hat: the tests above it are already collected with whatever
    # pytestmark held at their definition, so the assignment reads as if it
    # applied to the section under it and does not.
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "pytestmark":
                return "pytestmark = ..."

    # ``try: import cupy / except ImportError: pytest.skip(...)`` at module
    # scope skips just as hard as the one-liner does.
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            for sub in ast.walk(handler):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in _SKIPPING_CALLS
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == "pytest"):
                    return f"try/except -> pytest.{sub.func.attr}(...)"
    return None


def _is_test_def(node: ast.stmt) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name.startswith("test")
    if isinstance(node, ast.ClassDef):
        return node.name.startswith("Test")
    return False


def _test_modules() -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for tree in _TREES:
        for path in sorted((_ROOT / tree).rglob("test_*.py")):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            found.append(path)
    return found


def _late_skips(source: str) -> list[tuple[int, str, int]]:
    """``(line, label, tests above it)`` for every late module-level skip."""

    tree = ast.parse(source)
    first_test = next((n.lineno for n in tree.body if _is_test_def(n)), None)
    if first_test is None:
        return []
    late = []
    for node in tree.body:
        label = _module_skip_label(node)
        if label is None or node.lineno <= first_test:
            continue
        above = sum(1 for n in tree.body
                    if _is_test_def(n) and n.lineno < node.lineno)
        late.append((node.lineno, label, above))
    return late


def test_no_module_level_skip_sits_below_a_test():
    """The gate.  Every offender is reported, not just the first."""

    modules = _test_modules()
    assert len(modules) > 400, (
        f"only {len(modules)} test modules found -- the sweep is not looking "
        "at the tree it thinks it is")

    offenders = []
    for path in modules:
        source = path.read_text(encoding="utf-8", errors="replace")
        for line, label, above in _late_skips(source):
            offenders.append(
                f"{path.relative_to(_ROOT).as_posix()}:{line}  {label}  "
                f"-- silently skips the {above} test(s) defined above it")

    assert not offenders, (
        "a module-level skip below a test definition takes the tests ABOVE "
        "it too, and reads as if it does not:\n  " + "\n  ".join(offenders))


def test_the_gate_can_fail():
    """The detector must fire on the shape it exists to catch.

    Without this, a refactor that broke the AST walk would leave a silent
    green gate -- which is precisely the failure the gate is about.
    """

    caught = _late_skips(
        "import pytest\n"
        "def test_one():\n"
        "    pass\n"
        "pytest.importorskip('cupy')\n"
        "def test_two():\n"
        "    pass\n")
    assert len(caught) == 1, caught
    line, label, above = caught[0]
    assert (line, label, above) == (4, "pytest.importorskip(...)", 1)

    # ...and must NOT fire on the legitimate spelling, or the remedy it
    # recommends would trip it.
    assert _late_skips(
        "import pytest\n"
        "pytest.importorskip('cupy')\n"
        "def test_one():\n"
        "    pass\n") == []

    # A fixture that skips is the recommended remedy and is not module level.
    assert _late_skips(
        "import pytest\n"
        "def test_one():\n"
        "    pass\n"
        "@pytest.fixture\n"
        "def core():\n"
        "    return pytest.importorskip('wrf')\n") == []


@pytest.mark.parametrize("spelling", [
    "pytest.skip('x', allow_module_level=True)",
    "pytestmark = pytest.mark.skip('x')",
])
def test_the_other_spellings_are_caught_too(spelling):
    """The trap is the position, not the one call the incident used."""

    caught = _late_skips(
        "import pytest\n"
        "def test_one():\n"
        "    pass\n"
        f"{spelling}\n")
    assert len(caught) == 1, (spelling, caught)
