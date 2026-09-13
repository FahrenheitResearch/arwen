"""Every local-DA test file contributes items, on any installation.

THE BREAKAGE THIS PREVENTS.  A module-level ``pytest.importorskip`` empties
its whole file at collection when the package is absent, and
``tools/battery/no_silent_deselection.py`` can only report that as SILENT
DESELECTION or be told to excuse the file by name.  The local-DA leg shipped
with one: ``tests/test_local_da_observations.py`` took
``importorskip('arwen_global.obs_table')`` at module scope, so on every
installation without the shared observation package the file's fifteen
adapter contracts reported nothing at all, including the two that need no
such package.

The remedies this tree already names (``tests/test_module_skip_placement``)
are to split the module or to let the dependency skip exactly its own tests.
Both keep the items COLLECTED, which is the only state the guard and the
census floor can see.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

TESTS = pathlib.Path(__file__).resolve().parent
FILES = sorted(TESTS.glob("test_local_da_*.py"))


def _module_level_skips(tree: ast.Module) -> list[str]:
    """Calls at MODULE scope that empty the file before any test runs.

    Function and class bodies are pruned rather than walked: the same call
    inside a test skips exactly that test, which is one of the remedies
    this file asks for, so descending into them would condemn the fix.
    """

    found: list[str] = []

    def visit(node) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Call):
            target = node.func
            name = (target.attr if isinstance(target, ast.Attribute)
                    else getattr(target, "id", ""))
            if name == "importorskip":
                found.append("pytest.importorskip")
            if name == "skip" and any(kw.arg == "allow_module_level"
                                      for kw in node.keywords):
                found.append("pytest.skip(allow_module_level=True)")
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in tree.body:
        visit(statement)
    return found


def test_the_local_da_leg_has_files_to_check():
    assert FILES, "no tests/test_local_da_*.py were found to check"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_local_da_file_empties_itself_at_collection(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    skips = _module_level_skips(tree)
    assert not skips, (
        f"{path.name} takes {', '.join(sorted(set(skips)))} at module "
        "scope, so on an installation without that package the whole file "
        "collects nothing and the silent-deselection guard reports the file "
        "rather than the dependency.  Split the dependent tests into their "
        "own file and mark that file with a top-of-file skipif, whose items "
        "are still collected and whose reason is printed.")
    assert any(isinstance(node, ast.FunctionDef)
               and node.name.startswith("test_") for node in tree.body), (
        f"{path.name} defines no test at module scope")
