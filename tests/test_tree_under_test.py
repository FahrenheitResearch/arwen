"""The instrument that says which checkout the suite actually measured.

An editable install binds ``gpuwm`` and ``tools`` to ONE directory through
a finder on ``sys.meta_path``, and a meta-path finder answers before
``sys.path`` does.  A lane running its suite inside ``git worktree``
therefore imports the MAIN checkout, executes none of its own edits, and
prints a number of passing tests about a tree it never touched.

That is not a hypothetical.  It was found while verifying a harness lane:
``tools/check_negation_invariant.py`` had a committed fix, and five tests
of that fix FAILED in the worktree holding it -- because the worktree was
running the unfixed copy from the main checkout.  The same hole was under
every other lane worktree on the box, in the other direction: a broken
edit would have gone green.

``tools/tree_under_test.py`` closes it, ``tests/conftest.py`` refuses the
session when it fires, and this file is the both-direction proof.  The
predicate takes its modules as an argument precisely so the RED direction
can be tested without arranging a second checkout on disk.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

from tools import tree_under_test

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _module_at(path: pathlib.Path) -> types.ModuleType:
    """A stand-in module that claims to have been loaded from ``path``."""

    module = types.ModuleType("stand_in")
    module.__file__ = str(path)
    return module


def _checkout(root: pathlib.Path, name: str = "gpuwm") -> pathlib.Path:
    """A directory that looks like a source checkout of ``name``."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "9.9.9"\n', encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# GREEN: the shapes that must never fire
# ---------------------------------------------------------------------------


def test_the_live_session_is_running_this_tree():
    """The check applied to the process actually running these tests.

    If this ever fails, the rest of this file is describing some other
    directory and so is every other test in the suite.
    """

    assert tree_under_test.check(REPO_ROOT) is None, (
        "this pytest process is importing the product from a different "
        "checkout; nothing it reports is about this tree")


def test_a_module_inside_the_repository_is_not_foreign(tmp_path):
    inside = _module_at(REPO_ROOT / "gpuwm" / "__init__.py")
    assert tree_under_test.foreign_imports(
        REPO_ROOT, {"gpuwm": inside}) == []


def test_a_wheel_install_is_a_legitimate_subject_of_test(tmp_path):
    """site-packages is NOT an offence, and the distinction is the point.

    The user-door sweep installs a candidate wheel and runs against it
    deliberately.  A check that condemned "imported from outside the
    repository" would condemn the one shape that is supposed to happen,
    and would be switched off within a week.
    """

    site = tmp_path / "venv" / "Lib" / "site-packages"
    (site / "gpuwm").mkdir(parents=True)
    (site / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")

    assert tree_under_test.foreign_imports(
        REPO_ROOT,
        {"gpuwm": _module_at(site / "gpuwm" / "__init__.py")}) == []


def test_an_unrelated_project_above_the_file_is_not_this_project(tmp_path):
    """A ``pyproject.toml`` naming something else is not a rival checkout."""

    other = _checkout(tmp_path / "somebody-elses-tool", name="unrelated-tool")
    (other / "gpuwm").mkdir()
    (other / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")

    assert tree_under_test.foreign_imports(
        REPO_ROOT,
        {"gpuwm": _module_at(other / "gpuwm" / "__init__.py")}) == []


def test_a_module_that_did_not_import_says_nothing_about_the_tree():
    assert tree_under_test.foreign_imports(REPO_ROOT, {"tilestream": None}) == []


# ---------------------------------------------------------------------------
# RED: the exact incident
# ---------------------------------------------------------------------------


def test_a_sibling_checkout_of_this_project_is_refused(tmp_path):
    """THE INCIDENT.  Worktree collects the tests, main checkout runs.

    ``repo_root`` is the worktree; ``gpuwm`` resolved into a second
    checkout of the same distribution.  Every edit in the worktree is
    invisible and the verdict describes the other tree.
    """

    worktree = _checkout(tmp_path / "wt-lane")
    main = _checkout(tmp_path / "gpuwm")
    (main / "gpuwm").mkdir()
    (main / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")

    offences = tree_under_test.foreign_imports(
        worktree, {"gpuwm": _module_at(main / "gpuwm" / "__init__.py")})
    assert len(offences) == 1
    name, location, other = offences[0]
    assert name == "gpuwm"
    assert location == (main / "gpuwm" / "__init__.py").resolve()
    assert other == main.resolve()


def test_a_namespace_package_with_no_dunder_file_is_still_caught(tmp_path):
    """``tools`` can arrive as a namespace package, with ``__file__`` unset.

    Reading only ``__file__`` would let exactly the package that holds the
    repo-scanning gates through unchecked.
    """

    worktree = _checkout(tmp_path / "wt-lane")
    main = _checkout(tmp_path / "gpuwm")
    (main / "tools").mkdir()

    module = types.ModuleType("tools")
    module.__path__ = [str(main / "tools")]
    offences = tree_under_test.foreign_imports(worktree, {"tools": module})
    assert [entry[0] for entry in offences] == ["tools"]
    assert offences[0][2] == main.resolve()


def test_the_refusal_names_both_trees_and_carries_the_remedy(tmp_path):
    """A refusal a reader cannot act on is a refusal they route around."""

    worktree = _checkout(tmp_path / "wt-lane")
    main = _checkout(tmp_path / "gpuwm")
    (main / "gpuwm").mkdir()
    (main / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")

    message = tree_under_test.refusal(worktree, tree_under_test.foreign_imports(
        worktree, {"gpuwm": _module_at(main / "gpuwm" / "__init__.py")}))
    assert message is not None
    assert "REFUSING" in message
    assert str(worktree.resolve()) in message
    assert str(main.resolve()) in message
    assert "PYTHONPATH" in message


def test_the_conftest_hook_refuses_a_wrong_tree_session(tmp_path,
                                                        monkeypatch):
    """The wiring, not just the predicate: a bad session must not start.

    ``pytest_configure`` is where it lands, so a foreign tree costs the
    run zero collected tests and a usage error, rather than a green
    summary about somebody else's code.
    """

    conftest = sys.modules[
        "tests.conftest" if "tests.conftest" in sys.modules else "conftest"]

    main = _checkout(tmp_path / "gpuwm")
    (main / "gpuwm").mkdir()
    (main / "gpuwm" / "__init__.py").write_text("", encoding="utf-8")
    loaded = conftest._tree_under_test()

    monkeypatch.setattr(
        loaded, "imported_modules",
        lambda names=None: {
            "gpuwm": _module_at(main / "gpuwm" / "__init__.py")})
    monkeypatch.setattr(conftest, "_tree_under_test", lambda: loaded)

    with pytest.raises(pytest.UsageError, match="REFUSING"):
        conftest.pytest_configure(_StubConfig())


class _StubConfig:
    """Enough of ``pytest.Config`` for ``pytest_configure`` to run."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def addinivalue_line(self, name: str, value: str) -> None:
        self.lines.append(value)
