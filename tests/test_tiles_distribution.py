"""Whatever the distribution ships must be able to reach what it imports.

THE DEFECT THIS EXISTS FOR
--------------------------
``gpuwm/core/streaming.py`` shipped in the wheel and ``tilestream/`` did not.
``[tool.setuptools.packages.find]`` read ``include = ["gpuwm*", "tools"]``, so
the built wheel carried the ``[tiles]`` option surface and none of the
transport it calls.  From a clean install::

    mode=off   -> ok (step)
    mode=auto  -> ModuleNotFoundError: No module named 'tilestream'
    mode=on    -> ModuleNotFoundError: No module named 'tilestream'

``decide()`` reaches ``_halo_for`` for every mode except ``off``, and
``make_stepper`` calls ``decide`` first, so the headline feature of the lane
could not be reached from the product at all.  ``tools/les1m_probe.py`` ships
too and imports ``tilestream.harness`` the same way.

WHY NOTHING CAUGHT IT
---------------------
Two reasons, and both are the interesting half.

1. Every ``tilestream`` import in ``streaming.py`` is FUNCTION-LOCAL, so
   ``import gpuwm.core.streaming`` succeeds in a distribution that cannot run
   it.  An import-time smoke test proves nothing here.
2. ``tools/battery/tiles_gates.txt`` says to run the gates "FROM THE
   REPOSITORY ROOT", and a repository root on ``sys.path`` makes ``tilestream``
   importable whether or not it is installed.  Every gate was measuring the
   checkout, never the distribution.

So this file asks the packaging declaration itself, and then asks a subprocess
that has no repository root on its path.  Both are cheap -- a handful of AST
walks and one interpreter start -- which is why this belongs on the stage-1
list rather than at a release cut.

EVERY MEASUREMENT IS PAIRED WITH A DECLARATION
----------------------------------------------
The wheel's real contents can only be asked of ``setuptools.find_packages``,
and the documented project virtualenv has no setuptools -- so a file written
only that way SKIPS in the one environment where it is run, and a skipped
packaging gate reads exactly like a passing one.  This file's first version
was inert that way.  Each ``importorskip`` measurement therefore has a
``tomllib`` reading of the same declaration beside it, and a third test holds
the two against each other so the cheap one cannot drift into a false
comfort.  The pattern is ``tests/test_package_data_coverage.py``'s, stated at
``test_the_redistributed_table_is_not_excluded_in_the_pyproject_declaration``.

RELATED
-------
``tests/test_package_data_coverage.py`` asks the neighbouring question, "does
every non-Python file under ``gpuwm/`` ship", and stops its walk at ``gpuwm/``
by design.  This file asks "can every shipped module import what it imports",
which is the question that has to walk the other packages too.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: Directories that are never part of a distribution.
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache",
                        "build", "dist", ".git"})

#: Top-level import names that resolve to something OTHER than a directory in
#: this repository -- the standard library and third-party dependencies.  The
#: scan below only judges names that ARE repository directories, so this set
#: does not have to be maintained: a name that is not a directory here is
#: somebody else's package and pip resolves it.


def _configuration() -> dict:
    with PYPROJECT.open("rb") as stream:
        return tomllib.load(stream)


def _include_patterns() -> list[str]:
    """``[tool.setuptools.packages.find] include``, read from the file."""
    return list(_configuration()["tool"]["setuptools"]["packages"]["find"][
        "include"])


def _declared_top_level_packages() -> set[str]:
    """The top-level import names the wheel would carry.

    Asked of setuptools' own ``find_packages`` against the real include
    patterns, not re-derived: what the wheel contains is what that function
    returns, and a restatement of glob semantics here could agree with the
    comment in pyproject.toml while disagreeing with the build.

    IMPORTORSKIP, AND WHAT IT COSTS.  The documented project virtualenv has
    no setuptools, so every test routed through here SKIPS exactly where the
    suite is run, and a skipped packaging gate is indistinguishable from a
    passing one in a summary line.  That is why each measurement below is
    paired with a ``tomllib`` reading of the same declaration -- see
    :func:`_top_level_packages_the_declaration_selects` -- on the pattern
    ``tests/test_package_data_coverage.py`` states at
    ``test_the_redistributed_table_is_not_excluded_in_the_pyproject_declaration``.
    Neither replaces the other: this one cannot run without setuptools, and
    the declaration reading cannot tell you what the build actually did.
    Together they fail closed in both environments.
    """
    setuptools = pytest.importorskip("setuptools")

    found = setuptools.find_packages(where=str(REPO_ROOT),
                                     include=_include_patterns())
    return {name.split(".")[0] for name in found}


def _top_level_packages_the_declaration_selects() -> set[str]:
    """The same set, read out of ``pyproject.toml`` with no setuptools.

    Deliberately NOT a glob engine.  It understands the two forms this
    project's include list uses -- an exact package name, and ``NAME*`` --
    and REFUSES anything else rather than guessing, because a second
    half-correct authority that quietly disagrees with the build is worse
    than no second authority.  Refusing is a test failure, not a skip: the
    person who introduces a third pattern form is the person who has to
    teach this function about it.
    """
    here = _repository_top_level_dirs()
    selected: set[str] = set()
    for pattern in _include_patterns():
        stem = pattern[:-1] if pattern.endswith("*") else pattern
        if any(character in stem for character in "*?[]"):
            raise AssertionError(
                f"[tool.setuptools.packages.find] include carries {pattern!r},"
                " which is neither an exact package name nor NAME*.  This "
                "reader deliberately refuses to guess at glob semantics; "
                "teach it the new form, in tests/test_tiles_distribution.py")
        if pattern.endswith("*"):
            selected.update(name for name in here if name.startswith(stem))
        else:
            selected.update(name for name in here if name == stem)
    return selected


def _repository_top_level_dirs() -> set[str]:
    """Every directory at the repository root that Python could import."""
    names = set()
    for entry in REPO_ROOT.iterdir():
        if not entry.is_dir() or entry.name in _SKIP_DIRS:
            continue
        if entry.name.startswith("."):
            continue
        if (entry / "__init__.py").is_file():
            names.add(entry.name)
    return names


def _python_files(package: str):
    root = REPO_ROOT / package
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield Path(dirpath, name)


def _imported_top_level_names(path: Path) -> set[str]:
    """Top-level names ``path`` imports, module scope and function scope alike.

    Function scope matters more than module scope here: the whole defect hid
    behind deferred imports, so a walk that only read the header of a file
    would have reproduced it.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):                # pragma: no cover
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_package_a_shipped_module_imports_is_itself_shipped():
    """The general rule the tilestream omission broke.

    Not "is tilestream in the include list" -- that would be a restatement of
    the fix, green forever and blind to the next package added the same way.
    The property is that the set of shipped packages is CLOSED under
    importing: a module in the distribution may not name a repository package
    the distribution leaves behind.
    """
    shipped = _declared_top_level_packages()
    importable_here = _repository_top_level_dirs()

    problems = []
    for package in sorted(shipped):
        for path in _python_files(package):
            rel = path.relative_to(REPO_ROOT).as_posix()
            for name in sorted(_imported_top_level_names(path)):
                if name in shipped or name not in importable_here:
                    continue
                problems.append(
                    f"{rel} imports {name!r}, which is a package in this "
                    f"repository and is NOT in the distribution")
    assert not problems, (
        "the wheel would ship modules reaching for packages it omits -- add "
        "them to [tool.setuptools.packages.find] include, or stop importing "
        "them from shipped code:\n" + "\n".join(sorted(set(problems))[:40]))


def test_the_declaration_is_closed_under_importing():
    """The same closure property, out of tomllib, so it runs without setuptools.

    The test above is the measurement and this is the declaration, on the
    pattern ``tests/test_package_data_coverage.py`` sets out at
    ``test_the_redistributed_table_is_not_excluded_in_the_pyproject_declaration``.
    It exists because the documented project virtualenv has no setuptools:
    with only the measurement, the two strongest tests in this file SKIPPED
    in the one environment where the suite is actually run, and a green
    summary line said nothing about packaging at all.  Reverting the include
    list to the state that shipped the defect has to turn this file red
    whether setuptools is installed or not.
    """
    shipped = _top_level_packages_the_declaration_selects()
    importable_here = _repository_top_level_dirs()

    problems = []
    for package in sorted(shipped):
        for path in _python_files(package):
            rel = path.relative_to(REPO_ROOT).as_posix()
            for name in sorted(_imported_top_level_names(path)):
                if name in shipped or name not in importable_here:
                    continue
                problems.append(
                    f"{rel} imports {name!r}, which is a package in this "
                    f"repository and is NOT declared in "
                    f"[tool.setuptools.packages.find] include")
    assert not problems, (
        "pyproject.toml declares a distribution whose modules reach for "
        "packages it leaves behind:\n" + "\n".join(sorted(set(problems))[:40]))


def test_setuptools_agrees_with_the_declaration_reading():
    """The two authorities, checked against each other rather than trusted.

    A declaration reader that drifts from the build is how a paired test
    becomes a false comfort: it would stay green over an include list that
    ships something else entirely.  Skipped without setuptools, because with
    only one authority present there is nothing to compare -- and the test
    above is the one that still runs there.
    """
    assert (_declared_top_level_packages()
            == _top_level_packages_the_declaration_selects()), (
        "setuptools' find_packages and this file's tomllib reading of "
        "[tool.setuptools.packages.find] include disagree about which "
        "top-level packages ship.  The reading is the one that runs in the "
        "project virtualenv, so a disagreement makes it a false comfort: fix "
        "_top_level_packages_the_declaration_selects to match the build")


def test_the_tiles_transport_ships_with_the_tiles_option_surface():
    """The specific claim, stated so a reader sees it without running it.

    ``gpuwm/core/streaming.py`` is the ``[tiles]`` surface and ``tilestream``
    is what it calls.  The general rule above would catch the two travelling
    apart, but only as one line in a list; a reader who wants to know whether
    ``[tiles]`` is reachable from a pip install should not have to infer it.

    Read from the DECLARATION so this runs in an environment without
    setuptools -- which is the documented project virtualenv, and was
    therefore the environment in which this test proved nothing.
    """
    shipped = _top_level_packages_the_declaration_selects()
    surface = REPO_ROOT / "gpuwm" / "core" / "streaming.py"
    assert surface.is_file(), "the [tiles] option surface moved"
    assert "tilestream" in _imported_top_level_names(surface), (
        "gpuwm/core/streaming.py no longer imports tilestream, so this test "
        "is measuring nothing -- point it at whatever the transport is now")
    assert "tilestream" in shipped, (
        "gpuwm/core/streaming.py ships and imports tilestream, which does "
        "not; [tiles] mode 'auto' and 'on' would raise ModuleNotFoundError "
        "in decide() from a clean install")


def test_every_module_shipped_code_imports_from_tilestream_exists():
    """A declared package is not the same as a present module.

    ``from tilestream.harness import halo_radius`` needs ``harness.py`` to be
    inside the package the include pattern selects.  Checked separately
    because the include pattern and the file tree can disagree.
    """
    missing = []
    for package in ("gpuwm", "tools"):
        for path in _python_files(package):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):        # pragma: no cover
                continue
            for node in ast.walk(tree):
                wanted = set()
                if isinstance(node, ast.ImportFrom) and node.level == 0 \
                        and (node.module or "").startswith("tilestream"):
                    parts = node.module.split(".")
                    if len(parts) > 1:
                        wanted.add(parts[1])
                    else:
                        wanted.update(alias.name for alias in node.names)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        parts = alias.name.split(".")
                        if parts[0] == "tilestream" and len(parts) > 1:
                            wanted.add(parts[1])
                for name in sorted(wanted):
                    module = REPO_ROOT / "tilestream" / f"{name}.py"
                    package_dir = REPO_ROOT / "tilestream" / name
                    if module.is_file() or package_dir.is_dir():
                        continue
                    missing.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()} imports "
                        f"tilestream.{name}, which does not exist")
    assert not missing, "\n".join(sorted(set(missing)))


def test_the_data_a_shipped_tilestream_module_opens_is_declared():
    """``Path(__file__).with_name(...)`` resolves inside site-packages.

    Any non-Python file a shipped module opens that way has to be declared as
    package data or the install has a ``FileNotFoundError`` waiting in it.
    Derived from the source rather than from the comment in pyproject.toml,
    so the two cannot drift.
    """
    declared = _configuration()["tool"]["setuptools"]["package-data"].get(
        "tilestream", [])
    problems = []
    for path in _python_files("tilestream"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):            # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "with_name"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                continue
            wanted = node.args[0].value
            if wanted.endswith(".py"):
                continue
            if wanted not in declared:
                problems.append(
                    f"{path.relative_to(REPO_ROOT).as_posix()} opens "
                    f"{wanted!r} beside itself, which "
                    "[tool.setuptools.package-data] tilestream does not "
                    "declare")
    assert not problems, "\n".join(sorted(set(problems)))


def test_an_installed_gpuwm_can_import_tilestream_from_any_directory():
    """The half a static scan cannot reach: the install, not the checkout.

    Run with ``cwd`` somewhere neutral and ``-P`` so the script directory is
    not prepended, which is what strips the repository root off ``sys.path``.
    Skipped rather than failed when ``gpuwm`` itself is not importable that
    way: a bare checkout with nothing installed has no distribution to judge,
    and reporting that as a packaging defect is the false alarm that gets a
    gate deleted.
    """
    probe = ("import gpuwm, tilestream, sys;"
             "print(gpuwm.__file__);print(tilestream.__file__)")
    baseline = subprocess.run(
        [sys.executable, "-P", "-c", "import gpuwm;print(gpuwm.__file__)"],
        cwd=Path(sys.prefix), capture_output=True, text=True, check=False)
    if baseline.returncode != 0:
        pytest.skip("gpuwm is not installed in this interpreter, so there is "
                    "no distribution to ask")
    if str(REPO_ROOT) in baseline.stdout:
        # An editable install pointing at THIS tree is the normal development
        # case and is exactly what must also resolve tilestream.
        pass

    completed = subprocess.run(
        [sys.executable, "-P", "-c", probe],
        cwd=Path(sys.prefix), capture_output=True, text=True, check=False)
    assert completed.returncode == 0, (
        "gpuwm imports from an installed distribution but tilestream does "
        "not, so [tiles] is unreachable outside the repository root:\n"
        + completed.stderr)
