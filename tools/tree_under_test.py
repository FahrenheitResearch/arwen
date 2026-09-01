"""Which checkout is pytest actually importing?

THE DEFECT THIS CLOSES
----------------------
``gpuwm`` is installed editable on the development box, and a setuptools
editable install binds the distribution name to ONE directory -- the main
checkout -- through a finder on ``sys.meta_path``.  A meta-path finder
answers before ``sys.path`` does.  So a lane running its own test suite
inside ``git worktree`` gets this:

    cd <some-lane-worktree>
    python -m pytest tests/test_thing.py        # 34 passed

...where ``import gpuwm`` and ``import tools.x`` resolved to
the MAIN CHECKOUT'S editable install.  The lane's edits were never executed.  The
suite measured the MAIN CHECKOUT and reported green about the lane, and
it does it just as cheerfully when the lane's change is broken.

This was found while verifying a harness lane's own work: a committed fix
to ``tools/check_negation_invariant.py`` had five tests failing in the
worktree that held the fix, because the worktree was running the
unfixed main-checkout copy of the file it had just fixed.  Every lane
worktree on this box has had the same hole under it.

There is no clever fix inside pytest for this; the remedy is one
environment variable, and the job of this module is to make sure nobody
ever again has to discover its absence from a mysterious result.  A
``sys.path`` entry beats the editable finder because setuptools APPENDS
its finder to ``sys.meta_path``, behind ``PathFinder``.

WHAT COUNTS AS WRONG
--------------------
Not "imported from outside the repository" -- that would condemn the one
shape that is legitimate and deliberate.  A wheel install is a real
subject of test: the user-door sweep installs a candidate wheel and runs
against it on purpose, and there ``gpuwm`` lives in ``site-packages``
with no source checkout anywhere near it.

The failure is narrower and unambiguous: the imported module belongs to
ANOTHER SOURCE CHECKOUT OF THIS SAME PROJECT.  A directory that holds a
``pyproject.toml`` declaring this distribution is a checkout; if it is not
the one the tests were collected from, two trees are in the room and the
results describe the wrong one.
"""

from __future__ import annotations

import pathlib
import re
import sys
import types

#: The top-level names whose resolution decides which tree ran.  Both are
#: in the list because they fail independently: ``tools`` is where the
#: repo-scanning gates live and ``gpuwm`` is the product, and an editable
#: install maps whichever of them its wheel declared.
WATCHED = ("gpuwm", "tools", "tilestream")

#: Read out of a candidate ``pyproject.toml`` to decide "checkout of THIS
#: project" versus "some other Python project that happens to sit above
#: an installed file".  Deliberately a regex over the text rather than a
#: TOML parse: this runs before the session and must not depend on a
#: parser, and a false NEGATIVE here only costs a missed warning.
_NAME_LINE = re.compile(r"""^\s*name\s*=\s*["']([^"']+)["']""", re.M)

#: How far up from an imported file to look for the checkout root.  A
#: package five directories deep inside a checkout is still in it; a file
#: in ``site-packages`` reaches ``site-packages`` and stops, which has no
#: ``pyproject.toml`` and is therefore correctly not a checkout.
_MAX_DEPTH = 8


def distribution_name(root: pathlib.Path) -> str | None:
    """The distribution ``root/pyproject.toml`` declares, or ``None``."""

    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8",
                                                   errors="replace")
    except OSError:
        return None
    match = _NAME_LINE.search(text)
    return match.group(1).strip().lower().replace("_", "-") if match else None


def owning_checkout(path: pathlib.Path, project: str) -> pathlib.Path | None:
    """The checkout of ``project`` that contains ``path``, or ``None``.

    Walks up looking for the ``pyproject.toml`` that names ``project``.
    ``None`` means the file is not inside a source checkout of it -- an
    installed wheel, a vendored copy, a zip -- which is a legitimate
    subject of test and never an error here.
    """

    project = project.lower().replace("_", "-")
    for parent in list(path.parents)[:_MAX_DEPTH]:
        if distribution_name(parent) == project:
            return parent
    return None


def _module_locations(module: types.ModuleType) -> list[pathlib.Path]:
    """Every directory a module was loaded from.

    ``__file__`` for an ordinary package or module; ``__path__`` for a
    namespace package, which has no ``__file__`` at all and would
    otherwise slip through the check entirely.
    """

    found: list[pathlib.Path] = []
    filename = getattr(module, "__file__", None)
    if filename:
        found.append(pathlib.Path(filename).resolve())
    for entry in list(getattr(module, "__path__", ()) or ()):
        try:
            found.append(pathlib.Path(str(entry)).resolve() / "__init__.py")
        except (OSError, ValueError):        # pragma: no cover - exotic path
            continue
    return found


def foreign_imports(repo_root: pathlib.Path,
                    modules: dict[str, types.ModuleType | None],
                    project: str = "gpuwm",
                    ) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """``[(module name, file, the other checkout it came from)]``.

    Pure, and takes its modules as an argument, so both directions can be
    tested without arranging a second checkout on disk.  Empty means the
    run is honest: every watched module either came from ``repo_root`` or
    came from something that is not a competing checkout.
    """

    repo_root = pathlib.Path(repo_root).resolve()
    offences: list[tuple[str, pathlib.Path, pathlib.Path]] = []
    for name, module in modules.items():
        if module is None:
            continue
        for location in _module_locations(module):
            if location == repo_root or repo_root in location.parents:
                continue
            other = owning_checkout(location, project)
            if other is not None and other != repo_root:
                offences.append((name, location, other))
                break                        # one report per module is plenty
    return offences


def imported_modules(names=WATCHED) -> dict[str, types.ModuleType | None]:
    """The watched modules AS THIS PROCESS RESOLVES THEM, importing if needed.

    Importing is the point.  Asking ``sys.modules`` alone would answer
    "not imported yet" during session start and check nothing at all --
    the same shape of silence this module exists to end.
    """

    resolved: dict[str, types.ModuleType | None] = {}
    for name in names:
        module = sys.modules.get(name)
        if module is None:
            try:
                __import__(name)
            except Exception:                # noqa: BLE001
                # Not importable at all is somebody else's error, and a
                # loud one; it is not evidence about WHICH tree ran.
                resolved[name] = None
                continue
            module = sys.modules.get(name)
        resolved[name] = module
    return resolved


def refusal(repo_root: pathlib.Path,
            offences: list[tuple[str, pathlib.Path, pathlib.Path]],
            ) -> str | None:
    """The message, with the one-line remedy, or ``None`` when clean."""

    if not offences:
        return None
    repo_root = pathlib.Path(repo_root).resolve()
    lines = [
        f"REFUSING to run: these tests were collected from {repo_root}, but "
        f"the code under test is being imported from a DIFFERENT checkout "
        f"of this project:",
    ]
    for name, location, other in offences:
        lines.append(f"  import {name}  ->  {location}")
        lines.append(f"      which belongs to the checkout at {other}")
    lines.append("")
    lines.append(
        "An editable install binds the distribution name to one directory "
        "through a finder on sys.meta_path, and a meta-path finder answers "
        "before sys.path does.  Every edit in this tree would have been "
        "invisible and the suite would have reported green about the other "
        "one.")
    lines.append("")
    lines.append(f"Remedy -- put this tree ahead of the editable finder:")
    lines.append(f"    PowerShell:  $env:PYTHONPATH = '{repo_root}'")
    lines.append(f"    bash:        export PYTHONPATH='{repo_root}'")
    lines.append(
        "Then re-run pytest from this directory.  (Installing this tree "
        "editable instead would repoint the binding for every other "
        "worktree on the box, which is the same defect aimed elsewhere.)")
    return "\n".join(lines)


def check(repo_root: pathlib.Path, project: str = "gpuwm") -> str | None:
    """The whole question in one call: the refusal, or ``None``."""

    return refusal(repo_root, foreign_imports(
        repo_root, imported_modules(), project))


def main(argv: list[str] | None = None) -> int:
    """``python tools/tree_under_test.py`` -- report and exit 1 if foreign."""

    root = pathlib.Path(__file__).resolve().parents[1]
    message = check(root)
    if message is None:
        print(f"tree-under-test: OK -- {', '.join(WATCHED)} all resolve "
              f"inside {root}")
        return 0
    print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":                   # pragma: no cover
    raise SystemExit(main())
