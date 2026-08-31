"""Repository-root conftest: arm the pre-commit line-ending hook.

WHY THIS FILE EXISTS.  ``tools/git_hooks/pre_commit_line_endings.py``
refuses a commit that would give an authored file a carriage return, and
it stopped nothing at all, because arming it meant reading an
``--install`` flag documented only inside its own docstring and nothing
in the repository ran it.  Measured on 2026-08-29: ``.git/hooks/pre-commit``
did not exist in this clone, no ``core.hooksPath`` was set, no
``.pre-commit-config.yaml``, and the two incidents the hook was written
for were both committed straight through it --
``tests/test_offline_child.py`` rewritten to CRLF across all 673 lines
with a real 79-line addition buried inside the whole-file diff (repaired
by 39ef138c5), and
``tools/rustwx/crates/rw-mpas/src/bin/fp32_floor_probe.rs`` born with 525
carriage returns (repaired by abb5ff270).

"Fixed means default" is a project law, so the hook arms itself.  pytest
is the arming point because it is the one command every lane in this
repository runs, and a conftest at the ROOT is loaded for any test under
it -- so a fresh clone is armed by its first test run, with nobody
remembering anything.

It is deliberately quiet on success and it does NOT make itself true:
``tests/test_line_ending_stability.py::test_the_pre_commit_hook_is_armed``
asks the filesystem separately and fails if this did not take.  A hook
that is quietly not there is the exact defect being fixed, so failing to
arm has to be visible rather than absent.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import warnings

_HOOK = (pathlib.Path(__file__).resolve().parent
         / "tools" / "git_hooks" / "pre_commit_line_endings.py")


def _is_checkout() -> bool:
    """THIS repository's working tree, so its hooks are ours to arm.

    Asked FIRST and answered in silence, because an unpacked sdist
    carries this file and `tools/git_hooks/` with it -- setuptools' sdist
    default file set sweeps every .py under the project root -- and there
    is no repository there to have a pre-commit hook.  Warning in that
    case would put a RuntimeWarning on every `pytest` run of an installed
    source distribution and say nothing true.

    The toplevel comparison is the ownership check: an sdist unpacked
    INSIDE some other project's git repository still answers "yes" to
    ``rev-parse``, and arming there would write a pre-commit hook into a
    git directory we do not own.  Only when the working tree's root IS
    the directory holding this conftest is the hooks directory ours.  A
    linked gpuwm worktree passes (its root carries this file) and arms
    the shared hooks path, which is the intended one-per-clone shape.
    """
    here = pathlib.Path(__file__).resolve().parent
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=here, capture_output=True, check=True,
        ).stdout.decode("utf-8", "replace").strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    if not toplevel:
        return False
    try:
        return pathlib.Path(toplevel).resolve().samefile(here)
    except OSError:
        return False


def _arm() -> None:
    if not _HOOK.is_file() or not _is_checkout():
        return  # an sdist, or a tree older than the hook: nothing to arm
    spec = importlib.util.spec_from_file_location("_pre_commit_line_endings",
                                                  _HOOK)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        armed, story = module.ensure_installed()
    except Exception as failure:      # not a checkout, no git, read-only .git
        armed, story = False, "%s: %s" % (type(failure).__name__, failure)
    if not armed:
        warnings.warn(
            "the pre-commit line-ending hook is NOT armed (%s).  Commits "
            "in this clone are not checked for CRLF flips; see "
            "tools/git_hooks/pre_commit_line_endings.py." % story,
            RuntimeWarning, stacklevel=1)


_arm()
