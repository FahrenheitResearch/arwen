"""The fast-fix lane's selector, and the list it cannot select.

``tools/battery/fastfix.py`` decides which suites a change has to run before
it is published.  That makes it a thing that can be silently wrong, which is
the exact failure mode the 2026-08-13 test-estate audit was chartered to
find -- a gate that reports green while measuring nothing.  A selector with
no gate of its own would be the audit's own finding, rebuilt.

So this file pins the two properties the audit's validation established, in
both directions:

* a real defect commit must still be caught (``6e9c690f0`` ->
  ``tests/test_pd_advection.py``), and
* a change the selector structurally cannot analyse must fall back to the
  ALWAYS list rather than selecting nothing.

plus the manifest hygiene on ``tools/battery/always_files.txt``, in the
shape ``tests/test_stage1_manifest.py`` established for the stage-1 list.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
SELECTOR = REPOSITORY_ROOT / "tools" / "battery" / "fastfix.py"
ALWAYS_LIST = REPOSITORY_ROOT / "tools" / "battery" / "always_files.txt"

#: The commit the audit validated the depth-0 setting against.  It touched
#: gpuwm/core/moist.py and gpuwm/core/dycore.py and broke two things: the
#: cp.ElementwiseKernel the numpy-substituted CPU backend cannot execute
#: (caught by the test below) and a stale FTZ route receipt (caught only by
#: the ALWAYS list).  One commit, both halves of the design.
VALIDATION_COMMIT = "6e9c690f0"
VALIDATION_CATCH = "tests/test_pd_advection.py"


def _load_selector():
    spec = importlib.util.spec_from_file_location("battery_fastfix", SELECTOR)
    assert spec and spec.loader, SELECTOR
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fastfix = _load_selector()


def _have_commit(ref: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{ref}^{{commit}}"],
        cwd=REPOSITORY_ROOT, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL).returncode == 0


# ---------------------------------------------------------------------------
# The validated behaviour
# ---------------------------------------------------------------------------


def test_the_validation_commit_still_selects_the_test_that_catches_it():
    """The audit's own experiment, pinned as a regression.

    ``6e9c690f0`` replaced an eager ``cp.multiply/add/divide`` chain in
    ``gpuwm/core/moist.py`` with a ``cp.ElementwiseKernel``, which the
    numpy-substituted CPU backend cannot execute.
    ``tests/test_pd_advection.py`` is what fails on it.  The audit chose
    depth 0 over depth 1 and depth 2 *because* depth 0 catches this and the
    deeper settings add nothing but three times the cost -- so if this ever
    stops selecting, the setting has been changed by accident.
    """

    if not _have_commit(VALIDATION_COMMIT):
        pytest.skip(f"{VALIDATION_COMMIT} is not in this clone")

    touched = fastfix.changed_files(f"{VALIDATION_COMMIT}~1",
                                    VALIDATION_COMMIT)
    assert "gpuwm/core/moist.py" in touched, touched
    assert "gpuwm/core/dycore.py" in touched, touched

    selected = fastfix.select(touched)
    assert VALIDATION_CATCH in selected, (
        f"{VALIDATION_CATCH} is the test that catches {VALIDATION_COMMIT}'s "
        "CPU-backend defect, and the selector no longer picks it.  Selected "
        f"{len(selected)} files: {sorted(selected)[:20]}...")
    assert any("moist.py" in reason or "dycore.py" in reason
               for reason in selected[VALIDATION_CATCH]), \
        selected[VALIDATION_CATCH]


def test_the_selection_stays_a_selection():
    """Depth 0's reason to exist: it must not select most of the estate.

    Transitive closure selected 441 of 591 files (75%) on this repository,
    which is why it was rejected.  If a future edit reintroduces transitive
    walking, this is what says so -- the failure is not "wrong answer", it
    is "no longer a selector", and that is measurable.
    """

    if not _have_commit(VALIDATION_COMMIT):
        pytest.skip(f"{VALIDATION_COMMIT} is not in this clone")

    touched = fastfix.changed_files(f"{VALIDATION_COMMIT}~1",
                                    VALIDATION_COMMIT)
    selected = fastfix.select(touched)
    total = sum(len(list((REPOSITORY_ROOT / tree).rglob("test_*.py")))
                for tree in fastfix.TEST_TREES)
    share = len(selected) / total
    assert share < 0.40, (
        f"the selector picked {len(selected)} of {total} test files "
        f"({share:.0%}).  The audit rejected transitive closure at 75% on "
        "the grounds that a selector which selects most of the estate is not "
        "a selector; something has widened the walk")


def test_a_non_python_change_falls_back_to_the_always_list():
    """The honest answer to a ``.cu`` edit, and it must not be silence.

    Import analysis says nothing about a CUDA translation unit, a config
    TOML or a receipt JSON.  Selecting nothing would be a lane that
    publishes a kernel change having run no gate at all; the fallback is the
    ALWAYS list, which is what catches the receipt half of the very commit
    this file pins above.
    """

    always = set(fastfix.read_manifest(ALWAYS_LIST))
    for touched in (["gpuwm/core/kernels/morrison.cu"],
                    ["configs/some_case.toml"],
                    ["tools/ftz_receipt/receipt/route_inventory.json"],
                    ["docs/public/HARDWARE.md"]):
        selected = fastfix.select(touched)
        assert set(selected) == always, (touched, sorted(selected))
        for entry in selected.values():
            assert entry == ["always (repo-scanning gate)"], entry


def test_an_empty_change_still_runs_the_always_list():
    """A selector that can return nothing is a lane that can gate nothing."""

    selected = fastfix.select([])
    assert set(selected) == set(fastfix.read_manifest(ALWAYS_LIST))
    assert selected, "the ALWAYS list parsed to nothing"


def test_a_touched_test_file_selects_itself():
    selected = fastfix.select(["tests/test_stage1_manifest.py"])
    assert "tests/test_stage1_manifest.py" in selected
    assert "its own file was touched" in \
        selected["tests/test_stage1_manifest.py"]


def test_the_cheapest_file_is_offered_first():
    """A red should arrive fast, so the order is by measured cost."""

    rows = fastfix._ordered(
        {"tests/a.py": ["x"], "tests/b.py": ["x"], "tests/c.py": ["x"]},
        {"tests/a.py": 90.0, "tests/b.py": 0.5})
    assert [path for path, _s, _m in rows] == [
        "tests/b.py", "tests/c.py", "tests/a.py"]
    # An unmeasured file is flagged, so an estimate is never read as a
    # measurement.
    assert [measured for _p, _s, measured in rows] == [True, False, True]


# ---------------------------------------------------------------------------
# The ALWAYS list itself
# ---------------------------------------------------------------------------


def test_the_always_list_exists_and_lists_something():
    assert ALWAYS_LIST.is_file(), f"{ALWAYS_LIST} is missing"
    assert fastfix.read_manifest(ALWAYS_LIST), (
        "tools/battery/always_files.txt parsed to zero entries; every fastfix "
        "lane would run only its selected files and no repo-scanning gate")


@pytest.mark.parametrize("entry", fastfix.read_manifest(ALWAYS_LIST))
def test_every_always_entry_exists(entry: str):
    assert not pathlib.PurePosixPath(entry).is_absolute(), entry
    assert "\\" not in entry, (
        f"{entry} uses a backslash; forward slashes only, so the same list "
        "works on the Windows cut box and a Linux runner")
    assert entry.startswith("tests/") and entry.endswith(".py"), entry
    assert (REPOSITORY_ROOT / entry).is_file(), (
        f"{entry} is listed in tools/battery/always_files.txt but does not "
        "exist; pytest would fail the whole lane on the missing argument")


def test_no_always_entry_is_listed_twice():
    entries = fastfix.read_manifest(ALWAYS_LIST)
    duplicates = sorted({e for e in entries if entries.count(e) > 1})
    assert not duplicates, duplicates


def test_the_always_list_is_lf_only():
    assert b"\r" not in ALWAYS_LIST.read_bytes(), (
        "tools/battery/always_files.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


def test_the_manifest_parse_ignores_comments():
    """The audit's own instrument failed exactly here.

    A ``grep -F`` of a path against ``stage1_files.txt`` reported
    ``tests/test_stage1_manifest.py`` as ON the list; the string occurs only
    inside a comment.  The finding survived -- the gate on the stage-1 list
    was not on the stage-1 list -- but only because the instrument was
    re-checked.  Every consumer of these files parses through
    ``read_manifest``, and this is what pins that it strips comments.
    """

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "list.txt"
        path.write_text(
            "# tests/test_commented_out.py is the gate\n"
            "\n"
            "   # indented comment mentioning tests/test_also_not.py\n"
            "tests/test_real.py\n"
            "\n",
            encoding="utf-8", newline="\n")
        assert fastfix.read_manifest(path) == ["tests/test_real.py"]


def test_a_receipt_gate_is_not_reachable_from_the_code_it_inventories():
    """The ALWAYS list's premise, stated precisely enough to be true.

    The audit reported ``tests/test_ftz_route_inventory.py`` as "a MISS at
    every depth".  Building the index here shows that is *nearly* right and
    worth sharpening: the file does import ``tools.ftz_receipt.
    route_inventory``, its own generator, so touching THAT selects it.  What
    it has no edge to is the code it inventories -- the cupy
    kernel-construction sites in ``gpuwm/core/``.  The receipt goes stale
    when someone edits a kernel site, and that is the edit with no edge.

    Which is the same conclusion (this gate must be unconditional) resting
    on a claim that survives being checked.  ``6e9c690f0`` is the worked
    example: it touched ``gpuwm/core/moist.py`` and
    ``gpuwm/core/dycore.py``, added two ``cp.ElementwiseKernel`` sites, and
    left the receipt stale.
    """

    _modules, importers = fastfix.build_index()
    gate = "tests/test_ftz_route_inventory.py"

    for inventoried in ("gpuwm/core/moist.py", "gpuwm/core/dycore.py"):
        assert gate not in importers.get(inventoried, set()), (
            f"{gate} now has a direct import edge from {inventoried}.  The "
            "ALWAYS list exists because it did not; if that has changed, "
            "re-state the premise rather than deleting the entry")

    # ...and it is on the ALWAYS list, which is the coverage that closes it.
    assert gate in fastfix.read_manifest(ALWAYS_LIST)

    # The positive control: the index is not simply empty for those files.
    assert importers.get("gpuwm/core/moist.py"), (
        "no test imports gpuwm/core/moist.py, so the assertion above passes "
        "vacuously and proves nothing")
