"""The GPU pytest shard's list, gated the way the stage-1 list is.

``tools/battery/gpu_shard_files.txt`` closes gap G2 of the 2026-08-13
test-estate audit: **there was no GPU pytest leg at all.**
``stage1_files.txt`` is CPU by construction, ``cargo_gates.txt`` is Rust,
``tiles_gates.txt`` names ``python -m tilestream.*`` modules, and no file
anywhere named a GPU pytest shard.  The consequence the audit measured is
that the entire WRF-parity dycore ran on no list -- 6th-order diffusion
including its boundary faces, both Smagorinsky closures, the open lateral
boundary, the acoustic solver, the mapped mass closure, the TKE budget --
and that ``tests/test_wrfout_conformance.py``, the only file that judges a
gpuwm wrfout with a real WRF reader, is 18 of 18 skipped on a CPU box.

This suite is hermetic and runs on the CPU legs, for the same reason
``tests/test_tiles_gate_manifest.py`` does: a gate that stops existing -- a
file renamed, an entry lost in a tidy-up -- must fail on every commit rather
than at a cut, on the node, when there is no time.

FIRST EXECUTION
---------------
Shard 1 was run for the first time in the project's history on 2026-08-13,
on a rented RTX 5080 (sm_120, driver 595.71.05, CUDA 13.2, NVRTC 13.2, cupy
14.1.1): **314 passed, 1 failed, 6 skipped in 121 s**.  The single failure is
``tests/test_pd_advection.py``'s pre-existing CPU-backend defect, which is
why that file is deliberately on the list.  Shard 2 ran 765 passed in 31 s.
"""

from __future__ import annotations

import pathlib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "gpu_shard_files.txt"

#: The line that divides the per-cut leg from the weekly one.
SHARD2_MARKER = "SHARD-2-BEGIN"


def _lines() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def _entries(which: str = "all") -> list[str]:
    """Entries, parsed exactly as the queue scripts parse the stage-1 list.

    Comment lines are prose about entries, never entries.  The audit's own
    membership check got this wrong on the stage-1 list -- a ``grep -F``
    reported ``tests/test_stage1_manifest.py`` as present when the string
    occurred only inside a comment -- so every consumer of these files parses
    rather than greps.
    """

    shard1: list[str] = []
    shard2: list[str] = []
    seen_marker = False
    for raw in _lines():
        line = raw.strip()
        if SHARD2_MARKER in line:
            seen_marker = True
            continue
        if not line or line.startswith("#"):
            continue
        (shard2 if seen_marker else shard1).append(line)
    if which == "shard1":
        return shard1
    if which == "shard2":
        return shard2
    return shard1 + shard2


def test_the_manifest_exists_and_lists_something() -> None:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    assert _entries("shard1"), (
        "tools/battery/gpu_shard_files.txt parsed to zero shard-1 entries; "
        "the GPU leg would run no tests and still exit 0, which is the "
        "green-on-nothing shape gap G2 was opened about")


def test_the_two_shards_are_both_populated_and_disjoint() -> None:
    """The per-cut leg and the weekly leg must stay distinguishable.

    A runner that could not tell them apart would either run the weekly GF
    block on every cut (which the audit's shard-design finding says is the
    wrong trade) or silently drop it altogether.
    """

    shard1, shard2 = _entries("shard1"), _entries("shard2")
    assert shard1 and shard2, (len(shard1), len(shard2))
    overlap = sorted(set(shard1) & set(shard2))
    assert not overlap, (
        f"{overlap} appear in both shards; the GPU leg would run them twice "
        "and the counts would overstate the coverage")


@pytest.mark.parametrize("entry", _entries())
def test_every_listed_path_exists(entry: str) -> None:
    assert not pathlib.PurePosixPath(entry).is_absolute(), entry
    assert "\\" not in entry, (
        f"{entry} uses a backslash; forward slashes only, so the same list "
        "works on the Windows cut box and the Linux GPU node")
    # tests/ or tilestream/: the tilestream pytest suites joined the shard
    # at the 2.5.0 gating pass (release blocker #5) -- their device halves
    # ran on no list before it.  The GPU-bound property below applies to
    # them unchanged, through the same conftest detector.
    assert entry.startswith(("tests/", "tilestream/")) \
        and entry.endswith(".py"), entry
    assert (REPOSITORY_ROOT / entry).is_file(), (
        f"{entry} is listed in tools/battery/gpu_shard_files.txt but does not "
        "exist; pytest would fail the whole GPU leg on the missing argument")


def test_no_entry_is_listed_twice() -> None:
    entries = _entries()
    duplicates = sorted({e for e in entries if entries.count(e) > 1})
    assert not duplicates, duplicates


def test_the_file_is_lf_only() -> None:
    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/gpu_shard_files.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


@pytest.mark.parametrize("entry", _entries())
def test_every_entry_is_actually_gpu_bound(entry: str) -> None:
    """The property that earns a place here, checked rather than asserted.

    A CPU-only file on this list is worse than useless: it makes the shard
    look broader than it is, and it burns node time that is rented by the
    hour.  ``tests/conftest.py`` is the authority on what "GPU-bound" means
    in this tree -- it marks a module ``gpu`` when cupy is imported at module
    scope, and individual functions when the import is inside them -- so this
    asks conftest rather than pattern-matching the filename.
    """

    from conftest import _cupy_scope

    whole, functions = _cupy_scope(str(REPOSITORY_ROOT / entry))
    assert whole or functions, (
        f"{entry} opens no CUDA device by conftest's own detector, so it is "
        "already covered by a CPU leg and does not belong on the GPU shard.  "
        "If it reaches a device transitively, say so here and pin the reason")


def test_the_dycore_parity_files_the_audit_named_are_all_present() -> None:
    """Gap G2's own enumeration, pinned so the list cannot quietly shrink.

    These are the files the audit measured as 100% skipped on CPU, by name
    and by test count: openbc 47, diff6_boundary_face 43, diff6 39, smag3d
    19, terrain_dynamics 19, smag2d 18, small_step_lateral_wrap 14,
    mapped_mass_closure 12, acoustic 11, tke_budget 11, tke_km2 10.  Every
    one of those counts was reproduced on the first GPU run.
    """

    required = {
        "tests/test_openbc.py", "tests/test_diff6.py",
        "tests/test_diff6_boundary_face.py", "tests/test_smag2d.py",
        "tests/test_smag3d.py", "tests/test_terrain_dynamics.py",
        "tests/test_small_step_lateral_wrap.py",
        "tests/test_mapped_mass_closure.py", "tests/test_tke_budget.py",
        "tests/test_tke_km2.py", "tests/test_acoustic.py",
    }
    missing = sorted(required - set(_entries("shard1")))
    assert not missing, (
        f"{missing} were named by gap G2 as running on no list and have been "
        "dropped from shard 1.  Removing one is allowed, but it is an "
        "argument made in a commit message, not a tidy-up")


def test_the_wrf_reader_conformance_file_is_on_the_per_cut_shard() -> None:
    """The single entry that most justifies the leg existing.

    Its own docstring says every other wrfout test is self-consistency and
    this is the only file that judges a gpuwm wrfout with a WRF reader.  It
    is 18 of 18 skipped on a CPU box, so a CPU-only battery never asks the
    one question a user's tooling will ask.
    """

    assert "tests/test_wrfout_conformance.py" in _entries("shard1")


def test_the_weekly_shard_is_the_gf_family() -> None:
    """Shard 2's membership follows a stated rule, not a mood.

    The audit's shard-design finding is that defect density is inverted
    against test mass: frozen bit-exact WRF leaf ports carry the tests and
    almost none of the defects.  GF cumulus is 376 GPU tests and nine of its
    ten files have never co-moved with a fix, which makes it the right first
    thing to move off the per-cut cadence -- and the wrong thing to delete.
    """

    shard2 = _entries("shard2")
    assert shard2, "shard 2 is empty"
    for entry in shard2:
        assert pathlib.PurePosixPath(entry).name.startswith("test_gf_"), (
            f"{entry} is in the weekly shard but is not part of the GF "
            "cumulus family, which is the only rule shard 2 currently "
            "states.  Widen the rule in the file's header, or move the entry")
