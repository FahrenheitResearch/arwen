"""The release battery's stage-1 curated list, kept honest in the repository.

``tools/battery/stage1_files.txt`` is the list the battery's stage-1 leg
runs.  Until task #122 it existed only as a ``$stage1`` array inside the
per-assembly scratchpad queue scripts, propagated by copy from one
assembly to the next: a lost scratchpad silently lost every amendment
ever made to it, and nothing anywhere failed when it did.

This suite is the gate on the file that replaced that.  It checks the
mechanical properties the queue scripts depend on -- every path resolves,
nothing is listed twice, every entry is a repository-relative test file --
and it pins the two entries whose absence caused a real miss, so a
"tidy-up" that drops one of them has to argue with a named incident
rather than with a bare list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "stage1_files.txt"


def _lines() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def _entries() -> list[str]:
    """The manifest read exactly the way the queue scripts read it.

    The PowerShell in the manifest header is ``Get-Content | Trim() |
    Where-Object { $_ -and -not $_.StartsWith('#') }``.  This is that,
    so a file this suite accepts is a file the battery can consume.
    """

    stripped = (line.strip() for line in _lines())
    return [line for line in stripped if line and not line.startswith("#")]


def _reasons() -> dict[str, str]:
    """Each entry mapped to the comment block written directly above it.

    A blank line ends a block, which is what keeps the file header from
    attaching itself to the first entry.
    """

    reasons: dict[str, str] = {}
    block: list[str] = []
    for raw in _lines():
        line = raw.strip()
        if not line:
            block = []
        elif line.startswith("#"):
            block.append(line.lstrip("#").strip())
        else:
            reasons[line] = " ".join(part for part in block if part)
            block = []
    return reasons


def test_the_manifest_exists_and_lists_something() -> None:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    assert _entries(), (
        "tools/battery/stage1_files.txt parsed to zero entries; the battery's "
        "stage-1 leg would run no tests and still exit 0")


@pytest.mark.parametrize("entry", _entries())
def test_every_listed_path_exists(entry: str) -> None:
    assert (REPOSITORY_ROOT / entry).is_file(), (
        f"{entry} is listed in tools/battery/stage1_files.txt but does not "
        "exist; pytest would fail the whole stage-1 leg on the missing "
        "argument, so fix the path or drop the entry")


def test_no_entry_is_listed_twice() -> None:
    entries = _entries()
    duplicates = sorted({e for e in entries if entries.count(e) > 1})
    assert not duplicates, (
        f"tools/battery/stage1_files.txt lists {duplicates} more than once; "
        "pytest would collect those files twice and the leg's counts would "
        "overstate the coverage")


@pytest.mark.parametrize("entry", _entries())
def test_entries_are_repository_relative_test_files(entry: str) -> None:
    assert not Path(entry).is_absolute(), (
        f"{entry} is absolute; the queue scripts join these against the cut "
        "worktree, so entries have to be repository-relative")
    assert "\\" not in entry, (
        f"{entry} uses a backslash; forward slashes only, so the same list "
        "works on the Windows cut box and a Linux runner")
    # Two trees, both collected by pyproject's testpaths.  tilestream/
    # joined at the 2.5.0 gating pass (release blocker #5): its pytest
    # suites ran on no list at all, and the check that used to read "not
    # under tests/" would have refused the entries that fixed that.  The
    # breakage this line still prevents is unchanged -- a path outside the
    # repository's test trees is a typo the leg discovers only at runtime.
    assert entry.startswith(("tests/", "tilestream/")), (
        f"{entry} is not under tests/ or tilestream/")
    assert entry.endswith(".py"), f"{entry} is not a Python test file"


def test_the_file_is_lf_only() -> None:
    """``.gitattributes`` says ``* -text``: disk bytes are committed bytes.

    Nothing hashes this file today, but the promise is repository-wide and
    a CRLF copy is how it would quietly stop being true.
    """

    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/stage1_files.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")


def test_the_composition_suite_is_listed() -> None:
    """Pinned by the 1.8.5 assembly's real red (task #122).

    ``tests/test_physics_registry_composition.py`` gates the artifacts
    DERIVED from the physics registry.  On the 1.8.5 tree its
    ``test_ground_truth_receipt_regenerates_byte_for_byte`` failed --
    ``docs/public/receipts/F2-ground-truth.json`` is regenerated from the
    registry and had not been landed alongside the registry edit.  The
    stage-1 leg reported 0 failures on that same tree because this file
    was on no leg's list; only the per-cut lane union caught it, and a
    lane union is assembled by hand each cut, so it is not a gate.
    """

    entry = "tests/test_physics_registry_composition.py"
    assert entry in _entries(), (
        f"{entry} was dropped from tools/battery/stage1_files.txt.  It is "
        "there because the 1.8.5 assembly went red on it and stage 1 missed "
        "it entirely; a registry edit reaches the F2 receipts on every cut")


def test_the_physics_registry_suite_is_listed() -> None:
    """Pinned by task #121, the amendment that a lost scratchpad would eat.

    ``tests/test_physics_registry.py`` is the gate on the registry itself
    and its citation checker.  It was added to the curated list during the
    1.8.5 assembly, when the list still lived only in a scratchpad script.
    """

    entry = "tests/test_physics_registry.py"
    assert entry in _entries(), (
        f"{entry} was dropped from tools/battery/stage1_files.txt.  Task "
        "#121 added it so stage 1 has a gate on the registry itself")


#: The radiation pair.  A refusal gate and a runs gate are not
#: interchangeable, and the list carried only the first one.
RADIATION_REFUSED = "tests/test_nocturnal_radiation_guard.py"
RADIATION_RUNS = "tests/test_wrf_legacy_radiation.py"


def test_the_radiation_pair_is_listed_in_both_directions() -> None:
    """Both halves, or the gate is green on radiation that does nothing.

    Found by the green-on-nothing gate audit (2026-08-09).  Stage 1
    listed ``test_nocturnal_radiation_guard.py`` -- proof that a
    shortwave-on/longwave-off pairing is REFUSED -- and nothing that
    proves radiation RUNS.  Those are different claims.  On a tree whose
    radiation step had stopped producing heating the refusal would still
    fire, every listed suite would still pass, and the battery would
    report a clean stage 1: a guard is not a measurement.

    ``test_wrf_legacy_radiation.py`` is the measurement, on the CPU:
    atmosphere -> stock Dudhia adapter -> the production PhysicsDriver
    radiation seam, asserting a positive shortwave heating rate and a
    positive SWDOWN through the real driver, with the exact-zero night
    arm beside it so the instrument is exercised both ways.
    """

    entries = _entries()
    assert RADIATION_REFUSED in entries, (
        f"{RADIATION_REFUSED} was dropped from "
        "tools/battery/stage1_files.txt; it is the half that proves a bad "
        "radiation pairing is refused")
    assert RADIATION_RUNS in entries, (
        f"{RADIATION_RUNS} was dropped from tools/battery/stage1_files.txt.  "
        f"Without it stage 1 proves only that a bad radiation config is "
        f"refused ({RADIATION_REFUSED}) and never that radiation produces "
        "heating at all -- which is exactly the shape of gate the "
        "2026-08-09 audit was called to close")


def test_the_radiation_runs_entry_carries_its_reason_inline() -> None:
    """Same amendment discipline as the two task-numbered entries.

    This one names an incident rather than a ledger number, which the
    header explicitly allows ("the task OR the incident"), so it is
    checked here rather than folded into the ``#12`` parametrization
    below.
    """

    reason = _reasons().get(RADIATION_RUNS, "")
    assert reason, (
        f"{RADIATION_RUNS} has no comment block above it in "
        "tools/battery/stage1_files.txt; additions carry their reason inline")
    assert "green-on-nothing" in reason, (
        f"the comment above {RADIATION_RUNS} does not name the audit that "
        f"earned it: {reason!r}")


#: Every suite whose subject is the physics registry or an artifact
#: derived from it.  A scheme port edits the registry on every lane, so
#: this is the group stage 1 has repeatedly been blind to -- once really
#: (the 1.8.5 receipt, task #122) and twice latently (task #128 found the
#: other two off the list entirely).  Pinned as a GROUP so the next
#: registry-derived suite has an obvious place to join.
REGISTRY_DERIVED_ENTRIES = (
    "tests/test_physics_registry.py",
    "tests/test_physics_registry_composition.py",
    "tests/test_evidence_axes.py",
    "tests/test_native_wrf_distribution.py",
)


@pytest.mark.parametrize("entry", REGISTRY_DERIVED_ENTRIES)
def test_every_registry_derived_gate_is_listed(entry: str) -> None:
    """The group above, each member required by name.

    Dropping one is allowed -- but it has to be an argument about that
    suite, made in a commit message, not a line that disappears in a
    tidy-up.  ``test_evidence_axes.py`` and
    ``test_native_wrf_distribution.py`` joined at task #128: the WDM6 port
    edited the registry and added a CUDA translation unit, and neither
    consequence had a stage-1 gate.
    """

    assert entry in _entries(), (
        f"{entry} is not in tools/battery/stage1_files.txt.  It gates the "
        "physics registry or an artifact derived from it, and every lane "
        "that ports a scheme edits the registry")


@pytest.mark.parametrize(
    "entry",
    ["tests/test_physics_registry.py",
     "tests/test_physics_registry_composition.py"])
def test_the_pinned_entries_carry_their_reason_inline(entry: str) -> None:
    """The amendment discipline, enforced where it was written down.

    The header asks every addition to name the task or incident that
    earned it, on the lines directly above.  The two entries this suite
    pins are the worked examples; checking them keeps the convention
    visible to whoever adds the next one.
    """

    reason = _reasons().get(entry, "")
    assert reason, (
        f"{entry} has no comment block above it in "
        "tools/battery/stage1_files.txt; additions carry their reason inline")
    assert "#12" in reason, (
        f"the comment above {entry} does not name the task that earned it: "
        f"{reason!r}")


#: The release machinery's own gates.  Pinned as a GROUP, like the
#: registry-derived one, so the next release-tooling suite has an obvious
#: place to join.
RELEASE_MACHINERY_ENTRIES = (
    "tests/test_verify_release_artifacts.py",
    "tests/test_release_snapshot_front_door.py",
    "tests/test_release_snapshot_machine_paths.py",
    "tests/test_release_snapshot_modes.py",
    "tests/test_release_notes_are_public_facing.py",
)


@pytest.mark.parametrize("entry", RELEASE_MACHINERY_ENTRIES)
def test_every_release_machinery_gate_is_listed(entry: str) -> None:
    """Earned by the 2.1.0 assembly, where the miss was nearly terminal.

    The tenth bundled artifact is vendored: deliberately unstamped, and
    proved instead by its declared contract marker.  Commit 6775e7a0a
    taught that to ``tools/build_bridge_bundle.py`` and not to
    ``tools/verify_release_artifacts.py``, so seven cases of
    ``tests/test_verify_release_artifacts.py`` were red on the assembled
    branch and the verifier would have refused the bundle in the prepare
    job -- after the tag was pushed, mid cut.  Nothing caught it because
    no leg ran that file: the union battery is assembled by hand each
    cut, and the curated list did not carry the release machinery at all.

    These suites are the cheapest possible insurance -- CPU-only, no
    card, fixtures built through the real packer and the real snapshot
    builder -- against the one class of defect a cut cannot route around,
    because by the time it fires the tag already exists.
    """

    assert entry in _entries(), (
        f"{entry} is not in tools/battery/stage1_files.txt.  It gates the "
        "machinery a release cut itself runs, and a defect there surfaces "
        "between the tag and PyPI, where there is no cheap way back")
