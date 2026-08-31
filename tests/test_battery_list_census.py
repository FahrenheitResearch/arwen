"""Every battery-list entry has a census row, and every row is still listed.

THE BREAKAGE THIS PREVENTS
--------------------------
F20 of the 2026-08-28 fault-injection audit: delete ONE line from
``tools/battery/stage1_files.txt``.  The battery stops running that whole
suite and nothing anywhere says so.  Reproduced at ``b47a400a5``, three
separate entries, one at a time::

    entry removed                     tests/test_stage1_manifest.py
    ------------------------------    -----------------------------
    (clean)                           362 passed
    tests/test_water_overlay.py       360 passed   GREEN
    tests/test_doctor.py              360 passed   GREEN
    tests/test_speedrun.py            360 passed   GREEN

and ``tests/test_battery_route.py`` (29 passed),
``tests/test_fastfix_selector.py`` (34 passed) and
``tests/test_ci_test_replay.py`` (2 passed, 1 skipped) were green in every one
of those runs.  The manifest gate pins four entries BY NAME and counts
nothing, so the 168 it does not name can leave silently.  A leg that runs 171
files instead of 172 reports a smaller number that nobody compares against
anything.

WHAT THIS FILE IS
-----------------
``tools/battery/list_census.json`` records, per leg, the number of tests each
listed file contributed to that leg's collection at a named commit.  This file
checks the BIJECTION in both directions -- every entry has a row, every row is
still an entry -- so a deleted entry fails on the second direction and names
the coverage that left with it.

The counts themselves are enforced at RUNTIME by
``tools/battery/no_silent_deselection`` against the collection the leg already
performed, so the floor costs no wall clock.  This file is the static half: it
costs two file reads, it is the reason that plugin has something to compare
against, and it is what keeps the census an armed instrument rather than a
file nobody checks.

It imports nothing, collects nothing and runs nothing.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CENSUS_PATH = REPOSITORY_ROOT / "tools" / "battery" / "list_census.json"

#: The marker expression the CPU battery legs run under.  Pinned here as a
#: literal rather than read out of the census, because a census that records
#: whatever expression it was last run under cannot notice the expression
#: changing.  If the leg's expression legitimately changes, re-record the
#: census under the new one and change this line in the same commit.
LEG_MARKER_EXPRESSION = "not gpu and not slow and not network"


def _census() -> dict:
    assert CENSUS_PATH.is_file(), (
        f"{CENSUS_PATH} is missing.  It is the only record of how much "
        "coverage each battery-list entry contributes, and without it a "
        "deleted entry is invisible.")
    return json.loads(CENSUS_PATH.read_text(encoding="utf-8"))


def _listed(list_path: str) -> list[str]:
    text = (REPOSITORY_ROOT / list_path).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")]


CENSUS = _census()
LEGS = CENSUS["legs"]
#: ``(leg, entry)`` for every recorded row and for every listed entry, so a
#: failure names which leg lost it and shows up as its own node id.
ROWS = [(leg, entry) for leg, body in sorted(LEGS.items())
        for entry in sorted(body["files"])]
LISTED = [(leg, entry) for leg, body in sorted(LEGS.items())
          for entry in _listed(body["list"])]


def test_the_census_records_at_least_the_two_cpu_legs() -> None:
    """A census over an empty set of legs passes and protects nothing."""

    assert set(LEGS) >= {"stage1", "always"}, (
        f"the census names legs {sorted(LEGS)}; the two CPU battery legs are "
        "'stage1' and 'always' and both were recorded at b47a400a5.  A leg "
        "that leaves this file stops being counted, which is the fault this "
        "gate exists to catch, one level up.")


@pytest.mark.parametrize("leg,entry", ROWS, ids=lambda value: value)
def test_every_census_row_is_still_listed(leg: str, entry: str) -> None:
    """THE F20 DIRECTION.  A row with no entry is coverage that left."""

    recorded = LEGS[leg]["files"][entry]
    assert entry in _listed(LEGS[leg]["list"]), (
        f"{entry} is recorded in tools/battery/list_census.json as "
        f"contributing {recorded} tests to the {leg!r} leg, and it is no "
        f"longer on {LEGS[leg]['list']}.  That leg now runs {recorded} fewer "
        "tests and reports a smaller number instead of a failure.  If the "
        "removal is intended, delete the census row in the SAME commit and "
        "say in that commit where the coverage went -- a removal is not a "
        "tidy-up, which is what that list's own header already says.")


@pytest.mark.parametrize("leg,entry", LISTED, ids=lambda value: value)
def test_every_listed_entry_has_a_census_row(leg: str, entry: str) -> None:
    """The other direction.  An entry with no row is uncounted coverage.

    Without this the census could be kept green by never adding rows, and the
    list would drift out from under it one addition at a time.
    """

    assert entry in LEGS[leg]["files"], (
        f"{entry} is on {LEGS[leg]['list']} and has no row in "
        "tools/battery/list_census.json, so nothing records how much it "
        "contributes and its later removal would be invisible.  Re-record "
        "the census -- the command is in the file's _how_to_re_record field "
        "-- in the commit that adds the entry.")


@pytest.mark.parametrize("leg,entry", ROWS, ids=lambda value: value)
def test_no_census_row_records_zero_tests(leg: str, entry: str) -> None:
    """A row at zero is a line on a list that reads as coverage and is not.

    Measured at b47a400a5, before this landed:
    ``tests/test_anisotropic_mixing_w_stability.py`` collected 0 of 2 and
    ``tests/test_dycore_advective_forcing_export.py`` 0 of 9, both entirely
    GPU-marked files occupying stage-1 lines, and the second was on no GPU
    list either, so its nine tests ran on no leg at all.  Both moved to
    ``tools/battery/gpu_shard_files.txt``.  This assertion is what stops a
    third arriving and being recorded as normal.
    """

    assert LEGS[leg]["files"][entry] > 0, (
        f"{entry} contributes zero tests to the {leg!r} leg.  It occupies a "
        "line that reads as coverage and takes none.  Either its marks are "
        "wrong, or it belongs on the leg that does run it -- do not record "
        "the zero.")


@pytest.mark.parametrize("leg,entry", LISTED, ids=lambda value: value)
def test_every_listed_entry_is_a_file_that_is_here(leg: str, entry: str) -> None:
    """A list naming a path that is not here is a leg one file short.

    tests/test_stage1_manifest.py already holds this for the stage-1 list.  It
    is here as well because the ALWAYS list has no manifest gate at all: its
    own header names ``tests/test_always_manifest.py`` and NO SUCH FILE EXISTS
    in this tree (measured 2026-08-29, and the only file that reads
    always_files.txt is tests/test_fastfix_selector.py).  A census that
    verified rows against entries but never checked the entries were real
    would inherit the same hole.
    """

    assert (REPOSITORY_ROOT / entry).is_file(), (
        f"{LEGS[leg]['list']} names {entry} and that file is not in the tree. "
        " The leg is given a path that does not exist, or it quietly runs one "
        "file fewer than its list claims.  Fix the spelling, or remove the row "
        "and its census row in the same commit as the deletion.")


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_no_list_names_the_same_file_twice(leg: str) -> None:
    """A duplicate runs the file twice and reads as two decisions."""

    listed = _listed(LEGS[leg]["list"])
    repeated = sorted({entry for entry in listed if listed.count(entry) > 1})
    assert not repeated, (
        f"{LEGS[leg]['list']} names {repeated} more than once; the leg pays "
        "for them twice and the amendment history reads as two arguments "
        "where there was one.")


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_the_recorded_total_is_the_sum_of_its_rows(leg: str) -> None:
    """An arithmetic tripwire on the file itself."""

    body = LEGS[leg]
    assert body["collected_total"] == sum(body["files"].values()), (
        f"the {leg!r} leg records collected_total={body['collected_total']} "
        f"against rows summing to {sum(body['files'].values())}; the file was "
        "hand-edited and its two halves disagree, so neither can be trusted "
        "to say what left.")


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_leg_records_the_marker_expression_it_was_collected_under(
    leg: str,
) -> None:
    """Counts taken under a different ``-m`` are not comparable.

    ``-m "not gpu"`` alone collects the slow and network tests too, so a
    census recorded under it would set floors the real leg can never meet;
    one recorded under a NARROWER expression sets floors so low that the
    runtime check passes anything.
    """

    assert LEGS[leg]["markexpr"] == LEG_MARKER_EXPRESSION, (
        f"the {leg!r} leg's census was recorded under "
        f"{LEGS[leg]['markexpr']!r} and the battery runs "
        f"{LEG_MARKER_EXPRESSION!r}.  Re-record it under the expression the "
        "leg actually uses, or change LEG_MARKER_EXPRESSION here in the same "
        "commit as the leg changes.")


@pytest.mark.parametrize("leg", sorted(LEGS))
def test_every_leg_names_a_list_file_that_is_here(leg: str) -> None:
    assert (REPOSITORY_ROOT / LEGS[leg]["list"]).is_file(), (
        f"the {leg!r} leg's census names {LEGS[leg]['list']}, which is not in "
        "the tree; the census is measuring nothing.")


# ---------------------------------------------------------------------------
# The instrument, tested rather than assumed
# ---------------------------------------------------------------------------
#
# The census is only half of the guard.  The other half is
# tools/battery/no_silent_deselection.py, registered from tests/conftest.py so
# that a bare `pytest` catches a retired suite.  That registration is a hook
# that PRINTS and returns when it cannot load the plugin, deliberately, because
# a conftest must never be the reason a run cannot start -- and a printed
# warning inside a captured battery log is exactly the kind of signal the whole
# audit found people stop seeing.  So the arming is asserted here, where a
# failure is a failure.


def test_the_silent_deselection_guard_is_registered_in_this_session(
    pytestconfig,
) -> None:
    """The guard is loaded, in the very session this test runs in.

    This is not a check that a file exists; it asks pytest's own plugin manager
    whether the guard is live.  It therefore fails if the plugin file is gone,
    if tests/conftest.py stops registering it, if the load raised, or if the
    plugin is renamed -- every route by which the default could quietly stop
    being a default.
    """

    assert pytestconfig.pluginmanager.hasplugin("no_silent_deselection_guard"), (
        "the silent-deselection guard is not registered in this pytest "
        "session.  tests/conftest.py loads tools/battery/"
        "no_silent_deselection.py by path and registers it, and its failure "
        "path only prints.  Without it, `pytestmark = pytest.mark.gpu` on any "
        "file retires that whole suite and every leg stays green -- measured "
        "on tests/test_ruc.py's sixty bitwise oracles, rc=0.")


def test_the_guard_carries_no_stale_zero_collect_exemption() -> None:
    """An exemption is a written argument, and it dies with its reason.

    ZERO_COLLECT_ALLOWED excuses a listed file from contributing tests.  Two
    entries were removed in the commit that added the guard, because the files
    they excused were moved to tools/battery/gpu_shard_files.txt instead.  An
    entry that stays after its file is fixed reads as coverage nobody takes.
    """

    import importlib.util
    import sys

    path = (REPOSITORY_ROOT / "tools" / "battery"
            / "no_silent_deselection.py")
    assert path.is_file(), f"{path} is missing; the guard cannot load"
    spec = importlib.util.spec_from_file_location(
        "gpuwm_no_silent_deselection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_no_silent_deselection_under_test"] = module
    spec.loader.exec_module(module)

    for entry, reason in module.ZERO_COLLECT_ALLOWED.items():
        assert (REPOSITORY_ROOT / entry).is_file(), (
            f"ZERO_COLLECT_ALLOWED excuses {entry}, which is not in the tree; "
            "the exemption outlived the file it excused")
        assert reason.strip(), (
            f"ZERO_COLLECT_ALLOWED excuses {entry} with no reason.  An "
            "exemption without a written argument is an exit code with extra "
            "steps.")
        listed = [leg for leg, body in LEGS.items()
                  if entry in _listed(body["list"])]
        assert listed, (
            f"ZERO_COLLECT_ALLOWED excuses {entry} and no census leg lists it "
            "any more, so the exemption is excusing nothing.  Remove the row.")
