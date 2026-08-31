"""Fail a battery leg when a file it was told to run stopped speaking.

    python -m pytest -p tools.battery.no_silent_deselection <files...>

THE BREAKAGE THIS PREVENTS
--------------------------
The 2026-08-28 fault-injection audit put ``pytestmark = pytest.mark.gpu`` at
the top of ``tests/test_ruc.py``.  All sixty RUC bitwise-oracle tests -- the
suite that detects a ONE-ULP change to Stefan-Boltzmann -- were deselected by
the battery's own ``-m "not gpu"``, and the leg stayed green.  Nothing
noticed: not ``test_gpu_marker_discipline.py`` (84 s), not
``test_module_skip_placement.py``, not ``test_stage1_manifest.py``, not
``test_health_field_census.py``.  Re-measured at ``b47a400a5`` on a
three-file leg::

    clean,    no guard : rc=0   68 passed,  1 deselected
    injected, no guard : rc=0    8 passed, 61 deselected   <- NOT CAUGHT
    injected, guard    : rc=1   SILENT DESELECTION: tests/test_ruc.py
    clean,    guard    : rc=0   68 passed,  1 deselected

The only signal was pytest's exit code 5, "no tests collected" -- and exit 5
was already normal on this list, because two entries
(``tests/test_anisotropic_mixing_w_stability.py``,
``tests/test_dycore_advective_forcing_export.py``) were entirely GPU-marked
and contributed nothing to the CPU leg.  A reader trained to accept exit 5
cannot see a third one arrive.  Both were moved to
``tools/battery/gpu_shard_files.txt`` in the commit that added this plugin,
so ``ZERO_COLLECT_ALLOWED`` is empty and every listed file must speak.

So the mark that silently retires a whole suite was invisible, and the way to
retire any suite was to add one line.

THE HALF-SUITE CASE, AND WHY THE CENSUS IS HERE
-----------------------------------------------
Zero is only the loudest version of the fault.  Mark HALF a file's tests, or
delete thirty of sixty, and the file still contributes items, so a
zero-collect check sees nothing.  ``tools/battery/list_census.json`` records
what each listed file contributed at ``b47a400a5``; this plugin compares the
collection the leg ALREADY PERFORMED against that floor.  It is the same
check done better, not an extra one: no test is run twice and nothing is
collected twice, so the floor costs no measurable wall clock.

A count that ROSE is fine and needs no argument.  A count that FELL is
coverage that left, and it fails the leg by name with both numbers.

WHAT IT COSTS
-------------
Nothing measurable.  It reads the collection the run already performed and
reports at session end; it runs no test, collects nothing extra and imports
nothing but ``json``.

LIMITS, STATED
--------------
* It considers only paths given as command-line arguments, so a ``-k`` run of
  a directory is unaffected.
* Under an explicit ``-k`` or ``--deselect`` the operator is deliberately
  running a subset, so the census floor is not applied -- the zero-collect
  half still is.
* Under ``pytest-xdist`` the controller process holds no items, so the guard
  cannot see the collection.  It says so on stdout rather than passing
  quietly; a guard that goes silent when the runner changes is the fault this
  file exists to catch.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections import Counter

#: Paths allowed to contribute nothing, each with the reason.
#: An entry here is a claim that the file's coverage lives on another leg.
#: Anything not listed is a failure, which is the point.
ZERO_COLLECT_ALLOWED: dict[str, str] = {
    # EMPTY, AND THAT IS THE POINT.  The two files that would have been here
    # -- tests/test_anisotropic_mixing_w_stability.py (0 of 2 collected on the
    # CPU leg) and tests/test_dycore_advective_forcing_export.py (0 of 9) --
    # were moved to tools/battery/gpu_shard_files.txt in the same commit that
    # added this plugin, because a fix retires its guards.  An entry added
    # here is a claim that the file's coverage lives on another leg; write the
    # leg's name in the reason, and delete the entry when the move happens.
}

CENSUS_RELATIVE = "tools/battery/list_census.json"


def _rel(path: str, root: pathlib.Path) -> str:
    try:
        return pathlib.Path(path).resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return pathlib.Path(path).as_posix()


def _floors(root: pathlib.Path, markexpr: str) -> dict[str, tuple[str, int]]:
    """``{path: (leg, count)}`` for every leg collected under ``markexpr``.

    A census that cannot be read yields no floors and the zero-collect half
    still runs, so this is never weaker than the guard without it.  That the
    census is PRESENT and well-formed is asserted by
    ``tests/test_battery_list_census.py``, which is where a missing instrument
    is meant to be caught -- not here, silently, at battery time.
    """

    path = root / CENSUS_RELATIVE
    try:
        census = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    floors: dict[str, tuple[str, int]] = {}
    for leg, body in census.get("legs", {}).items():
        if body.get("markexpr") != markexpr:
            continue
        for entry, count in body.get("files", {}).items():
            previous = floors.get(entry)
            if previous is None or count > previous[1]:
                floors[entry] = (leg, int(count))
    return floors


class _Guard:
    def __init__(self) -> None:
        self.requested: set[str] = set()
        self.contributed: Counter[str] = Counter()
        self.root = pathlib.Path(os.getcwd()).resolve()
        self.floors: dict[str, tuple[str, int]] = {}
        self.subset_run = False
        self.blind = ""

    def pytest_cmdline_main(self, config) -> None:
        self.root = pathlib.Path(str(config.rootpath)).resolve()
        for arg in config.args:
            head = arg.split("::")[0]
            if head.endswith(".py") and (self.root / head).is_file():
                self.requested.add(_rel(head, self.root))
        markexpr = getattr(config.option, "markexpr", "") or ""
        self.subset_run = bool(getattr(config.option, "keyword", "")
                               or getattr(config.option, "deselect", None))
        self.floors = {} if self.subset_run else _floors(self.root, markexpr)
        if getattr(config.option, "dist", "no") not in ("no", None) and not \
                hasattr(config, "workerinput"):
            self.blind = (
                "pytest-xdist controller: this process holds no collected "
                "items, so the silent-deselection guard and the census floor "
                "did not run on this leg")

    def pytest_collection_finish(self, session) -> None:
        """After every deselection, never during one.

        ``pytest_collection_modifyitems`` is the hook that PERFORMS marker
        deselection, so a plugin reading ``items`` there can see the list
        before ``-m`` has emptied it and record as covered a file whose tests
        will not run -- which is the exact failure this guard exists to
        catch, reintroduced inside the guard.  ``pytest_collection_finish``
        is called once, with the final set.
        """

        for item in session.items:
            self.contributed[_rel(str(item.path), self.root)] += 1

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if self.blind:
            print("\n" + self.blind)
            return
        lines: list[str] = []
        silent = sorted(self.requested - set(self.contributed)
                        - set(ZERO_COLLECT_ALLOWED))
        if silent:
            lines += ["", "SILENT DESELECTION -- these files were run and "
                          "contributed no tests:"]
            lines += [f"  {rel}" for rel in silent]
            lines.append(
                "Every test in each was deselected, skipped at collection, or "
                "removed.  The leg reported no failure for them because it "
                "ran none of them.  Either fix the marker/collection, or add "
                "the path to ZERO_COLLECT_ALLOWED in "
                "tools/battery/no_silent_deselection.py with the reason and "
                "the leg its coverage moved to.")

        shrunk = []
        for rel in sorted(self.requested):
            if rel in silent or rel not in self.floors:
                continue
            leg, floor = self.floors[rel]
            got = self.contributed.get(rel, 0)
            if got < floor:
                shrunk.append(f"  {rel}: {got} collected, "
                              f"{floor} recorded for the {leg!r} leg "
                              f"({floor - got} fewer)")
        if shrunk:
            lines += ["", "COVERAGE BELOW THE CENSUS -- these files collected "
                          "fewer tests than tools/battery/list_census.json "
                          "records:"]
            lines += shrunk
            lines.append(
                "Tests were marked out, skipped at collection, or deleted, and "
                "the leg reported a smaller number instead of a failure.  If "
                "the reduction is intended, re-record the census in the same "
                "commit and say which tests went and where they went to.")

        if not lines:
            return
        print("\n".join(lines))
        import pytest

        session.exitstatus = int(pytest.ExitCode.TESTS_FAILED)


#: One registration per session, whatever route loaded this file.
#: tests/conftest.py loads it BY PATH so that the guard is default-on, and a
#: battery leg may ALSO pass ``-p tools.battery.no_silent_deselection``.  Those
#: are two different module objects with two different ``pytest_configure``
#: functions, and pytest raises
#: ``ValueError: Plugin name already registered`` as an INTERNALERROR when the
#: second one registers -- measured 2026-08-29, and it takes the whole session
#: down before a test runs.  The name is the identity, so the name is the
#: check.
PLUGIN_NAME = "no_silent_deselection_guard"


def pytest_configure(config) -> None:
    if config.pluginmanager.hasplugin(PLUGIN_NAME):
        return
    guard = _Guard()
    guard.pytest_cmdline_main(config)
    config.pluginmanager.register(guard, PLUGIN_NAME)
