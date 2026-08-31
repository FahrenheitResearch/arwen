"""The mutation gate's two lists stay readable, honest and narrow.

WHAT BREAKS WITHOUT THIS
    tools/battery/run_mutation_gate.py is expensive enough that nobody
    runs it to find out whether its manifest still parses.  Three
    concrete breakages, each of which has a sibling that already
    happened in this tree:

    * A package listed in tools/battery/mutation_gates.txt that no
      longer exists, or that lost its test leg in
      tools/battery/cargo_gates.txt.  cargo-mutants reports 100%
      survival for a package with no tests, and 100% survival printed in
      a matrix reads like a catastrophic finding when it is an absence.
      Worse in `block` mode: every mutant in it is fresh, so the gate is
      red for a reason that has nothing to do with the commit.

    * A debt line in tools/battery/mutation_survivors.txt whose file no
      longer exists, or whose package is not on the gate.  A debt entry
      that outlives its code is amnesty granted to nothing, and it hides
      the fact that the hole it recorded was never closed -- the same
      way a stale _CRLF_DEBT entry would.

    * The debt list growing without anybody deciding to grow it.  The
      list is a work list; a widening is a ruling.  This test pins the
      shape so a widening is visible in review as an edit to a file
      whose header says what an edit means, rather than as a number that
      moved.

    None of this needs a Rust toolchain, which is the point: the checks
    that can run without cargo run in the ordinary pytest battery, and
    the run that needs cargo is a separate leg.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BATTERY = REPOSITORY_ROOT / "tools" / "battery"
MANIFEST = BATTERY / "mutation_gates.txt"
DEBT = BATTERY / "mutation_survivors.txt"

#: How many allowed survivors tools/battery/mutation_survivors.txt is
#: permitted to carry.  It is the ratchet's visible half.
#:
#: The debt list is a thousand generated lines, and the failure it invites
#: is somebody appending one more of them instead of writing the test the
#: runner just asked for: in a file that size a single added line is
#: invisible in review.  Pinning the count here makes that append a
#: two-file change, one of which is a hand-edited integer in a test whose
#: docstring says what raising it means.
#:
#: LOWERING IT IS THE WORK.  Every entry removed is a mutation the suite
#: now notices; drop the number to match.  Raising it says a hole was
#: accepted rather than closed, and that is a ruling, not a commit.
DEBT_CEILING = 986


def _load(name: str):
    """Import a battery runner by path; they are scripts, not a package."""

    path = BATTERY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_battery_{name}", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load("run_mutation_gate")
cargo_gates = _load("run_cargo_gates")


@pytest.fixture(scope="module")
def entries():
    return gate.read_manifest(MANIFEST)


@pytest.fixture(scope="module")
def debt():
    return gate.read_debt(DEBT)


def test_the_manifest_parses_and_is_not_empty(entries):
    assert entries, f"{MANIFEST} lists no packages"


def test_every_listed_package_is_a_workspace_member(entries):
    """A package cargo cannot resolve makes the whole run exit 3."""

    for entry in entries:
        crate = (REPOSITORY_ROOT / entry.workspace / "crates" /
                 entry.package / "Cargo.toml")
        assert crate.is_file(), (
            f"{MANIFEST} lists {entry.package} in {entry.workspace}, but "
            f"{crate.relative_to(REPOSITORY_ROOT).as_posix()} does not "
            "exist")


def test_every_listed_package_also_has_a_test_leg(entries):
    """Survival is only a finding where tests exist to survive.

    cargo-mutants reports every mutant of an untested package as missed.
    On the `block` list that is a red gate on day one; on the `report`
    list it is a 100% survival rate printed as if it were a measurement.
    Either way the number says nothing about the code.
    """

    tested = {(e.workspace, e.package)
              for e in cargo_gates.read_manifest()}
    for entry in entries:
        assert (entry.workspace, entry.package) in tested, (
            f"{MANIFEST} lists {entry.package}, which has no entry in "
            "tools/battery/cargo_gates.txt.  A package with no test leg "
            "has nothing for a mutant to survive: every mutant would be "
            "reported missed, which is an absence dressed up as a "
            "finding.")


def test_no_package_is_listed_twice(entries):
    seen: dict[str, str] = {}
    for entry in entries:
        assert entry.package not in seen, (
            f"{MANIFEST} lists {entry.package} twice, once as "
            f"{seen[entry.package]} and once as {entry.mode}.  Two lines "
            "for one package let a demotion from block to report hide "
            "inside a merge.")
        seen[entry.package] = entry.mode


def test_at_least_one_package_actually_blocks(entries):
    """A manifest of nothing but `report` is a gate that cannot fail."""

    blocking = [e.package for e in entries if e.mode == gate.GATE_MODE]
    assert blocking, (
        f"{MANIFEST} has no `{gate.GATE_MODE}` entries, so a fresh "
        "surviving mutant could not fail any run.  Demoting the last "
        "blocking package is how this gate would stop existing without "
        "anybody deleting it.")


def test_the_debt_list_parses(debt):
    """Malformed debt is refused loudly, never skipped quietly."""

    assert isinstance(debt, dict)


def test_the_debt_list_does_not_grow(debt):
    """The ratchet: allowed survivors may fall, never rise unasked."""

    assert len(debt) <= DEBT_CEILING, (
        f"{DEBT} carries {len(debt)} allowed survivors, above the "
        f"{DEBT_CEILING} this suite pins.  Each entry is a mutation the "
        "Rust suites do not notice; an added one is a hole accepted "
        "rather than closed.  If a package was genuinely re-baselined "
        "(run_mutation_gate.py --baseline), say so and raise DEBT_CEILING "
        "deliberately; if the gate asked for a test, write the test.")
    assert len(debt) == DEBT_CEILING, (
        f"{DEBT} is down to {len(debt)} allowed survivors from "
        f"{DEBT_CEILING}.  That is the good direction -- lower "
        "DEBT_CEILING in this file to match, so the next entry that "
        "appears is caught.")


def test_every_debt_entry_names_a_file_that_still_exists(entries, debt):
    workspace = {e.package: e.workspace for e in entries}
    for key in debt:
        assert key.package in workspace, (
            f"{DEBT} allows a survivor in {key.package}, which "
            f"{MANIFEST} does not list.  A debt entry for a package the "
            "gate never runs is amnesty granted to nothing, and it hides "
            "that the hole was never closed.")
        path = REPOSITORY_ROOT / workspace[key.package] / key.file
        assert path.is_file(), (
            f"{DEBT} allows a survivor in {key.file}, which does not "
            "exist.  Either the file moved and the entry should move "
            "with it, or the code went away and the entry should go with "
            "it.")


def test_debt_entries_are_only_for_blocking_packages(entries, debt):
    """`report` packages are not gated, so debt for them means nothing."""

    modes = {e.package: e.mode for e in entries}
    for key in debt:
        assert modes.get(key.package) == gate.GATE_MODE, (
            f"{DEBT} carries an entry for {key.package}, which is "
            f"`{modes.get(key.package)}` and therefore never fails a run. "
            "An allowance against a gate that cannot fire is dead text "
            "that makes the list look longer than the debt is.")


def test_a_survivor_not_on_the_list_is_fresh(entries, debt):
    """The ratchet's actual arithmetic, checked without a toolchain.

    The tally is the one thing in the runner that decides red from
    green, and it is nine lines that nothing else exercises.  Here it is
    driven directly: an observation matching a debt line is forgiven, a
    second copy of it is not, and one the list has never seen is fresh.
    """

    package = next(e.package for e in entries if e.mode == gate.GATE_MODE)
    listed = gate.MutantKey(package=package, file="src/lib.rs",
                            function="already_known", genre="BinaryOperator",
                            replacement="<")
    unlisted = gate.MutantKey(package=package, file="src/lib.rs",
                              function="brand_new", genre="BinaryOperator",
                              replacement="<")
    selected = [e for e in entries if e.package == package]
    observation = gate.Observation(key=listed, name="listed",
                                   summary="MissedMutant")
    fresh = gate.Observation(key=unlisted, name="fresh",
                             summary="MissedMutant")
    caught = gate.Observation(key=unlisted, name="caught",
                              summary="CaughtMutant")

    forgiven = gate.tally(selected, [observation], {listed: 1})
    assert forgiven[package].survived == 1
    assert forgiven[package].fresh == []

    twice = gate.tally(selected, [observation, observation], {listed: 1})
    assert twice[package].survived == 2
    assert len(twice[package].fresh) == 1, (
        "a second survivor under a key allowed once must be fresh, or "
        "one debt line silently forgives an unbounded number of new "
        "holes in the same function")

    unknown = gate.tally(selected, [fresh], {listed: 1})
    assert len(unknown[package].fresh) == 1

    detected = gate.tally(selected, [caught], {})
    assert detected[package].caught == 1
    assert detected[package].fresh == []


def test_unviable_and_timeout_are_not_counted_as_survivors(entries):
    """A mutant that did not compile proves nothing either way.

    Counting Unviable as caught would inflate the mutation score;
    counting it as survived would make the gate red for code that cannot
    exist.  Timeout means the mutant hung the suite, which is a
    detection.
    """

    package = next(e.package for e in entries if e.mode == gate.GATE_MODE)
    selected = [e for e in entries if e.package == package]
    key = gate.MutantKey(package=package, file="src/lib.rs", function="f",
                         genre="FnValue", replacement="Default::default()")
    results = gate.tally(selected, [
        gate.Observation(key=key, name="u", summary="Unviable"),
        gate.Observation(key=key, name="t", summary="Timeout"),
    ], {})
    assert results[package].unviable == 1
    assert results[package].timeout == 1
    assert results[package].survived == 0
    assert results[package].caught == 0
    assert results[package].fresh == []


def test_the_grib_core_path_dependency_is_staged(entries):
    """The blocker that makes the staging exist, pinned as a fact.

    cargo-mutants' own copy mode copies only the workspace directory.
    Four crates in tools/rustwx reach grib-core through
    ../../../grib1_bridge/vendor/grib-core, which escapes it, so the
    baseline build dies before a single mutant runs.  The runner finds
    escaping path dependencies by reading manifests rather than by
    naming grib-core; if that discovery ever returns nothing, the copies
    it stages are missing a dependency and every run exits 3 with a
    message about a missing Cargo.toml.
    """

    workspace = REPOSITORY_ROOT / entries[0].workspace
    escaping = gate.escaping_path_dependencies(workspace)
    assert escaping, (
        "no escaping path dependency was found under "
        f"{entries[0].workspace}.  If the vendoring genuinely changed, "
        "this test should change with it; if the parser broke, every "
        "mutation run is about to fail its baseline build.")
    for path in escaping:
        assert path.is_dir(), path
