"""The selector's cost model is measured the way the battery actually runs.

THE ACCIDENTAL COST THIS ANSWERS
--------------------------------
``tools/battery/fastfix.py --record`` spawns ONE PYTEST PROCESS PER FILE and
records the wall clock of each process.  The battery does not run that way: it
passes the whole list to a single invocation.  Measured on this box, an
invocation costs about 2.45 s of interpreter start, plugin load and conftest
import before any test runs -- the 172-file stage-1 leg measured 2,252 s as
172 processes against 1,441.68 s as one, a difference of 421 s, 29% of the
leg, spent on nothing the battery pays.

So every row in ``tools/battery/durations.json`` carried about 2.45 s of cost
the battery never incurs.  That is a rounding error on a 300 s file and it is
50-100% on a 2 s one -- and 100 of the 172 stage-1 files are under 5 s, which
is exactly the end of the range where ``fastfix``'s cheapest-first ordering is
decided.  A selector sorting on a systematically wrong number puts the red
later than it needs to be, which is the one thing the ordering exists to stop.

``--record-from`` reads the ``--durations=0`` report of a batched run instead:
a run that already happened, in the arrangement the battery uses, at no extra
cost.  These tests hold that parser to what pytest actually emits.

WHAT IT DOES NOT CLAIM
----------------------
pytest attributes only setup, call and teardown.  Collection -- which is where
a module's import cost lands -- is not in the report, so a batched row is
EXECUTION seconds and understates an import-heavy file.  That is stated in the
function's docstring and here, and it is still the better of the two numbers
for ordering, because the term it omits is smaller and more uniform than the
2.45 s the per-process spelling adds.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _fastfix():
    """Load ``tools/battery/fastfix.py`` by path.

    ``tools.battery`` resolves through a package whose parent is shadowed by
    an unrelated ``tools`` on this box's site-packages; the path is explicit
    and asserted so the gate cannot end up testing another tree's selector.
    """

    path = REPOSITORY_ROOT / "tools" / "battery" / "fastfix.py"
    assert path.is_file(), f"{path} is missing; it is the selector"
    spec = importlib.util.spec_from_file_location("gpuwm_fastfix", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gpuwm_fastfix"] = module
    spec.loader.exec_module(module)
    assert pathlib.Path(module.__file__).resolve() == path
    return module


fastfix = _fastfix()

#: A verbatim shape of pytest's ``--durations=0`` tail, including the header
#: line, a line for each of the three phases, and trailing summary prose that
#: must not be mistaken for a measurement.
REPORT = """============================= slowest durations ==============================
2.50s call     tests/test_alpha.py::test_one
0.40s setup    tests/test_alpha.py::test_one
0.10s teardown tests/test_alpha.py::test_one
1.25s call     tests/test_beta.py::test_two[case-3]
0.05s call     tilestream/test_gamma.py::TestClass::test_three
(3 durations < 0.005s hidden.  Use -vv to show these durations.)
======================== 4 passed, 1 skipped in 9.99s =========================
"""


def test_the_three_phases_of_one_test_are_summed_into_its_file() -> None:
    """A fixture is part of what a file costs.

    Counting only ``call`` would price ``tests/test_domain_wizard.py`` -- 229
    tests, 20-30% of the whole stage-1 leg -- at whatever fraction of its time
    is not fixture setup, which is the opposite of what the ordering needs.
    """

    seconds = fastfix.durations_from_batched_report(REPORT)
    assert seconds["tests/test_alpha.py"] == 3.0


def test_every_file_in_the_report_gets_a_row() -> None:
    seconds = fastfix.durations_from_batched_report(REPORT)
    assert sorted(seconds) == ["tests/test_alpha.py", "tests/test_beta.py",
                               "tilestream/test_gamma.py"]
    assert seconds["tests/test_beta.py"] == 1.25
    assert seconds["tilestream/test_gamma.py"] == 0.05


def test_prose_lines_are_not_measurements() -> None:
    """The header, the hidden-durations note and the summary all end in `s`.

    A looser pattern reads ``in 9.99s`` out of the summary line and invents a
    file called ``9.99s``, which would then sort first and be run first.
    """

    seconds = fastfix.durations_from_batched_report(REPORT)
    assert all(path.endswith(".py") for path in seconds), seconds


def test_a_report_with_no_durations_records_nothing() -> None:
    """A parser that matches nothing must say so, not write an empty receipt.

    ``record_durations_batched`` refuses on an empty parse; the alternative is
    a durations.json emptied by a mis-captured log, after which every file is
    priced at the default and the ordering silently stops being measured.
    """

    assert fastfix.durations_from_batched_report(
        "======== 4 passed in 9.99s ========\n") == {}


def test_windows_node_ids_are_normalised_to_forward_slashes() -> None:
    """pytest prints the path with the platform separator.

    The battery lists, the census and durations.json are all forward-slash, so
    a backslash row would never match the file it measured and every stage-1
    entry would price as unmeasured on the box the cut runs on.
    """

    seconds = fastfix.durations_from_batched_report(
        "0.75s call     tests\\test_delta.py::test_four\n")
    assert seconds == {"tests/test_delta.py": 0.75}


def test_the_recorded_receipt_is_still_readable_by_the_selector() -> None:
    """The two halves of the receipt have to agree on their shape.

    ``load_durations`` reads ``payload["seconds"]`` as ``{path: float}``; this
    asserts the tracked file still is that, so a hand edit or a half-written
    record cannot leave the selector silently pricing everything at the
    default.
    """

    durations = fastfix.load_durations()
    assert durations, (
        "tools/battery/durations.json parsed to nothing, so fastfix prices "
        "every selected file at the default and its cheapest-first ordering "
        "is not an ordering")
    assert all(isinstance(path, str) and path.endswith(".py")
               for path in durations)
    assert all(value >= 0.0 for value in durations.values())
