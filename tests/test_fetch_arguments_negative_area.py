"""A stored fetch argv whose area leads with a minus reaches the fetch.

Named breakage: ``gpuwm go`` handed its configuration's ``[fetch]`` table
to the fetch parser as ``--area -20.53,176.87,-14.46,-176.97``.  argparse
reads a leading-minus token as an option unless it is a bare number, so
every configuration south of the equator stopped at the fetch stage with
``argument --area: expected one argument`` before a byte was downloaded.
The typed command line never saw it, because ``main`` joins a coordinate
flag to its value; the doors that parse a stored argv did not.
"""

from __future__ import annotations

import gpuwm.fetch as fetch
from gpuwm import download_budget, runplan

HINTS = {"source": "gdas", "cycle": "2026-09-27T18", "hours": 1,
         "cadence": 1, "area": "-20.53,176.87,-14.46,-176.97"}


def test_the_go_fetch_stage_reads_an_area_south_of_the_equator(
        tmp_path, monkeypatch):
    seen = []

    def stand_in(args):
        seen.append((args.source, args.area))
        return 0

    monkeypatch.setattr(fetch, "fetch_main", stand_in)
    argv = runplan._fetch_arguments_from_hints(HINTS, out=tmp_path / "data")
    report = runplan._run_fetch(argv, tmp_path)
    assert seen == [("gdas", HINTS["area"])]
    assert report["arguments"] == argv


def test_a_run_plan_fetch_list_and_the_download_budget_read_it_too(tmp_path):
    argv = ["--source", "gdas", "--cycle", "2026-09-27T18", "--hours", "3",
            "--area", "-34.5,150.5,-33.0,152.0", "--out", str(tmp_path)]
    runplan._validate_fetch_arguments(argv)
    request = download_budget.request_from_arguments(argv)
    assert request is not None
    assert request["area"] == "-34.5,150.5,-33.0,152.0"


def test_a_value_that_is_not_a_coordinate_is_still_refused(tmp_path):
    # The join is for coordinates only: a flag swallowed as the area's
    # value is still the parser's refusal, as on the command line.
    argv = ["--source", "gdas", "--area", "--hours", "3", "--out",
            str(tmp_path)]
    assert download_budget.request_from_arguments(argv) is None
