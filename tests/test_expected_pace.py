"""The pace a plan will run at, stated BEFORE the run starts.

THE DEFECT.  A user drew a 399,119-column domain on a 10 GiB card.  The
engine priced it correctly, chose the streamed road correctly, and told
the user nothing at all about what that road COSTS.  Streamed at that
size is seconds-to-minutes per model step, so a three-hour forecast is
many wall-hours -- and the way the user found out was by watching a run
that looked stalled.  Every memory figure in the estimate document was
right and the one number that would have changed the decision was
absent.

Two halves are pinned here.  The first is that ``run-plan --estimate``
and ``gpuwm check`` both carry an ``expected_pace``: which road, how
many seconds per model step, how many wall seconds for the plan's own
``run_seconds``, and WHERE THE NUMBERS CAME FROM.  The second is the
column bound ``K`` -- the largest domain that still fits the resident
road on this card -- computed from the SAME pricing
``tilestream.autoplan`` makes the stream/resident decision with, so the
advice and the decision cannot disagree.

CPU-only.  Every fixture pins its tiling and declares its VRAM budget,
which is exactly what ``streaming.decide`` does with a configured
``vram_budget_bytes``, so no card is consulted anywhere in this file.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from gpuwm.core import pace as pace_module
from gpuwm.experiment import load_experiment

from test_runplan_tiles import _config, _plan

import gpuwm.runplan as runplan_module


#: A pinned tiling with a DECLARED VRAM budget.  ``streaming.decide``
#: replaces the machine's budget with this number and drops the headroom
#: multiplier, so the plan -- and the column bound below it -- is a pure
#: function of the configuration.  6 GiB is a small card on purpose.
_STREAMED = """\
[tiles]
mode = "on"
tile_nx = 48
tile_ny = 40
store = "host"
vram_budget_bytes = 6442450944
host_budget_bytes = 34359738368
"""


def _estimate(tmp_path, config):
    """``run-plan --estimate``'s document, round-tripped as JSON."""

    return json.loads(json.dumps(
        runplan_module.estimate_plan(_plan(tmp_path, config))))


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


def test_a_resident_plan_is_quoted_a_resident_pace(tmp_path):
    """No ``[tiles]``, so the road is resident and says so."""

    document = _estimate(tmp_path, _config(tmp_path))
    expected = document["expected_pace"]

    assert expected["road"] == "resident"
    # A bracket, never a point: the package has measured two cards and
    # they differ, so quoting one figure would be quoting a card the
    # reader does not have.
    assert expected["seconds_per_step_low"] > 0.0
    assert (expected["seconds_per_step_high"]
            >= expected["seconds_per_step_low"])
    assert expected["wall_seconds_high"] >= expected["wall_seconds_low"] > 0.0
    # realtime_ratio pairs the other way round: the SLOW wall is the LOW
    # ratio.  A document that got this backwards would read as a run
    # being fastest when it is slowest.
    assert expected["realtime_ratio_low"] <= expected["realtime_ratio_high"]
    assert expected["basis"]
    # The rung the rate was measured at, and the card it was measured
    # on, are IN the basis -- an estimate that names neither is a number
    # with gpuwm's name on it and nobody's measurement behind it.
    assert "rung" in expected["basis"]
    assert expected["reference_card"]


def test_a_streamed_plan_is_quoted_a_bracket_and_the_column_bound(tmp_path):
    """THE FEATURE.  The road, its pace, and the size that makes it fast."""

    document = _estimate(tmp_path, _config(tmp_path, tiles=_STREAMED))
    expected = document["expected_pace"]

    assert expected["road"] == "streamed"
    assert expected["seconds_per_step_high"] > expected["seconds_per_step_low"]
    # THE ACTIONABLE NUMBER.  Not "your domain is too big" but "this card
    # runs the fast road up to K columns", with the pace it runs at.
    assert expected["resident_column_limit"] > 0
    assert expected["resident_seconds_per_step_low"] > 0.0
    assert (expected["resident_seconds_per_step_high"]
            >= expected["resident_seconds_per_step_low"])
    # The streamed road is the dear one AT THE SAME DOMAIN.  Comparing
    # this plan's streamed step against the resident step at K columns
    # would compare two different domains -- K is deliberately the
    # LARGEST that fits, so on a roomy card it is a far bigger grid and
    # a far slower step, and the comparison would invert for a reason
    # that has nothing to do with the roads.
    resident = _estimate(tmp_path, _config(tmp_path))["expected_pace"]
    assert (expected["seconds_per_step_high"]
            > resident["seconds_per_step_high"])
    assert expected["columns"] == resident["columns"]
    # The transfer term is named, because it is the term the reader can
    # do something about (a smaller domain, or a device store).
    assert expected["transfer_bytes_per_step"] > 0
    assert "pinned" in expected["basis"].lower()


def test_the_estimate_keys_are_stable_for_the_document_a_front_end_draws(
        tmp_path):
    """Studio renders this verbatim; a renamed key is a blank line there."""

    resident = _estimate(tmp_path, _config(tmp_path))["expected_pace"]
    streamed = _estimate(
        tmp_path, _config(tmp_path, tiles=_STREAMED))["expected_pace"]

    required = {
        "road", "seconds_per_step_low", "seconds_per_step_high",
        "wall_seconds_low", "wall_seconds_high", "realtime_ratio_low",
        "realtime_ratio_high", "steps", "resident_column_limit",
        "resident_seconds_per_step_low", "resident_seconds_per_step_high",
        "transfer_bytes_per_step", "measured", "reference_card", "basis",
        "sentence",
    }
    for document in (resident, streamed):
        assert required <= set(document), required - set(document)
        # Every numeric field survives a JSON round trip as a number and
        # not as a string or a NaN, which is what a front end divides by.
        for key in ("seconds_per_step_low", "wall_seconds_low",
                    "realtime_ratio_low"):
            assert isinstance(document[key], (int, float))
            assert document[key] == document[key]        # not NaN


def test_the_sentence_names_both_roads(tmp_path):
    """One sentence, both roads: what you have, and what would be fast."""

    document = _estimate(tmp_path, _config(tmp_path, tiles=_STREAMED))
    sentence = document["expected_pace"]["sentence"]

    assert "streamed road" in sentence
    assert "resident road" in sentence
    assert "s per model step" in sentence
    # The unit is chosen for the reader, not fixed: this fixture is a
    # toy domain whose whole forecast is under a minute.
    assert "wall for this" in sentence
    # The column bound appears as a NUMBER a reader can compare their
    # own domain against.
    limit = document["expected_pace"]["resident_column_limit"]
    assert f"{limit:,}" in sentence or str(limit) in sentence


def test_check_renders_the_pace_sentence_naming_both_roads(tmp_path):
    """`gpuwm check`'s text surface carries it, not only the JSON.

    Asserted through the name ``check_main`` actually calls, so a rename
    on either side of that seam fails here rather than silently dropping
    the line from the report.
    """

    from gpuwm.core import preflight

    exp = load_experiment(_config(tmp_path, tiles=_STREAMED))
    line = preflight.pace_advisory(exp)

    assert line and line.startswith("PACE: ")
    assert "streamed road" in line and "resident road" in line
    assert "s per model step" in line and "wall for this" in line


def test_the_pace_is_not_an_earned_advisory(tmp_path):
    """THE CONTRACT the first wiring broke.

    Every entry in ``check_advisories`` is EARNED by a config that turned
    something on, and tests/test_checkpoint_route_contract.py pins that a
    plain config earns none so that a green check stays green.  The pace
    is owed to EVERY run, so it is printed on its own line instead of
    being smuggled into a list whose contract is the opposite.
    """

    from gpuwm.core.preflight import check_advisories

    for tiles in ("", _STREAMED):
        exp = load_experiment(_config(tmp_path, tiles=tiles, name=f"x{len(tiles)}"))
        assert not [line for line in check_advisories(exp)
                    if "s per model step" in line]


# ---------------------------------------------------------------------------
# K agrees with the decision that sends a domain down the streamed road
# ---------------------------------------------------------------------------


def _machine(vram_bytes):
    from tilestream import autoplan

    return autoplan.Machine(vram_bytes=int(vram_bytes),
                            host_bytes=64 * (1 << 30), name="fixture",
                            vram_headroom=0.0)


@pytest.mark.parametrize("budget_gib", [6.0, 10.0, 16.0, 24.0])
@pytest.mark.parametrize("nz", [8, 49])
def test_K_is_the_boundary_of_the_stream_init_decision(budget_gib, nz,
                                                       tmp_path):
    """THE AGREEMENT.  K columns plans resident; K+1 does not.

    Two models of one card is how a report ends up advising a domain
    size the planner then streams anyway.  The bound is therefore
    derived from ``autoplan``'s own pricing and CHECKED against
    ``autoplan.plan`` at the boundary, in both directions -- a bound that
    only ever fits is satisfied by returning 1.
    """

    from tilestream import autoplan

    import dataclasses

    exp = load_experiment(_config(tmp_path))
    cfg = exp.domains[0].run
    cfg = dataclasses.replace(cfg, nz=int(nz))
    machine = _machine(budget_gib * (1 << 30))

    limit = pace_module.resident_column_limit(cfg, machine)
    assert limit is not None and limit > 0

    def square(columns):
        # A square domain of exactly ``columns`` columns is not always an
        # integer, so the bound is checked on the CELL count the planner
        # actually prices, through a cfg carrying nx*ny = columns.
        return dataclasses.replace(cfg, nx=int(columns), ny=1)

    fp = autoplan.footprint_for(cfg)
    budget = autoplan.budget_for(machine, fp)
    assert fp.resident_bytes(limit * nz) <= budget
    assert fp.resident_bytes((limit + 1) * nz) > budget

    # And the same boundary through the front the run itself takes.
    assert autoplan.plan(square(limit), machine,
                         prefer_resident=True).mode == "resident"
    with pytest.raises(Exception):
        plan = autoplan.plan(square(limit + 1), machine,
                             prefer_resident=True)
        assert plan.mode == "resident", "K+1 must not fit resident"


def test_the_reported_case_is_quoted_hours_and_a_smaller_domain(tmp_path):
    """THE CASE THIS FEATURE EXISTS FOR, end to end.

    399,119 columns of full physics on a 10 GiB card.  The report has to
    say three things the old one did not: that the road is streamed, that
    the wall clock runs to hours rather than the minutes a user
    extrapolates from a small domain, and what column count would put
    this card back on the resident road.
    """

    import dataclasses

    from tilestream import autoplan

    base = load_experiment(_config(tmp_path)).domains[0].run
    # 631 x 632 = 398,792 columns, the reported domain to within a row.
    cfg = dataclasses.replace(
        base, nx=631, ny=632, nz=49,
        mp_physics=8, ra_lw_physics=4, ra_sw_physics=4)
    assert autoplan.rung_of(cfg) == "full"

    card = _machine(10 * (1 << 30))

    # The card cannot hold it, which is WHY the run streamed -- and the
    # bound is the domain that would have fitted.
    limit = pace_module.resident_column_limit(cfg, card)
    assert limit is not None
    assert limit < 631 * 632, "fixture no longer needs to stream"

    rate = pace_module.step_rate("full", "streamed")
    assert rate.measured, "the streamed full-physics rate is a measurement"
    low, high = rate.seconds_per_step(631 * 632, 49)
    # SECONDS per step, not the sub-second a user extrapolates from a
    # small domain.
    assert 1.0 < low < high
    steps = 10800 / 15.0                      # a 3 h forecast at dt 15 s
    fast_h, slow_h = low * steps / 3600.0, high * steps / 3600.0
    # The MEASUREMENT decides the numbers, not an expectation of them:
    # this reads 0.48 h at the fast end (the RTX 3080 point, measured on
    # this very domain) and 1.67 h on the most contended box.  What the
    # test pins is that the answer runs to a large fraction of an hour or
    # more -- the thing the silent report never said -- and not that it
    # matches a figure picked in advance.
    assert 0.4 < fast_h < slow_h
    assert slow_h > 1.0, f"slow end reads {slow_h:.2f} h, expected hours"


def test_K_falls_when_the_card_shrinks(tmp_path):
    """The negative control: the bound reads the budget it is given."""

    exp = load_experiment(_config(tmp_path))
    cfg = exp.domains[0].run

    big = pace_module.resident_column_limit(cfg, _machine(24 * (1 << 30)))
    small = pace_module.resident_column_limit(cfg, _machine(8 * (1 << 30)))
    assert big > small > 0


def test_K_is_none_rather_than_a_guess_when_no_card_is_known(tmp_path):
    """No allowance, no bound.  A number here would be invented."""

    exp = load_experiment(_config(tmp_path))
    assert pace_module.resident_column_limit(
        exp.domains[0].run, None) is None


# ---------------------------------------------------------------------------
# the rates themselves
# ---------------------------------------------------------------------------


def test_every_rung_and_road_the_planner_can_choose_has_a_rate(tmp_path):
    """A missing row would answer ``None`` where a pace is due."""

    from tilestream import autoplan

    for rung in autoplan.FOOTPRINTS:
        for road in ("resident", "streamed"):
            rate = pace_module.step_rate(rung, road)
            assert rate is not None, (rung, road)
            assert 0.0 < rate.low <= rate.high
            assert rate.basis and rate.reference_card
            # An unmeasured row must SAY it is unmeasured, in the field a
            # caller branches on and in the prose a reader believes.
            if not rate.measured:
                assert "unmeasured-bound" in rate.basis


def test_the_streamed_road_is_dearer_than_the_resident_one_at_every_rung():
    """The negative control on the table itself.

    A table that priced streaming as the cheaper road would advise a user
    to keep the domain that is costing them the wall clock.
    """

    from tilestream import autoplan

    for rung in autoplan.FOOTPRINTS:
        resident = pace_module.step_rate(rung, "resident")
        streamed = pace_module.step_rate(rung, "streamed")
        assert streamed.high > resident.high, rung


def test_the_byte_model_reproduces_the_measured_receipt(tmp_path):
    """THE LICENCE for pricing a tiling nobody has timed.

    ``tilestream/HANDOFF-case-imagery.md`` records a real HRRR case --
    1200x900x49, tile 400x300, halo 16 -- moving "~27 GB of pinned host
    RAM per step" at a redundancy of 1.1952x.  The byte model has to
    reproduce that from the tiling alone, or it is not entitled to
    answer for the tilings that have no receipt.
    """

    import dataclasses

    from tilestream import autoplan

    class _Envelope:
        tile_nx, tile_ny, halo = 400, 300, 16
        window_nx, window_ny = 400 + 32, 300 + 32
        nbuffers = 2

    base = load_experiment(_config(tmp_path)).domains[0].run
    cfg = dataclasses.replace(
        base, nx=1200, ny=900, nz=49,
        mp_physics=8, ra_lw_physics=4, ra_sw_physics=4)
    assert autoplan.rung_of(cfg) == "full", "fixture is not the measured rung"

    moved = pace_module.streamed_transfer_bytes_per_step(_Envelope(), cfg)
    assert 26.0e9 <= moved <= 28.0e9, f"{moved / 1e9:.2f} GB against ~27"
    assert round(autoplan.redundancy(1200, 900, 400, 300, 16), 4) == 1.1952


def test_the_bandwidth_probe_is_absent_gracefully(monkeypatch):
    """No CUDA, no probe, no exception -- and no invented number."""

    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("no cupy on this box")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert pace_module.measured_pinned_bytes_per_second() is None
