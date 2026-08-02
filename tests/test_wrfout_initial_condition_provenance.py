# tests/test_wrfout_initial_condition_provenance.py
"""The wrfout says what its initial condition WAS, not only when it began.

v1.4.0 computed the initial-condition provenance correctly and wrote it to
``run/report.json`` only.  The wrfout -- the durable artifact, the one that
outlives the run directory and the one every downstream consumer reads --
carried nothing: a run initialized from a cycle's 174 h forecast and a run
initialized from that cycle's analysis produced byte-comparable global
attribute sets, differing only in the model clock.  Publishing the picture
lost the provenance silently.

These guards are written against the file, read back with netCDF4, not
against the attribute dict: only the file knows the union the writer
emits.  Every one of them has a control that must fail without the fix.
"""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.io.wrfout import (
    INITIAL_CONDITION_GLOBAL_ATTRS, WRFOUT_INITIAL_CONDITION_SCHEMA,
    WrfoutWriter, initial_condition_global_attrs,
)
from gpuwm.runtime import _global_wrf_attrs

_NX, _NY, _NZ = 4, 3, 5

CYCLE = "2026-08-01T00:00:00Z"


def _provenance(lead_hours):
    """Exactly the block ``gfs_direct.initial_condition_provenance`` writes."""

    analysis = lead_hours == 0
    start = (datetime.datetime(2026, 8, 1)
             + datetime.timedelta(hours=lead_hours))
    return {
        "schema": "gpuwm-gfs-initial-condition-provenance-v1",
        "cycle": CYCLE,
        "initial_forecast_lead_hours": int(lead_hours),
        "model_start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_condition_kind": "analysis" if analysis else "forecast",
        "forecast_generating_process_id": 81 if analysis else 96,
        "statement": (
            f"initialized from GFS cycle {CYCLE} analysis (f000)"
            if analysis else
            f"initialized from GFS cycle {CYCLE} at lead "
            f"f{lead_hours:03d}: the initial condition is itself a "
            f"{lead_hours} h forecast"),
    }


def _grid():
    return SimpleNamespace(
        truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
        ref_lat=39.0, ref_lon=-96.5,
        wrf_map_proj=1, map_proj_char="Lambert Conformal",
        moad_cen_lat=39.0, cen_lat=39.0, cen_lon=-96.5)


def _domain():
    run = RunConfig(nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                    dt=6.0, ztop=10000.0, run_seconds=12.0)
    return SimpleNamespace(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, run=run)


def _write_frame(path, *, start_time, initial_condition=None, source=None):
    """One frame through the production attribute assembly and writer."""

    attrs = _global_wrf_attrs(
        _grid(), start_time, domain=_domain(),
        coord=SimpleNamespace(hybrid_opt=2, etac=0.2),
        initial_condition=initial_condition, source=source)
    frame = {"T2": np.full((_NY, _NX), 290.0, np.float32)}
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs, soil_layers=4) as writer:
        writer.write_frame(
            start_time.strftime("%Y-%m-%d_%H:%M:%S"), frame)
    return path


def _read_globals(path):
    with netCDF4.Dataset(path, "r") as dataset:
        return {name: dataset.getncattr(name) for name in dataset.ncattrs()}


# ---------------------------------------------------------------------------
# The finding itself: two runs that were indistinguishable
# ---------------------------------------------------------------------------

def test_a_forecast_lead_run_is_not_readable_as_an_analysis(tmp_path):
    """C-03: the durable artifact carries the lead, the cycle and the kind."""

    start = datetime.datetime(2026, 8, 8, 6)
    emitted = _read_globals(_write_frame(
        tmp_path / "lead.nc", start_time=start,
        initial_condition=_provenance(174), source="gfs"))

    assert emitted["GPUWM_INITIAL_CONDITION_KIND"] == "forecast"
    assert int(emitted["GPUWM_INITIAL_FORECAST_LEAD_HOURS"]) == 174
    assert emitted["GPUWM_INITIAL_CONDITION_CYCLE"] == "2026-08-01_00:00:00"
    assert emitted["GPUWM_INITIAL_CONDITION_SOURCE"] == "GFS"
    assert int(
        emitted["GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID"]) == 96
    assert (emitted["GPUWM_INITIAL_CONDITION_MODEL_START_DATE"]
            == "2026-08-08_06:00:00")
    assert "174 h forecast" in emitted["GPUWM_INITIAL_CONDITION_STATEMENT"]
    assert (emitted["GPUWM_INITIAL_CONDITION_SCHEMA"]
            == WRFOUT_INITIAL_CONDITION_SCHEMA)
    # WRF's own clock attributes keep WRF's meaning: they are the model's
    # time zero, which at a nonzero lead is NOT the cycle.
    assert emitted["START_DATE"] == "2026-08-08_06:00:00"
    assert emitted["SIMULATION_START_DATE"] == "2026-08-08_06:00:00"


def test_an_analysis_run_says_analysis_rather_than_saying_nothing(tmp_path):
    """Silence is not a label: an analysis run states that it is one."""

    emitted = _read_globals(_write_frame(
        tmp_path / "analysis.nc",
        start_time=datetime.datetime(2026, 8, 1),
        initial_condition=_provenance(0), source="gfs"))

    assert emitted["GPUWM_INITIAL_CONDITION_KIND"] == "analysis"
    assert int(emitted["GPUWM_INITIAL_FORECAST_LEAD_HOURS"]) == 0
    assert int(
        emitted["GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID"]) == 81
    assert emitted["GPUWM_INITIAL_CONDITION_CYCLE"] == "2026-08-01_00:00:00"
    assert "analysis (f000)" in emitted["GPUWM_INITIAL_CONDITION_STATEMENT"]


def test_the_two_runs_are_distinguishable_in_the_file_alone(tmp_path):
    """The reported defect, stated as an assertion.

    Before the fix these two files differed only in their model clock, so a
    reader holding one frame could not tell a genuine analysis chart from a
    174 h-lead chart.  They must now differ in the provenance the file
    itself carries -- and the attribute NAMES must be identical, so a
    consumer keys on values rather than on presence.
    """

    analysis = _read_globals(_write_frame(
        tmp_path / "a.nc", start_time=datetime.datetime(2026, 8, 8, 6),
        initial_condition=_provenance(0), source="gfs"))
    lead = _read_globals(_write_frame(
        tmp_path / "b.nc", start_time=datetime.datetime(2026, 8, 8, 6),
        initial_condition=_provenance(174), source="gfs"))

    assert sorted(analysis) == sorted(lead)
    differing = {name for name in analysis
                 if analysis[name] != lead[name]}
    # The model clock is identical between these two by construction, so
    # every difference is provenance -- and there must be some.
    assert differing, "the two runs are still indistinguishable"
    assert differing <= set(INITIAL_CONDITION_GLOBAL_ATTRS)
    assert "GPUWM_INITIAL_CONDITION_KIND" in differing
    assert "GPUWM_INITIAL_FORECAST_LEAD_HOURS" in differing


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def test_the_writer_refuses_provenance_that_relabels_a_lead(tmp_path):
    """CONTROL: the promise the receipt keeps is enforced at the writer too.

    A block claiming lead f174 is an ``analysis`` (process 81) must not
    reach a file.  Without this the attributes would faithfully transcribe
    a lie instead of refusing it.
    """

    lying = {**_provenance(174),
             "initial_condition_kind": "analysis",
             "forecast_generating_process_id": 81}
    with pytest.raises(ValueError, match="is not an analysis"):
        initial_condition_global_attrs(lying, source="gfs")

    # ... and the same block refused by the runner's own proof reader, so
    # the two readers of one document cannot disagree about it.
    from gpuwm.prepared_single_domain_forecast import (
        proof_initial_forecast_lead)
    with pytest.raises(ValueError):
        proof_initial_forecast_lead({"initial_condition": lying})


@pytest.mark.parametrize("mutation,pattern", [
    ({"initial_forecast_lead_hours": -3}, "unreadable"),
    ({"initial_forecast_lead_hours": "174"}, "unreadable"),
    ({"cycle": "not-a-time"}, "cycle"),
    ({"model_start_time": None}, "model start"),
    ({"initial_condition_kind": "guess"}, "is not an analysis"),
])
def test_a_malformed_provenance_block_is_refused(mutation, pattern):
    """CONTROL: every field the attributes publish is read, not copied."""

    with pytest.raises(ValueError, match=pattern):
        initial_condition_global_attrs(
            {**_provenance(174), **mutation}, source="gfs")


def test_cycle_plus_lead_must_be_the_model_start_in_the_block():
    """CONTROL: the three fields are checked against each other, not stored.

    A file whose cycle, lead and start do not compose is a file whose
    provenance cannot be trusted; it must be refused rather than written.
    """

    with pytest.raises(ValueError, match="does not compose"):
        initial_condition_global_attrs(
            {**_provenance(174), "model_start_time": "2026-08-09T06:00:00Z"},
            source="gfs")


def test_a_route_with_no_source_provenance_writes_no_provenance_attribute(
        tmp_path):
    """Omission means UNKNOWN and must never be readable as ``analysis``.

    Idealized cases and routes whose preparation publishes no
    initial-condition receipt keep exactly the attribute set they had.
    """

    emitted = _read_globals(_write_frame(
        tmp_path / "none.nc", start_time=datetime.datetime(2020, 6, 1, 12)))
    for name in INITIAL_CONDITION_GLOBAL_ATTRS:
        assert name not in emitted


def test_every_published_attribute_is_named_in_the_public_tuple():
    """The tuple is the contract downstream consumers read; keep it exact."""

    attrs = initial_condition_global_attrs(_provenance(12), source="gfs")
    assert sorted(attrs) == sorted(INITIAL_CONDITION_GLOBAL_ATTRS)
    assert all(name.startswith("GPUWM_")
               for name in INITIAL_CONDITION_GLOBAL_ATTRS)


def test_the_integer_attributes_are_netcdf_integers(tmp_path):
    """WRF writes integer globals as NC_INT; so do these."""

    path = _write_frame(
        tmp_path / "types.nc", start_time=datetime.datetime(2026, 8, 8, 6),
        initial_condition=_provenance(174), source="gfs")
    with netCDF4.Dataset(path, "r") as dataset:
        for name in ("GPUWM_INITIAL_FORECAST_LEAD_HOURS",
                     "GPUWM_INITIAL_CONDITION_GENERATING_PROCESS_ID"):
            assert isinstance(dataset.getncattr(name), np.integer), name


# ---------------------------------------------------------------------------
# The downstream consumer that reads a wrfout back
# ---------------------------------------------------------------------------

def test_downscale_reads_the_lineage_off_an_archived_parent(tmp_path):
    """`gpuwm downscale` can now see what its parent was initialized from."""

    from gpuwm.downscale import parent_initial_condition

    parent = _write_frame(
        tmp_path / "wrfout_d01_parent.nc",
        start_time=datetime.datetime(2026, 8, 8, 6),
        initial_condition=_provenance(174), source="gfs")
    lineage = parent_initial_condition(parent)
    assert lineage["GPUWM_INITIAL_CONDITION_KIND"] == "forecast"
    assert lineage["GPUWM_INITIAL_FORECAST_LEAD_HOURS"] == 174
    # JSON-serializable: it is written into the downscale plan.
    json.dumps(lineage)


def test_the_offline_child_inherits_its_parents_lineage(tmp_path):
    """CONTROL for the archive path: the child does not drop the lead.

    Publishing a downscaled chart separates it from the parent run
    directory just as decisively as publishing the parent's did.
    """

    from gpuwm.offline_child_run import _parent_grid_metadata

    parent = _write_frame(
        tmp_path / "wrfout_d01_parent.nc",
        start_time=datetime.datetime(2026, 8, 8, 6),
        initial_condition=_provenance(240), source="gfs")
    _dx, _dy, attrs = _parent_grid_metadata(parent)
    assert attrs["GPUWM_INITIAL_CONDITION_KIND"] == "forecast"
    assert int(attrs["GPUWM_INITIAL_FORECAST_LEAD_HOURS"]) == 240
    # The projection identity it already carried is untouched.
    assert int(attrs["MAP_PROJ"]) == 1

    # A parent that carries no provenance -- a stock-WRF archive, or a
    # file written before this contract -- hands the child nothing, and
    # the child must not invent an analysis from the silence.
    plain = _write_frame(
        tmp_path / "wrfout_d01_plain.nc",
        start_time=datetime.datetime(2026, 8, 8, 6))
    _dx, _dy, plain_attrs = _parent_grid_metadata(plain)
    assert not set(plain_attrs) & set(INITIAL_CONDITION_GLOBAL_ATTRS)


def test_the_golden_attribute_set_carries_the_provenance_shape():
    """The set guard must know about both shapes, or it guards nothing."""

    golden = json.loads(
        (Path(__file__).parent / "data"
         / "wrfout_global_attribute_set_v1.json").read_text())
    base = set(golden["global_attributes"])
    with_provenance = set(golden["global_attributes_with_initial_condition"])
    assert with_provenance - base == set(INITIAL_CONDITION_GLOBAL_ATTRS)
