# tests/test_wrfout_global_attribute_set.py
"""Exact global-attribute SET of a frame written through the production path.

Nothing in the suite watches the wrfout global-attribute set as a set.  Field
inventories and individual attribute values are guarded elsewhere; an
attribute ADDED to every emitted file passes all of it.  That is the change a
capsule-versus-sidecar decision would make, so the guard has to exist before
the decision is taken rather than after, or the decision is taken blind.

The golden set is whatever ``gpuwm.runtime._global_wrf_attrs`` +
``WrfoutWriter`` emit at this commit, read back off a real NETCDF4_CLASSIC
file with netCDF4 rather than off the attrs dict: the writer adds its own
header attributes on top of the caller's, and only the file knows the union.
The mutation control adds one synthetic global and requires the guard to fail.
"""

import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np

from gpuwm.config import RunConfig
from gpuwm.io.wrfout import INITIAL_CONDITION_GLOBAL_ATTRS, WrfoutWriter
from gpuwm.runtime import _global_wrf_attrs

_GOLDEN_PATH = (Path(__file__).parent / "data"
                / "wrfout_global_attribute_set_v1.json")

_NX, _NY, _NZ = 4, 3, 5


def _grid():
    """Projection identity a real-case domain carries into the writer."""

    return SimpleNamespace(
        truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
        ref_lat=39.0, ref_lon=-96.5,
        wrf_map_proj=1, map_proj_char="Lambert Conformal",
        moad_cen_lat=39.0, cen_lat=39.0, cen_lon=-96.5)


def _domain():
    # A real RunConfig, not a namespace mock: wrf_global_attrs stamps the
    # physics-selector globals off the resolved run configuration, so a
    # hand-listed mock would silently fall behind every selector the
    # production reader grows.  Defaults are the resolved truth here.
    run = RunConfig(nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                    dt=6.0, ztop=10000.0, run_seconds=12.0)
    return SimpleNamespace(
        grid_id=2, parent_id=1, i_parent_start=5, j_parent_start=5,
        parent_grid_ratio=3, run=run)


def _coord():
    return SimpleNamespace(hybrid_opt=2, etac=0.2)


#: The initial-condition provenance a front-door-prepared run carries.
#: Held here, in the golden's own production path, because the emitted
#: attribute SET now has two legitimate shapes -- with and without a
#: source receipt -- and a guard that watched only one of them would let
#: the other drift unobserved.
PROVENANCE = {
    "schema": "gpuwm-gfs-initial-condition-provenance-v1",
    "cycle": "2020-06-01T00:00:00Z",
    "initial_forecast_lead_hours": 12,
    "model_start_time": "2020-06-01T12:00:00Z",
    "initial_condition_kind": "forecast",
    "forecast_generating_process_id": 96,
    "statement": "initialized from cycle 2020-06-01T00:00:00Z at lead f012",
}


def _write_production_frame(path, *, extra_attrs=None,
                            initial_condition=None, source=None):
    """One frame through the production attribute assembly and writer."""

    attrs = _global_wrf_attrs(
        _grid(), datetime.datetime(2020, 6, 1, 12),
        domain=_domain(), coord=_coord(),
        initial_condition=initial_condition, source=source)
    if extra_attrs is not None:
        attrs = {**attrs, **extra_attrs}
    frame = {
        "T2": np.full((_NY, _NX), 290.0, np.float32),
        "PSFC": np.full((_NY, _NX), 98000.0, np.float32),
    }
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs, soil_layers=4) as writer:
        writer.write_frame("2020-06-01_12:00:00", frame)
    return path


def _emitted_global_attribute_set(path):
    with netCDF4.Dataset(path, "r") as dataset:
        return sorted(dataset.ncattrs())


def test_emitted_global_attribute_set_is_exactly_the_golden_set(tmp_path):
    """The SET, not a subset: an added or dropped global is a failure."""

    path = _write_production_frame(tmp_path / "wrfout_d02_frame.nc")
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())["global_attributes"]
    assert emitted == sorted(golden), (
        "the wrfout global-attribute set moved; added="
        f"{sorted(set(emitted) - set(golden))} removed="
        f"{sorted(set(golden) - set(emitted))}")


def test_the_provenance_carrying_set_is_exactly_its_golden_set(tmp_path):
    """The other legitimate shape: a run whose preparation named a source.

    Every front-door route writes this one.  Guarding only the shape
    without provenance would have let the initial-condition contract be
    added, dropped or renamed with the suite still green.
    """

    path = _write_production_frame(
        tmp_path / "wrfout_d02_provenance.nc",
        initial_condition=PROVENANCE, source="gfs")
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())[
        "global_attributes_with_initial_condition"]
    assert emitted == sorted(golden), (
        "the provenance-carrying wrfout global-attribute set moved; added="
        f"{sorted(set(emitted) - set(golden))} removed="
        f"{sorted(set(golden) - set(emitted))}")


def test_the_two_golden_shapes_differ_by_exactly_the_provenance(tmp_path):
    """MUTATION CONTROL across the shapes: neither may absorb the other."""

    golden = json.loads(_GOLDEN_PATH.read_text())
    base = set(golden["global_attributes"])
    with_provenance = set(golden["global_attributes_with_initial_condition"])
    assert with_provenance - base == set(INITIAL_CONDITION_GLOBAL_ATTRS)
    assert not base - with_provenance, (
        "a provenance-carrying file dropped an attribute the plain one has")


def test_one_synthetic_global_attribute_fails_the_guard(tmp_path):
    """MUTATION CONTROL: the guard can see exactly the change it exists for.

    A single extra global -- the shape an embedded evidence-tier attribute
    would take -- must make the set assertion above fail.  Without this the
    guard could be a subset check that never fires.
    """

    path = _write_production_frame(
        tmp_path / "wrfout_d02_mutated.nc",
        extra_attrs={"GPUWM_SYNTHETIC_CONTROL_ATTRIBUTE": "mutation-control"})
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())["global_attributes"]
    assert "GPUWM_SYNTHETIC_CONTROL_ATTRIBUTE" in emitted
    assert emitted != sorted(golden), (
        "the guard did not see a synthetic global attribute; it is not an "
        "exact-set assertion")


def test_dropping_one_global_attribute_fails_the_guard(tmp_path):
    """MUTATION CONTROL, other direction: a removed global must fail too."""

    path = _write_production_frame(tmp_path / "wrfout_d02_frame.nc")
    emitted = _emitted_global_attribute_set(path)
    golden = json.loads(_GOLDEN_PATH.read_text())["global_attributes"]
    shortened = sorted(set(golden) - {sorted(golden)[0]})
    assert emitted != shortened


def test_golden_set_carries_no_case_token():
    """Certification-path data stays generic; the case lives in configs."""

    text = _GOLDEN_PATH.read_text().lower()
    for token in ("real74", "1974", "ohio", "hrrr"):
        assert token not in text, (
            f"case token {token!r} leaked into the wrfout attribute golden set")


def test_a_delayed_start_nest_keeps_the_simulations_start_date(tmp_path):
    """audit io-01-02: START_DATE is per domain, SIMULATION_START_DATE is not.

    WRF writes them from two different places.  ``share/output_wrf.F:343-348``
    reads ``start_*`` at ``grid%id`` for ``START_DATE``, while ``:361-376``
    reads ``simulation_start_*`` at namelist index **1** for
    ``SIMULATION_START_DATE`` on every history file, and
    ``share/set_timekeeping.F:379-388`` sets those index-1 values once,
    ``IF ( grid%id .EQ. head_grid%id )`` -- so WRF's SIMULATION_START_DATE is
    the head grid's start, identical on every domain.

    ``PerDomainWrfoutWriters`` passes each domain's own ``cfg.start_time``
    when it declares one, and that value went to BOTH attributes.  Because
    ``rw_wrfbatch`` reads SIMULATION_START_DATE first as the run origin and
    measures every product's lead from it (``surface_wrfout.ORIGIN_ATTRS``),
    a d02 frame valid at 09Z was labelled F+03 while the d01 frame of the
    same instant was labelled F+09: two panels of one run, one valid time,
    two lead labels.
    """
    simulation_start = datetime.datetime(2026, 7, 18, 0)
    nest_start = datetime.datetime(2026, 7, 18, 6)

    nest = _global_wrf_attrs(
        _grid(), nest_start, domain=_domain(), coord=_coord(),
        simulation_start_time=simulation_start)
    assert nest["START_DATE"] == "2026-07-18_06:00:00"
    assert nest["SIMULATION_START_DATE"] == "2026-07-18_00:00:00"

    # The mother domain, whose own start IS the simulation's, is unchanged.
    mother = _global_wrf_attrs(
        _grid(), simulation_start, domain=_domain(), coord=_coord(),
        simulation_start_time=simulation_start)
    assert mother["START_DATE"] == "2026-07-18_00:00:00"
    assert mother["SIMULATION_START_DATE"] == "2026-07-18_00:00:00"

    # Omitting it means "this file starts the simulation", which is what
    # every single-domain and idealized caller is, and it must reproduce
    # the previous single-date behaviour attribute for attribute.
    legacy = _global_wrf_attrs(
        _grid(), nest_start, domain=_domain(), coord=_coord())
    assert legacy["SIMULATION_START_DATE"] == legacy["START_DATE"]
    assert legacy == _global_wrf_attrs(
        _grid(), nest_start, domain=_domain(), coord=_coord(),
        simulation_start_time=nest_start)


def test_every_per_domain_writer_hands_over_the_simulations_start():
    """The plumbing half, asserted at the call sites that own it.

    The unit check above passes even if ``PerDomainWrfoutWriters`` still
    sends the domain's own start to both attributes, because the default
    keeps the two equal.  Three call sites construct a domain's global
    attributes from a per-domain ``domain_start_time`` -- ``__init__``,
    ``add_domain`` and ``refresh_domain`` -- and a repair that reaches two
    of them leaves the third writing the old, wrong origin on exactly the
    domains a relocation or a mid-run spawn produced.  Same shape as
    ``test_every_wrfout_caller_with_a_config_resolves_the_soil_axis``.
    """
    import ast

    source = (Path(__file__).resolve().parents[1] / "gpuwm" / "io"
              / "wrfout.py").read_text(encoding="utf-8")
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", "") == "_global_wrf_attrs"]
    assert len(calls) == 3, [node.lineno for node in calls]
    for node in calls:
        keywords = {keyword.arg for keyword in node.keywords}
        assert "simulation_start_time" in keywords, (
            f"gpuwm/io/wrfout.py:{node.lineno} builds a domain's global "
            "attributes without simulation_start_time, so a delayed-start "
            "nest would publish its own start as the run origin")
        # And it must be the RUN's start, not the domain's -- passing
        # `domain_start_time` here would satisfy the keyword and change
        # nothing.
        value = next(keyword.value for keyword in node.keywords
                     if keyword.arg == "simulation_start_time")
        spelling = ast.unparse(value)
        assert spelling in ("start_time", "self.start_time"), spelling
        assert node.args[1].id == "domain_start_time", ast.unparse(node.args[1])
