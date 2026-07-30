"""Is a gpuwm wrfout a *WRF* wrfout, judged by a WRF reader?

Every other wrfout test in this repository reads the file back with
``netCDF4`` and compares it against what this repository believes it wrote.
That is a self-consistency check, and self-consistency is exactly what the
v1.1 scheme extension had: it wrote ``ISNOWXY`` as a float32 and read
``ISNOWXY`` back as a float32, and every test passed while no WRF-name
consumer on earth could find Noah-MP's state at all.

So this file judges the file by two authorities that are not this
repository:

* **wrf-rust** -- the mandated downstream analysis library, its real
  ``WrfFile``/``getvar`` API, opening a file the production writer produced
  from a real forecast.  It answers "can a WRF reader find this field?",
  and its answer for a Noah-MP field before v1.1.3 was ``unknown
  variable``.
* the **pinned WRF v4.6.1 Registry**, transcribed in
  ``gpuwm.io.wrf_output_schema``, plus the group's own v4.6.1 reference
  wrfout where it carries the same variable.  Those answer "is the schema
  the field claims the schema WRF declares?".

The wrf-rust reader promotes everything it returns to float64, which is a
perfectly reasonable thing for an analysis library to do and is also why it
cannot be the witness for on-disk type.  The witness for type is the file:
``netCDF4`` reading the variable's ``dtype`` and ``FieldType`` is not a
mock of the writer, it is the artifact.  So the two halves are split on
purpose -- names and values through the real consumer, declared type and
dimensions through the real file.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from conftest import requires_gpu

from gpuwm.io.wrf_output_schema import (
    PHYSICS_SELECTOR_GLOBALS, PRECIPITATION_OUTPUT_FIELDS,
    SCHEME_OUTPUT_FIELDS, WRF_FIELD_TYPE_INTEGER, WRF_FIELD_TYPE_REAL,
)
from test_wrf_output_schema import WRF_RUST_SHADOWED_NAMES

#: The four integer scheme fields, and the scheme each belongs to.  Both
#: schemes and all four names are exercised: a dtype gate proved at one
#: field name in one scheme proves that field name in that scheme.
_INTEGER_FIELDS = {
    "ktop_plume": "bl_pbl_physics=5",
    "kpbl": "bl_pbl_physics=5",
    "isnowxy": "sf_surface_physics=4",
    "pgsxy": "sf_surface_physics=4",
}

#: Real-valued neighbours of the integer fields, one per scheme.  A gate that
#: only ever sees ``i4`` cannot tell "integer fields are i4" from "every field
#: is i4", so the real half of the dtype dimension is asserted too.
_REAL_FIELDS = {
    "ztop_plume": "bl_pbl_physics=5",
    "tvxy": "sf_surface_physics=4",
}

#: The names the pre-v1.1.3 writer produced for Noah-MP state: the runtime
#: key upper-cased.  WRF writes none of them, so their *absence* is half of
#: the conformance claim -- a file carrying both spellings would let a reader
#: succeed by accident.
_NEVER_WRF_NAMES = ("TVXY", "TGXY", "ISNOWXY", "PGSXY", "ZSNSOXY",
                    "TSNOXY", "SNICEXY", "SNLIQXY")

#: The three MYNN carriers WRF declares ``Z``-staggered, so the file holds
#: one more level than gpuwm computes.
_LIFTED_NAMES = ("EL_PBL", "EXCH_H", "EXCH_M")

_START = datetime(2026, 7, 1, 18, 0, 0)
_LATITUDE = 40.0
_LONGITUDE = -100.0


# ---------------------------------------------------------------------------
# The writer, judged on a real file.
# ---------------------------------------------------------------------------

def _build_mynn_noahmp(*, nx=8, ny=6, nz=20, snow_mm=40.0,
                       snow_depth_m=0.20):
    """A tiny MYNN + Noah-MP column set, built through the real run path.

    Both schemes together on purpose: the integer/name defect spans two
    selectors, and a fixture that ran one of them would leave the other's
    half of the claim untested.  8x6x20 for three steps is seconds on the
    reference card -- Noah-MP's per-land-column cost is the reason
    ``tests/test_noahmp_runtime.py`` keeps its grid small, and the same
    reason applies here.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=6, sf_sfclay_physics=5,
                    sf_surface_physics=4, bl_pbl_physics=5, bldt=0.0)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0110,
                        np.maximum(0.0110 - 5.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(nz, stretch=2.0)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(5.0)
    state.v[...] = cp.float32(1.0)

    landmask = np.ones((ny, nx), np.float64)
    landmask[:, -2:] = 0.0
    land = landmask == 1.0
    tsk = np.full((ny, nx), 303.0)
    tsk[~land] = 294.0
    soil_t = np.stack([tsk - 1.0, tsk - 2.0, tsk - 3.0, tsk - 4.0])
    soil_m = np.full((4, ny, nx), 0.28)
    soil_m[:, ~land] = 1.0
    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(land, 10, 17), isltyp=np.where(land, 6, 14),
        vegfra=60.0, tmn=288.0, swdown=700.0, glw=340.0, pblh=500.0,
        snow=snow_mm, snow_depth=snow_depth_m,
        noahmp_start_time=_START,
        noahmp_latitude=np.full((ny, nx), _LATITUDE, np.float64),
        noahmp_longitude=np.full((ny, nx), _LONGITUDE, np.float64))
    return state, cfg, driver


def _write_real_wrfout(path, state, cfg):
    """Publish one frame with the production writer, WRF globals and all."""
    from gpuwm.io.wrfout import WrfoutWriter, state_frame, wrf_global_attrs

    frame = state_frame(state, include_diagnostic_pressure=True)
    # wrf-rust locates a file's grid and projection through XLAT/XLONG and
    # the projection globals, so a fixture without them is a fixture the
    # oracle cannot open.  They are the same static metadata the runtime's
    # own metadata frame supplies.
    frame["XLAT"] = np.full((cfg.ny, cfg.nx), _LATITUDE, np.float32)
    frame["XLONG"] = np.full((cfg.ny, cfg.nx), _LONGITUDE, np.float32)
    valid = _START + timedelta(seconds=3.0 * cfg.dt)
    # Through the production entry point, not a hand-built dict: that is
    # what carries the resolved physics selectors, so the fixture exercises
    # the same globals path a real run does.
    grid = SimpleNamespace(
        wrf_map_proj=1, map_proj_char="Lambert Conformal",
        truelat1=30.0, truelat2=60.0, stand_lon=_LONGITUDE,
        ref_lat=_LATITUDE, ref_lon=_LONGITUDE, moad_cen_lat=_LATITUDE,
        cen_lat=_LATITUDE, cen_lon=_LONGITUDE)
    global_attrs = wrf_global_attrs(grid, _START, dt=cfg.dt, run=cfg,
                                    grid_id=1, parent_id=0,
                                    i_parent_start=1, j_parent_start=1,
                                    parent_grid_ratio=1)
    global_attrs["GRIDTYPE"] = "C"
    with WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz, dx=cfg.dx,
                      dy=cfg.dy, soil_layers=4,
                      global_attrs=global_attrs) as writer:
        writer.write_frame(valid.strftime("%Y-%m-%d_%H:%M:%S"), frame)
    return frame


@pytest.fixture(scope="module")
def real_wrfout(tmp_path_factory):
    """One real forecast, one real wrfout, shared by the readers below."""
    pytest.importorskip("cupy")
    from gpuwm.core.dycore import step

    state, cfg, _driver = _build_mynn_noahmp()
    for _ in range(3):
        step(state, cfg)
    path = (tmp_path_factory.mktemp("conformance")
            / "wrfout_d01_2026-07-01_18_00_36")
    frame = _write_real_wrfout(path, state, cfg)
    return path, frame


@requires_gpu
def test_wrf_rust_finds_every_scheme_field_under_its_wrf_name(real_wrfout):
    """The mandated downstream reader, on the real file.

    ``getvar`` here is wrf-rust's raw-variable path: the name is not one of
    its computed diagnostics, so it resolves straight to the file.  Before
    v1.1.3 every Noah-MP name in this list raised ``unknown variable``,
    because the file spelled them ``TVXY``/``ISNOWXY``/``ZSNSOXY`` -- names
    WRF has never written.
    """
    wrf = pytest.importorskip("wrf")
    if not hasattr(wrf, "WrfFile"):        # NCAR wrf-python, not wrf-rust
        pytest.skip("the installed 'wrf' module is not wrf-rust")
    path, frame = real_wrfout

    reader = wrf.WrfFile(str(path))
    assert reader.times() == ["2026-07-01_18:00:36"]

    checked = 0
    for key, schema in SCHEME_OUTPUT_FIELDS.items():
        if schema.netcdf_name not in frame:
            continue                        # scheme not active in this run
        if schema.netcdf_name in WRF_RUST_SHADOWED_NAMES:
            # wrf-rust's own computed registry claims these two names, so
            # getvar returns its diagnostic rather than the raw variable.
            # See tests/test_wrf_output_schema.py for why that is read
            # through the file instead of being renamed away.
            continue
        values = np.asarray(reader.getvar(schema.netcdf_name))
        expected = np.asarray(frame[schema.netcdf_name])
        if schema.netcdf_name in _LIFTED_NAMES:
            # WRF's own axis for these is one level taller than the mass
            # column; the reader must see the computed levels and the zero
            # interface WRF's PBL driver never writes.
            assert values.shape == (expected.shape[0] + 1,
                                    *expected.shape[1:]), schema.netcdf_name
            np.testing.assert_array_equal(
                values[-1], np.zeros_like(values[-1]))
            values = values[:-1]
        assert values.shape == expected.shape, schema.netcdf_name
        np.testing.assert_array_equal(
            values, expected.astype(np.float64),
            err_msg=f"{schema.netcdf_name} ({key}, {schema.registry})")
        checked += 1
    # Both schemes are live in this fixture, so the count is the union of
    # their inventories, not one of them.
    assert checked >= len(_INTEGER_FIELDS) + len(_REAL_FIELDS)
    assert checked >= 60, checked


@requires_gpu
def test_wrf_rust_cannot_find_the_names_wrf_never_writes(real_wrfout):
    """The other half: the wrong spellings are gone, not merely aliased."""
    wrf = pytest.importorskip("wrf")
    if not hasattr(wrf, "WrfFile"):
        pytest.skip("the installed 'wrf' module is not wrf-rust")
    path, _frame = real_wrfout
    reader = wrf.WrfFile(str(path))
    for name in _NEVER_WRF_NAMES:
        with pytest.raises(Exception) as excinfo:
            reader.getvar(name)
        assert name in str(excinfo.value)


@requires_gpu
def test_integer_scheme_fields_are_integers_on_disk(real_wrfout):
    """Two schemes, four integer names, and the real half of the dtype axis.

    wrf-rust returns float64 for everything, so it is the file that has to
    answer this one.  ``netCDF4`` reading the published artifact is not a
    mock of the writer; it is the artifact.
    """
    path, frame = real_wrfout
    with netCDF4.Dataset(path) as ds:
        schemes_seen = set()
        for key, scheme in _INTEGER_FIELDS.items():
            name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
            var = ds.variables[name]
            assert var.dtype == np.int32, (name, var.dtype)
            assert int(var.FieldType) == WRF_FIELD_TYPE_INTEGER, name
            stored = np.asarray(var[0])
            source = np.asarray(frame[name])
            assert source.dtype == np.int32, name
            np.testing.assert_array_equal(stored, source, err_msg=name)
            schemes_seen.add(scheme)
        assert len(schemes_seen) == 2, schemes_seen

        for key, scheme in _REAL_FIELDS.items():
            name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
            var = ds.variables[name]
            assert var.dtype == np.float32, (name, var.dtype)
            assert int(var.FieldType) == WRF_FIELD_TYPE_REAL, name


@requires_gpu
def test_scheme_variables_carry_their_registry_metadata(real_wrfout):
    """Description, units and stagger, on every scheme variable written."""
    path, frame = real_wrfout
    with netCDF4.Dataset(path) as ds:
        checked = 0
        for schema in SCHEME_OUTPUT_FIELDS.values():
            if schema.netcdf_name not in frame:
                continue
            var = ds.variables[schema.netcdf_name]
            assert var.description == schema.description, schema.netcdf_name
            assert var.units == schema.units, schema.netcdf_name
            assert var.stagger == schema.stagger, schema.netcdf_name
            checked += 1
        assert checked >= 60, checked


@requires_gpu
def test_staggered_scheme_fields_land_on_wrf_axes(real_wrfout):
    """Z-staggered rows get WRF's staggered axis, unstaggered ones do not.

    Both values of the staggering dimension are asserted, on both schemes:
    MYNN's ``EL_PBL`` against its unstaggered ``QKE``, Noah-MP's ``TSNO``
    and ``ZSNSO`` against its unstaggered ``TV``.
    """
    path, _frame = real_wrfout
    with netCDF4.Dataset(path) as ds:
        assert ds.variables["EL_PBL"].dimensions == (
            "Time", "bottom_top_stag", "south_north", "west_east")
        assert ds.variables["EXCH_H"].dimensions == (
            "Time", "bottom_top_stag", "south_north", "west_east")
        assert ds.variables["QKE"].dimensions == (
            "Time", "bottom_top", "south_north", "west_east")
        assert ds.variables["TSNO"].dimensions == (
            "Time", "snow_layers_stag", "south_north", "west_east")
        assert ds.variables["ZSNSO"].dimensions == (
            "Time", "snso_layers_stag", "south_north", "west_east")
        assert ds.variables["TV"].dimensions == (
            "Time", "south_north", "west_east")
        # The lifted mass-level carriers keep their computed levels and pad
        # the interface WRF's own PBL driver never writes.
        el = np.asarray(ds.variables["EL_PBL"][0])
        assert el.shape[0] == len(ds.dimensions["bottom_top"]) + 1
        np.testing.assert_array_equal(el[-1], np.zeros_like(el[-1]))


# ---------------------------------------------------------------------------
# The always-written precipitation accumulators, at both cumulus settings.
# ---------------------------------------------------------------------------

def _build_precipitation(cu_physics):
    """The same column set at one cumulus setting, Noah + YSU underneath.

    Deliberately *not* the MYNN/Noah-MP fixture above: this claim is about
    the core accumulation family, which has nothing to do with which PBL or
    land-surface scheme is selected, and running the cheap pair keeps a
    two-value sweep cheap enough to be a standing gate.
    """
    import cupy as cp

    from gpuwm.config import RunConfig
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    nx, ny, nz = 8, 6, 20
    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=6, sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1, bldt=0.0,
                    cu_physics=cu_physics, cudt_minutes=0.0)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0, 300.0 + 0.004 * (z - 1500.0))

    def qvapor(z):
        return np.maximum(0.012 - 6.0e-7 * np.asarray(z, np.float64), 1.0e-5)

    coord = make_vertical_coord(nz, stretch=2.0)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(5.0)
    driver = initialize_physics(state, cfg, landmask=1.0, tsk=303.0)
    return state, cfg, driver


@pytest.fixture(scope="module", params=[0, 1],
                ids=["cu_physics=0", "cu_physics=1_KF"])
def precipitation_wrfout(request, tmp_path_factory):
    """One real forecast per cumulus setting, published by the real writer."""
    pytest.importorskip("cupy")
    from gpuwm.core.dycore import step

    state, cfg, driver = _build_precipitation(request.param)
    for _ in range(2):
        step(state, cfg)
    path = (tmp_path_factory.mktemp(f"precip{request.param}")
            / "wrfout_d01_2026-07-01_18_00_24")
    frame = _write_real_wrfout(path, state, cfg)
    return path, frame, driver, request.param


@requires_gpu
def test_the_accumulation_family_is_written_at_both_cumulus_settings(
        precipitation_wrfout):
    """All six, present and Registry-conformant, whatever ran.

    WRF allocates and writes these six in every run -- they are core
    ``misc`` history rows with no package gate -- so a reader computing
    ``RAINC + RAINNC`` is entitled to find them. gpuwm used to emit each
    one only when its producer existed, so a ``cu_physics=0`` wrfout had no
    ``RAINC`` at all and every downstream QPF product failed on it.
    """
    path, frame, _driver, _cu = precipitation_wrfout
    with netCDF4.Dataset(path) as ds:
        for name, schema in PRECIPITATION_OUTPUT_FIELDS.items():
            assert name in ds.variables, name
            variable = ds.variables[name]
            assert variable.dtype == np.float32, name
            assert int(variable.FieldType) == WRF_FIELD_TYPE_REAL, name
            assert variable.description == schema.description, name
            assert variable.units == schema.units, name
            assert variable.stagger == "", name
            assert variable.dimensions == (
                "Time", "south_north", "west_east"), name
            np.testing.assert_array_equal(
                np.asarray(variable[0]), np.asarray(frame[name]),
                err_msg=name)


@requires_gpu
def test_rainc_is_zero_without_cumulus_and_is_the_accumulation_with_it(
        precipitation_wrfout):
    """The two values of the cumulus dimension, and what separates them.

    Zeros under ``cu_physics=0`` are the WRF answer, not a placeholder.
    Under KF the field must be the scheme's live accumulator and must
    actually be nonzero somewhere -- otherwise "we write zeros when no
    cumulus runs" and "we write zeros" would be the same test.
    """
    import cupy as cp

    path, _frame, driver, cu_physics = precipitation_wrfout
    with netCDF4.Dataset(path) as ds:
        rainc = np.asarray(ds.variables["RAINC"][0])
    if cu_physics == 0:
        assert driver.rainc is None
        np.testing.assert_array_equal(rainc, np.zeros_like(rainc))
    else:
        np.testing.assert_array_equal(rainc, cp.asnumpy(driver.rainc))
        assert rainc.max() > 0.0, (
            "the KF fixture accumulated no convective precipitation, so "
            "this run cannot distinguish a carried accumulation from the "
            "zero fill; re-tune the sounding rather than weakening this")


@requires_gpu
def test_rainsh_is_zero_because_gpuwm_runs_no_shallow_cumulus(
        precipitation_wrfout):
    """A true zero, not a placeholder -- and it is stated as such.

    gpuwm implements no shallow-cumulus scheme, and zero is exactly what
    stock WRF writes for ``shcu_physics=0``; the group's own v4.6.1
    reference wrfout carries ``RAINSH`` at max 0 for that reason.
    """
    path, _frame, _driver, _cu = precipitation_wrfout
    with netCDF4.Dataset(path) as ds:
        rainsh = np.asarray(ds.variables["RAINSH"][0])
    np.testing.assert_array_equal(rainsh, np.zeros_like(rainsh))


@requires_gpu
def test_wrf_rust_reads_the_accumulation_family_at_both_cumulus_settings(
        precipitation_wrfout):
    """The mandated downstream reader, on the field that broke it.

    ``getvar("RAINC")`` on a ``cu_physics=0`` file returned
    ``unknown variable: RAINC`` before this change, which is what took out
    21 of WRF-Runner's plots. It must now return a correctly shaped field.
    """
    wrf = pytest.importorskip("wrf")
    if not hasattr(wrf, "WrfFile"):        # NCAR wrf-python, not wrf-rust
        pytest.skip("the installed 'wrf' module is not wrf-rust")
    path, frame, _driver, cu_physics = precipitation_wrfout

    reader = wrf.WrfFile(str(path))
    for name in PRECIPITATION_OUTPUT_FIELDS:
        values = np.asarray(reader.getvar(name))
        expected = np.asarray(frame[name])
        assert values.shape == expected.shape, name
        np.testing.assert_array_equal(
            values, expected.astype(np.float64), err_msg=name)
    # The QPF recipe itself, on the real reader: RAINC + RAINNC must be a
    # finite field of the grid's shape at either cumulus setting.
    total = (np.asarray(reader.getvar("RAINC"))
             + np.asarray(reader.getvar("RAINNC")))
    assert total.shape == np.asarray(frame["RAINC"]).shape
    assert np.isfinite(total).all()
    if cu_physics == 0:
        np.testing.assert_array_equal(total, np.zeros_like(total))


@requires_gpu
def test_a_real_forecast_stamps_its_own_physics_selectors(
        precipitation_wrfout):
    """The selectors on a real forecast, at both cumulus settings.

    This is the integration half: the fixture goes through the production
    ``wrf_global_attrs`` path, so what lands in the file is what a real run
    would stamp. ``SHCU_PHYSICS=0`` beside an all-zero ``RAINSH`` is the
    whole point -- together they say "no shallow-cumulus scheme exists",
    which neither says alone.
    """
    path, _frame, _driver, cu_physics = precipitation_wrfout
    with netCDF4.Dataset(path) as ds:
        for selector in PHYSICS_SELECTOR_GLOBALS:
            assert selector.name in ds.ncattrs(), selector
            assert np.asarray(
                getattr(ds, selector.name)).dtype == np.int32, selector
        assert ds.CU_PHYSICS == cu_physics
        assert ds.MP_PHYSICS == 6
        assert ds.SF_SURFACE_PHYSICS == 2
        assert ds.BL_PBL_PHYSICS == 1
        assert ds.SHCU_PHYSICS == 0
        # and the zeros the selectors explain
        np.testing.assert_array_equal(
            np.asarray(ds.variables["RAINSH"][0]),
            np.zeros((ds.dimensions["south_north"].size,
                      ds.dimensions["west_east"].size), np.float32))


@requires_gpu
def test_wrf_rust_still_opens_a_file_carrying_the_selector_globals(
        precipitation_wrfout):
    """Eleven added globals must not perturb the mandated reader."""
    wrf = pytest.importorskip("wrf")
    if not hasattr(wrf, "WrfFile"):
        pytest.skip("the installed 'wrf' module is not wrf-rust")
    path, frame, _driver, _cu = precipitation_wrfout
    reader = wrf.WrfFile(str(path))
    assert reader.dx == 3000.0 and reader.dy == 3000.0
    assert np.asarray(reader.getvar("RAINC")).shape == np.asarray(
        frame["RAINC"]).shape


@requires_gpu
def test_projection_globals_are_single_precision(real_wrfout):
    """Stock WRF writes NC_FLOAT for these; a Python float would be double."""
    path, _frame = real_wrfout
    with netCDF4.Dataset(path) as ds:
        for name in ("DX", "DY", "TRUELAT1", "TRUELAT2", "STAND_LON",
                     "CEN_LAT", "CEN_LON", "MOAD_CEN_LAT", "POLE_LAT",
                     "POLE_LON", "DT"):
            assert np.asarray(getattr(ds, name)).dtype == np.float32, name
