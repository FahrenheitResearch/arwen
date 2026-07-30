"""The transcribed WRF v4.6.1 output schema, and the writer that obeys it.

``gpuwm.io.wrf_output_schema`` is a transcription, so the tests it needs are
the tests a transcription needs: is it complete against the inventories the
schemes actually publish, is it internally consistent, and -- where an
independent copy of the same truth exists -- does it agree with that copy?

Two independent copies are available and both are used:

* the group's own WRF v4.6.1 reference wrfout, for every variable it and
  gpuwm both carry (``GPUWM_TEST_WRF74_BUNDLE``);
* the file the production writer actually produces, read back by the
  mandated downstream reader wrf-rust and by ``netCDF4``.

This module holds the half that needs no GPU: the schema itself, and a
``WrfoutWriter`` round trip over hand-built arrays.  The half that runs a
real MYNN + Noah-MP forecast through the production writer and hands the
result to wrf-rust is ``tests/test_wrfout_conformance.py``, which needs a
device and is marked accordingly.  Neither half subsumes the other -- this
one proves the writer honours a declared integer schema on any box, that
one proves the schema the forecast path actually declares is this one.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from gpuwm.io.wrf_output_schema import (
    MYNN_PBL_OUTPUT_FIELDS, NOAHMP_OUTPUT_FIELDS,
    OUTPUT_FIELDS_BY_NETCDF_NAME, PHYSICS_SELECTOR_GLOBALS,
    PRECIPITATION_OUTPUT_FIELDS, RUC_OUTPUT_FIELDS, SCHEME_OUTPUT_FIELDS,
    WRF_FIELD_TYPE_INTEGER, WRF_FIELD_TYPE_REAL,
)
from gpuwm.io.wrfout import WrfoutWriter

#: The four integer scheme fields, and the selector each belongs to.  Both
#: schemes and all four names are listed: a dtype gate proved at one field
#: name in one scheme proves that field name in that scheme.
INTEGER_FIELDS = {
    "ktop_plume": "bl_pbl_physics=5",
    "kpbl": "bl_pbl_physics=5",
    "isnowxy": "sf_surface_physics=4",
    "pgsxy": "sf_surface_physics=4",
}

#: Real-valued neighbours, one per scheme.  A gate that only ever sees ``i4``
#: cannot tell "integer fields are i4" from "every field is i4".
REAL_FIELDS = {
    "ztop_plume": "bl_pbl_physics=5",
    "tvxy": "sf_surface_physics=4",
}

#: Two external WRF names that wrf-rust's *computed* registry also uses:
#: ``tv`` is its virtual temperature and ``wa`` its own diagnostic, so
#: ``getvar("TV")`` returns the diagnostic rather than Noah-MP's leaf
#: temperature.  That shadowing is a property of any Noah-MP wrfout, stock
#: WRF's included -- gpuwm now writes exactly the name WRF writes -- so the
#: right response here is to read those two through the file and say why,
#: not to spell them differently.  It is reported upstream rather than
#: worked around: wrf-rust interop is collaborative.
WRF_RUST_SHADOWED_NAMES = ("TV", "WA")

_REFERENCE_WRFOUT = Path(
    os.environ.get("GPUWM_TEST_WRF74_BUNDLE",
                   "gpuwm-fixture-unset/wrf74-bundle")
) / "wrfout_reference" / "wrfout_d01_1974-04-03_13_00_00"


def _runtime_inventories():
    from gpuwm.core.mynn_pbl_runtime import (MYNN_PBL_DIAGNOSTICS_2D,
                                             MYNN_PBL_DIAGNOSTICS_INT_2D,
                                             MYNN_PBL_STATE_3D)
    from gpuwm.core.noahmp_runtime import (NOAHMP_DIAGNOSTICS_2D,
                                           NOAHMP_STATE_2D,
                                           NOAHMP_STATE_INT_2D,
                                           NOAHMP_STATE_SNOWSOIL_3D,
                                           NOAHMP_STATE_SNOW_3D)
    from gpuwm.core.ruc_runtime import (RUC_DIAGNOSTICS_2D, RUC_STATE_2D,
                                        RUC_STATE_3D)
    return {
        "MYNN": (MYNN_PBL_OUTPUT_FIELDS,
                 (*MYNN_PBL_STATE_3D, *MYNN_PBL_DIAGNOSTICS_2D,
                  *MYNN_PBL_DIAGNOSTICS_INT_2D,
                  # shared EM_COMMON rows the MYNN block also publishes
                  "exch_h", "exch_m", "rmol", "kpbl")),
        "Noah-MP": (NOAHMP_OUTPUT_FIELDS,
                    (*NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                     *NOAHMP_STATE_SNOW_3D, *NOAHMP_STATE_SNOWSOIL_3D,
                     *NOAHMP_DIAGNOSTICS_2D)),
        "RUC": (RUC_OUTPUT_FIELDS,
                (*RUC_STATE_2D, *RUC_STATE_3D, *RUC_DIAGNOSTICS_2D)),
    }


# ---------------------------------------------------------------------------
# the schema
# ---------------------------------------------------------------------------

def test_every_emitted_scheme_field_has_a_schema_row():
    """Three schemes, and every field each of them publishes.

    This is the gate that makes the writer's ``KeyError`` unreachable in a
    shipped build rather than merely loud: a field added to a runtime
    inventory without a Registry row fails here, at collection speed, and
    not on the first Noah-MP forecast someone runs.
    """
    for scheme, (schema, inventory) in _runtime_inventories().items():
        missing = [name for name in inventory if name not in schema]
        assert missing == [], (scheme, missing)
        extra = sorted(set(schema) - set(inventory))
        assert extra == [], (scheme, extra)


def test_schema_types_and_field_types_agree():
    """``FieldType`` follows the declared type, on both types."""
    integer = {name for name, field in SCHEME_OUTPUT_FIELDS.items()
               if field.dtype == "i4"}
    assert integer == set(INTEGER_FIELDS), sorted(integer)
    for name, field in SCHEME_OUTPUT_FIELDS.items():
        expected = (WRF_FIELD_TYPE_INTEGER if field.dtype == "i4"
                    else WRF_FIELD_TYPE_REAL)
        assert field.field_type == expected, name
        assert field.dtype in ("i4", "f4"), name
        assert field.stagger in ("", "X", "Y", "Z"), name


def test_schema_rows_cite_the_pinned_registry_or_declare_themselves_gpuwm():
    """No row is metadata-shaped but provenance-free.

    A row with neither a Registry citation nor a self-declaration as a gpuwm
    addition is a row somebody invented, which is the failure mode the whole
    module exists to end.
    """
    cited = 0
    for name, field in SCHEME_OUTPUT_FIELDS.items():
        if field.registry:
            assert (field.registry.startswith("Registry.EM_COMMON:")
                    or field.registry.startswith("registry.noahmp:")), name
            assert field.description, name
            cited += 1
        else:
            assert "no WRF Registry counterpart" in field.description, name
            assert field.netcdf_name.startswith("RUC_"), name
    assert cited == len(SCHEME_OUTPUT_FIELDS) - 4


def test_no_two_gpuwm_fields_claim_one_netcdf_name():
    assert len(OUTPUT_FIELDS_BY_NETCDF_NAME) == (
        len(SCHEME_OUTPUT_FIELDS) + len(PRECIPITATION_OUTPUT_FIELDS))


def test_noahmp_external_names_are_not_the_runtime_keys_upper_cased():
    """The defect this schema exists to fix, stated as a property.

    Before v1.1.3 the writer produced ``field_name.upper()``.  For Noah-MP
    that is wrong for all but two of its rows, and this asserts the
    disagreement rather than trusting the transcription to be different --
    a schema that had quietly re-derived the old names would pass every
    other test in this file.
    """
    renamed = [key for key, field in NOAHMP_OUTPUT_FIELDS.items()
               if field.netcdf_name != key.upper()]
    unchanged = [key for key, field in NOAHMP_OUTPUT_FIELDS.items()
                 if field.netcdf_name == key.upper()]
    assert len(renamed) >= 40, renamed
    # QSNOWXY and QRAINXY really do keep the symbol spelling: it is their
    # Registry dname.  So does every field whose dname is its symbol.
    assert "qsnowxy" in unchanged and "qrainxy" in unchanged
    assert NOAHMP_OUTPUT_FIELDS["isnowxy"].netcdf_name == "ISNOW"
    assert NOAHMP_OUTPUT_FIELDS["tvxy"].netcdf_name == "TV"
    assert NOAHMP_OUTPUT_FIELDS["zsnsoxy"].netcdf_name == "ZSNSO"


@pytest.mark.skipif(not _REFERENCE_WRFOUT.is_file(),
                    reason="WRF_1974_MP55 reference bundle not present")
def test_schema_matches_the_reference_wrfout_where_they_overlap():
    """The independent copy: a wrfout stock WRF v4.6.1 actually wrote.

    Only variables the reference carries can be checked -- it ran neither
    MYNN nor Noah-MP -- but the ones it does carry check the transcription
    end to end: description, units, stagger and NetCDF type, against a file
    no part of this repository produced.
    """
    with netCDF4.Dataset(_REFERENCE_WRFOUT) as ds:
        checked: set[str] = set()
        for field in OUTPUT_FIELDS_BY_NETCDF_NAME.values():
            variable = ds.variables.get(field.netcdf_name)
            if variable is None:
                continue
            assert variable.description == field.description, field
            assert variable.units == field.units, field
            assert variable.stagger == field.stagger, field
            assert np.dtype(variable.dtype) == np.dtype(field.dtype), field
            assert int(variable.FieldType) == field.field_type, field
            checked.add(field.netcdf_name)
        # The six accumulators are named rather than counted: they are the
        # whole reason the reference is a useful oracle here, and a count
        # would still pass if the family silently left the schema.
        assert set(PRECIPITATION_OUTPUT_FIELDS) <= checked, sorted(checked)
        assert "EL_PBL" in checked, sorted(checked)


# ---------------------------------------------------------------------------
# the writer, over hand-built arrays (no device required)
# ---------------------------------------------------------------------------

def _schema_frame(nx, ny, nz):
    """One frame carrying both integer schemes and their real neighbours."""
    rng = np.random.default_rng(20260730)
    frame = {
        "T": np.zeros((nz, ny, nx), np.float32),
        "P": np.full((nz, ny, nx), 1000.0, np.float32),
        "PB": np.full((nz, ny, nx), 90000.0, np.float32),
        "PH": np.zeros((nz + 1, ny, nx), np.float32),
        "PHB": np.linspace(0.0, 1.0e5, nz + 1, dtype=np.float32)[
            :, None, None] * np.ones((1, ny, nx), np.float32),
        "XLAT": np.full((ny, nx), 40.0, np.float32),
        "XLONG": np.full((ny, nx), -100.0, np.float32),
    }
    for key in INTEGER_FIELDS:
        name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
        frame[name] = rng.integers(-3, nz, size=(ny, nx)).astype(np.int32)
    for key in REAL_FIELDS:
        name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
        frame[name] = rng.standard_normal((ny, nx)).astype(np.float32)
    frame["QKE"] = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    frame["EL_PBL"] = rng.standard_normal((nz, ny, nx)).astype(np.float32)
    frame["TSNO"] = rng.standard_normal((3, ny, nx)).astype(np.float32)
    frame["ZSNSO"] = rng.standard_normal((7, ny, nx)).astype(np.float32)
    return frame


def _write(path, frame, *, nx, ny, nz):
    attrs = {
        "MAP_PROJ": 1, "MAP_PROJ_CHAR": "Lambert Conformal",
        "TRUELAT1": np.float32(30.0), "TRUELAT2": np.float32(60.0),
        "STAND_LON": np.float32(-100.0), "MOAD_CEN_LAT": np.float32(40.0),
        "CEN_LAT": np.float32(40.0), "CEN_LON": np.float32(-100.0),
        "POLE_LAT": np.float32(90.0), "POLE_LON": np.float32(0.0),
        "START_DATE": "2026-07-01_18:00:00",
        "SIMULATION_START_DATE": "2026-07-01_18:00:00",
        "DT": np.float32(12.0), "GRIDTYPE": "C",
    }
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                      soil_layers=4, global_attrs=attrs) as writer:
        writer.write_frame("2026-07-01_18:00:36", frame)


def test_the_writer_publishes_integers_as_integers(tmp_path):
    """Four integer names across two schemes, and two real neighbours."""
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    _write(path, frame, nx=nx, ny=ny, nz=nz)

    with netCDF4.Dataset(path) as ds:
        schemes = set()
        for key, scheme in INTEGER_FIELDS.items():
            name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
            variable = ds.variables[name]
            assert variable.dtype == np.int32, (name, variable.dtype)
            assert int(variable.FieldType) == WRF_FIELD_TYPE_INTEGER, name
            np.testing.assert_array_equal(
                np.asarray(variable[0]), frame[name], err_msg=name)
            schemes.add(scheme)
        assert len(schemes) == 2, schemes
        for key in REAL_FIELDS:
            name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
            variable = ds.variables[name]
            assert variable.dtype == np.float32, (name, variable.dtype)
            assert int(variable.FieldType) == WRF_FIELD_TYPE_REAL, name


def test_the_writer_places_staggered_fields_on_wrf_axes(tmp_path):
    """Both values of the staggering dimension, on both schemes."""
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    _write(path, frame, nx=nx, ny=ny, nz=nz)

    with netCDF4.Dataset(path) as ds:
        assert ds.variables["EL_PBL"].dimensions == (
            "Time", "bottom_top_stag", "south_north", "west_east")
        assert ds.variables["EL_PBL"].stagger == "Z"
        assert ds.variables["QKE"].dimensions == (
            "Time", "bottom_top", "south_north", "west_east")
        assert ds.variables["QKE"].stagger == ""
        assert ds.variables["TSNO"].dimensions == (
            "Time", "snow_layers_stag", "south_north", "west_east")
        assert ds.variables["ZSNSO"].dimensions == (
            "Time", "snso_layers_stag", "south_north", "west_east")
        assert ds.variables["TV"].dimensions == (
            "Time", "south_north", "west_east")
        assert ds.variables["TV"].stagger == ""
        lifted = np.asarray(ds.variables["EL_PBL"][0])
        np.testing.assert_array_equal(lifted[:nz], frame["EL_PBL"])
        np.testing.assert_array_equal(lifted[nz], np.zeros((ny, nx)))


@pytest.mark.parametrize("key", sorted(INTEGER_FIELDS))
def test_the_writer_refuses_a_float_array_for_an_integer_field(key, tmp_path):
    """All four integer names, and the real-field negative control.

    netCDF4 casts on assignment, so a float array handed to an ``i4``
    variable is truncated in silence -- the schema would be honoured on the
    header and defeated on the payload.
    """
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
    frame[name] = frame[name].astype(np.float32)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    with pytest.raises(ValueError, match="declared integer"):
        _write(path, frame, nx=nx, ny=ny, nz=nz)


def test_a_real_field_still_accepts_a_float_array(tmp_path):
    """The negative control: the dtype gate is on integers, not on everything."""
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    for key in REAL_FIELDS:
        name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
        frame[name] = frame[name].astype(np.float64)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    _write(path, frame, nx=nx, ny=ny, nz=nz)
    with netCDF4.Dataset(path) as ds:
        for key in REAL_FIELDS:
            name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
            assert ds.variables[name].dtype == np.float32, name


def test_the_writer_refuses_a_field_whose_axis_contradicts_the_registry(
        tmp_path):
    """The cross-check, exercised.

    ``QKE`` is declared unstaggered, so routing it onto the staggered axis
    is a schema lie; the writer must refuse rather than publish it.  Without
    this the stagger attribute would be decoration.
    """
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    writer = WrfoutWriter(path, nx=5, ny=4, nz=6, dx=3000.0, dy=3000.0,
                          soil_layers=4)
    try:
        with pytest.raises(ValueError, match="EL_PBL"):
            writer._create_variable(
                "EL_PBL",
                ("Time", "bottom_top", "south_north", "west_east"))
        with pytest.raises(ValueError, match="QKE"):
            writer._create_variable(
                "QKE",
                ("Time", "bottom_top_stag", "south_north", "west_east"))
    finally:
        writer.abort()


# ---------------------------------------------------------------------------
# the physics-selector globals
# ---------------------------------------------------------------------------

#: Two resolved configurations that differ in every selector gpuwm owns.
#: One value of a selector proves nothing about the selector -- it could be a
#: constant -- so each gate below runs at both.
_SELECTOR_CONFIGS = {
    "wsm6_nocu_ysu_noah": dict(
        mp_physics=6, cu_physics=0, bl_pbl_physics=1,
        sf_sfclay_physics=1, sf_surface_physics=2,
        ra_lw_physics=1, ra_sw_physics=1),
    "morrison_kf_mynn_noahmp": dict(
        mp_physics=10, cu_physics=1, bl_pbl_physics=5,
        sf_sfclay_physics=5, sf_surface_physics=4,
        ra_lw_physics=4, ra_sw_physics=4),
}


def _selector_run(**overrides):
    from gpuwm.config import RunConfig

    return RunConfig(nx=5, ny=4, nz=6, dx=3000.0, dy=3000.0, ztop=12000.0,
                     dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                     **overrides)


def _selector_grid():
    from types import SimpleNamespace

    return SimpleNamespace(
        wrf_map_proj=1, map_proj_char="Lambert Conformal",
        truelat1=30.0, truelat2=60.0, stand_lon=-100.0,
        ref_lat=40.0, ref_lon=-100.0, moad_cen_lat=40.0,
        cen_lat=40.0, cen_lon=-100.0)


def test_every_selector_global_is_one_the_stock_wrfout_carries():
    """The set is enumerated from the artifact, not invented.

    ``PHYSICS_SELECTOR_GLOBALS`` may only name attributes stock WRF
    actually writes; this is the gate that says so, against a file no part
    of this repository produced.
    """
    if not _REFERENCE_WRFOUT.is_file():
        pytest.skip("WRF_1974_MP55 reference bundle not present")
    with netCDF4.Dataset(_REFERENCE_WRFOUT) as ds:
        carried = set(ds.ncattrs())
        for selector in PHYSICS_SELECTOR_GLOBALS:
            assert selector.name in carried, selector
            # WRF writes every one of these as NC_INT.
            value = getattr(ds, selector.name)
            assert np.asarray(value).dtype == np.int32, selector


def test_every_selector_global_cites_the_registry():
    for selector in PHYSICS_SELECTOR_GLOBALS:
        assert selector.registry.startswith("Registry.EM_COMMON:"), selector
        assert selector.source in (
            "config", "radiation_lw", "radiation_sw", "unimplemented"), (
                selector)
        if selector.source == "config":
            assert selector.run_config_field, selector
        else:
            assert not selector.run_config_field, selector


@pytest.mark.parametrize("label", sorted(_SELECTOR_CONFIGS))
def test_the_resolver_reports_the_configured_selectors(label):
    """Two configurations, and every selector gpuwm owns differs between."""
    from gpuwm.io.wrfout import wrf_physics_selector_attrs

    overrides = _SELECTOR_CONFIGS[label]
    attrs = wrf_physics_selector_attrs(_selector_run(**overrides))
    assert set(attrs) == {s.name for s in PHYSICS_SELECTOR_GLOBALS}
    for name, value in attrs.items():
        assert np.asarray(value).dtype == np.int32, name
    for field, expected in overrides.items():
        assert attrs[field.upper()] == expected, field
    # The four gpuwm does not implement are off, at both configurations.
    for name in ("SHCU_PHYSICS", "SF_URBAN_PHYSICS", "SF_SURFACE_MOSAIC",
                 "SF_OCEAN_PHYSICS"):
        assert attrs[name] == 0, name


def test_the_two_configurations_actually_differ():
    """The negative control for the parametrisation above.

    If the two rows happened to agree on a selector, that selector would be
    swept at one value while looking like it was swept at two.
    """
    from gpuwm.io.wrfout import wrf_physics_selector_attrs

    a, b = (wrf_physics_selector_attrs(_selector_run(**overrides))
            for overrides in _SELECTOR_CONFIGS.values())
    differing = {name for name in a if a[name] != b[name]}
    assert differing == {"MP_PHYSICS", "CU_PHYSICS", "BL_PBL_PHYSICS",
                         "SF_SFCLAY_PHYSICS", "SF_SURFACE_PHYSICS",
                         "RA_LW_PHYSICS", "RA_SW_PHYSICS"}, differing


def test_the_legacy_radiation_sentinel_is_resolved_not_written():
    """gpuwm's ``-1/-1`` is not a WRF scheme id and must never reach a file.

    It means "use the aggregate ``ra_physics``". Writing the sentinel would
    put a number in the file that no WRF selector has, so the resolver --
    the repository's own authority for what actually ran -- is what the
    attribute reports.
    """
    from gpuwm.io.wrfout import wrf_physics_selector_attrs

    legacy = _selector_run(ra_lw_physics=-1, ra_sw_physics=-1, ra_physics=4)
    attrs = wrf_physics_selector_attrs(legacy)
    assert attrs["RA_LW_PHYSICS"] == 4
    assert attrs["RA_SW_PHYSICS"] == 4
    off = wrf_physics_selector_attrs(
        _selector_run(ra_lw_physics=-1, ra_sw_physics=-1, ra_physics=0))
    assert off["RA_LW_PHYSICS"] == 0 and off["RA_SW_PHYSICS"] == 0


@pytest.mark.parametrize("label", sorted(_SELECTOR_CONFIGS))
def test_a_real_wrfout_carries_the_selectors_of_its_own_run(label, tmp_path):
    """Through the real writer, on a real file, at both configurations."""
    from gpuwm.io.wrfout import wrf_global_attrs

    overrides = _SELECTOR_CONFIGS[label]
    run = _selector_run(**overrides)
    attrs = wrf_global_attrs(
        _selector_grid(), datetime(2026, 7, 1, 18, 0, 0), run=run)
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                      soil_layers=4, global_attrs=attrs) as writer:
        writer.write_frame("2026-07-01_18:00:36", frame)
    with netCDF4.Dataset(path) as ds:
        for field, expected in overrides.items():
            stored = getattr(ds, field.upper())
            assert stored == expected, field
            assert np.asarray(stored).dtype == np.int32, field
        assert ds.SHCU_PHYSICS == 0


def test_a_writer_given_no_run_stamps_no_selector(tmp_path):
    """The negative control: they come from a resolved config or not at all.

    An idealized caller with no ``RunConfig`` must not receive eleven
    zeros, which would be eleven claims it is not entitled to make.
    """
    from gpuwm.io.wrfout import wrf_global_attrs

    attrs = wrf_global_attrs(
        _selector_grid(), datetime(2026, 7, 1, 18, 0, 0))
    for selector in PHYSICS_SELECTOR_GLOBALS:
        assert selector.name not in attrs, selector


def test_wrf_rust_opens_a_file_carrying_the_selector_globals(tmp_path):
    """What the mandated reader can actually be asked here, and what it cannot.

    wrf-rust 0.2.35 exposes **no** global-attribute accessor: its public
    surface is ``nx/ny/nz/nt``, ``dx``, ``dy``, ``times``, ``path`` and
    ``getvar``, and ``getvar("MP_PHYSICS")`` raises ``unknown variable``
    because a global is not a variable. So the witness for attribute values
    is the file itself, read with netCDF4, above.

    What wrf-rust *can* answer is whether adding eleven globals perturbs
    it, and whether the globals it does consume still arrive: ``dx``/``dy``
    are read straight out of the global set, so they are the one attribute
    read its API offers, and they are asserted here.
    """
    wrf = pytest.importorskip("wrf")
    if not hasattr(wrf, "WrfFile"):        # NCAR wrf-python, not wrf-rust
        pytest.skip("the installed 'wrf' module is not wrf-rust")
    from gpuwm.io.wrfout import wrf_global_attrs

    run = _selector_run(**_SELECTOR_CONFIGS["morrison_kf_mynn_noahmp"])
    attrs = wrf_global_attrs(
        _selector_grid(), datetime(2026, 7, 1, 18, 0, 0), run=run)
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    with WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0,
                      soil_layers=4, global_attrs=attrs) as writer:
        writer.write_frame("2026-07-01_18:00:36", frame)

    reader = wrf.WrfFile(str(path))
    assert (reader.nx, reader.ny, reader.nz, reader.nt) == (nx, ny, nz, 1)
    assert (reader.dx, reader.dy) == (3000.0, 3000.0)
    assert reader.times() == ["2026-07-01_18:00:36"]
    with pytest.raises(Exception, match="MP_PHYSICS"):
        reader.getvar("MP_PHYSICS")


def test_wrf_rust_reads_the_written_scheme_fields(tmp_path):
    """The mandated downstream reader, on a file the real writer produced.

    ``getvar`` here is wrf-rust's raw-variable path -- these are not its
    computed diagnostics -- so a name it cannot resolve is a name the file
    does not carry.  Before v1.1.3 ``ISNOW`` was such a name.
    """
    wrf = pytest.importorskip("wrf")
    if not hasattr(wrf, "WrfFile"):        # NCAR wrf-python, not wrf-rust
        pytest.skip("the installed 'wrf' module is not wrf-rust")
    nx, ny, nz = 5, 4, 6
    frame = _schema_frame(nx, ny, nz)
    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    _write(path, frame, nx=nx, ny=ny, nz=nz)

    reader = wrf.WrfFile(str(path))
    assert reader.times() == ["2026-07-01_18:00:36"]
    read = 0
    for key in (*INTEGER_FIELDS, *REAL_FIELDS):
        name = SCHEME_OUTPUT_FIELDS[key].netcdf_name
        if name in WRF_RUST_SHADOWED_NAMES:
            continue
        values = np.asarray(reader.getvar(name))
        np.testing.assert_array_equal(
            values, np.asarray(frame[name], dtype=np.float64), err_msg=name)
        read += 1
    assert read == len(INTEGER_FIELDS) + len(REAL_FIELDS) - 1
    for absent in ("ISNOWXY", "PGSXY", "TVXY", "TSNOXY", "ZSNSOXY"):
        with pytest.raises(Exception) as excinfo:
            reader.getvar(absent)
        assert absent in str(excinfo.value)
