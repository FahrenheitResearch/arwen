# tests/test_wrfout_surface_identity.py
"""gpuwm's own wrfout carries the land/soil identity its own child needs.

THE DEFECT THIS PINS.  ``gpuwm downscale`` takes a child-grid history file
as ``--child-surface-from`` and refuses to fabricate land identity; the
nine fields it requires are ``gpuwm.offline_child._SURFACE_REQUIRED_FIELDS``.
gpuwm's history writer published six of the nine.  ISLTYP, TMN and VEGFRA
were carried in ``PhysicsDriver.fields`` from initialization, written into
every restart, read by the land-surface scheme every step -- and never
emitted.  So a gpuwm wrfout was refused as a surface source for a gpuwm
child, on a real 12 km parent and again on a nested d02 (measured at the
2.2.1 cut).

The tests here are written against the two SEAMS rather than against a
run: ``_live_state_history_fields``, the pure mapping both frame builders
consume, and the file a ``WrfoutWriter`` actually emits -- read back with
netCDF4 and handed to the child's own reader.  That last one is the point:
the acceptance is proven by the CONSUMER, not by a list in this file
agreeing with a list in that one.
"""

import numpy as np
import netCDF4
import pytest

from types import SimpleNamespace

from gpuwm.io.wrfout import WrfoutWriter, _live_state_history_fields
from gpuwm.io.wrf_output_schema import SURFACE_IDENTITY_OUTPUT_FIELDS
from gpuwm.offline_child import (
    OfflineChildContractError,
    _SURFACE_REQUIRED_FIELDS,
    read_child_surface_state,
)

_NY, _NX, _SOIL = 3, 4, 4

#: The five rows this lane added, and the driver field each reads.
_IDENTITY_ROWS = (
    ("ISLTYP", "isltyp"), ("IVGTYP", "ivgtyp"),
    ("TMN", "tmn"), ("VEGFRA", "vegfra"), ("SEAICE", "xice"),
)


def _driver_fields():
    """A land-surface driver's live ``fields``, NumPy so the test is CPU-only.

    Integer dtypes on the two category fields on purpose: WRF declares
    ISLTYP/IVGTYP ``integer`` and the writer must carry that through to
    ``i4``/``FieldType=106`` rather than silently widening them to float.
    """
    return {
        "snow": np.full((_NY, _NX), 21.0, np.float32),
        "snowh": np.full((_NY, _NX), 22.0, np.float32),
        "snowc": np.full((_NY, _NX), 23.0, np.float32),
        "tslb": np.full((_SOIL, _NY, _NX), 285.0, np.float32),
        "smois": np.full((_SOIL, _NY, _NX), 0.25, np.float32),
        "sh2o": np.full((_SOIL, _NY, _NX), 0.24, np.float32),
        "isltyp": np.full((_NY, _NX), 8, np.int32),
        "ivgtyp": np.full((_NY, _NX), 10, np.int32),
        "tmn": np.full((_NY, _NX), 287.5, np.float32),
        "vegfra": np.full((_NY, _NX), 62.5, np.float32),
        "xice": np.zeros((_NY, _NX), np.float32),
        "tsk": np.full((_NY, _NX), 290.0, np.float32),
    }


def _state(*, land_surface: bool):
    """A state whose land-surface scheme is routed, or is not.

    ``scheme_dispatch`` is the driver's OWN resolved routing, which is the
    gate the writer reads -- so "no LSM" here is the real condition an
    idealized or microphysics-only run presents, not a deleted dict key.
    """
    dispatch = {"sf_surface_physics": "_run_noah" if land_surface else None}
    return SimpleNamespace(
        qi=None,
        physics=SimpleNamespace(
            fields=_driver_fields(), microphysics=None,
            scheme_dispatch=dispatch))


def test_land_surface_run_publishes_the_five_identity_rows():
    """The mapping both frame builders consume carries all five."""

    history = _live_state_history_fields(_state(land_surface=True))
    fields = _driver_fields()
    for netcdf_name, driver_key in _IDENTITY_ROWS:
        assert netcdf_name in history, (
            f"{netcdf_name} is missing from the live history mapping; "
            f"the driver carries it as fields[{driver_key!r}]")
        np.testing.assert_array_equal(
            np.asarray(history[netcdf_name]), fields[driver_key])


def test_a_run_without_a_land_surface_scheme_publishes_none_of_them():
    """The HONEST half, and the control for the test above.

    Without an LSM these arrays hold ``initialize_physics``'s scalar
    cold-start defaults -- soil category 6, vegetation fraction 50, 285 K --
    which describe no ground anywhere.  Publishing them would put a
    fabricated soil category into an idealized run's history under WRF's
    own name, which is worse than omitting it: a reader cannot tell.
    """

    history = _live_state_history_fields(_state(land_surface=False))
    emitted = {name for name, _ in _IDENTITY_ROWS} & set(history)
    assert not emitted, (
        f"{sorted(emitted)} published by a run with no land-surface scheme")


def _write_frame(path):
    """One frame through the real writer, with the production attribute set."""

    fields = _driver_fields()
    frame = {
        "T2": np.full((_NY, _NX), 290.0, np.float32),
        "PSFC": np.full((_NY, _NX), 98000.0, np.float32),
        "TSK": fields["tsk"],
        # LANDMASK/LU_INDEX reach the frame from the static geography
        # (gpuwm.runtime._metadata_frame), not from the driver, so they are
        # supplied here the way the production path supplies them.
        "LANDMASK": np.ones((_NY, _NX), np.float32),
        "LU_INDEX": np.full((_NY, _NX), 10.0, np.float32),
        **_live_state_history_fields(_state(land_surface=True)),
    }
    # The landuse identity WrfoutWriter's caller stamps.  read_child_surface
    # _state requires all four as evidence rather than assuming a table.
    attrs = {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
        "ISWATER": 17, "ISLAKE": 21, "ISICE": 15, "ISOILWATER": 14,
    }
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=5, dx=1000.0, dy=1000.0,
                      global_attrs=attrs, soil_layers=_SOIL) as writer:
        writer.write_frame("2020-06-01_12:00:00", frame)
    return path


def test_emitted_rows_carry_the_registry_schema(tmp_path):
    """Type, FieldType, description, units and stagger are WRF's own.

    Transcribed from the pinned v4.6.1 Registry and confirmed against the
    group's stock wrfout; asserted here against the file rather than against
    the schema table, so a writer that ignored the row would still fail.
    """

    path = _write_frame(tmp_path / "wrfout_d01_identity")
    with netCDF4.Dataset(path) as dataset:
        for netcdf_name, _ in _IDENTITY_ROWS:
            assert netcdf_name in dataset.variables, netcdf_name
            var = dataset.variables[netcdf_name]
            row = SURFACE_IDENTITY_OUTPUT_FIELDS[netcdf_name]
            assert var.dtype == np.dtype(row.dtype), netcdf_name
            assert int(var.FieldType) == row.field_type, netcdf_name
            assert var.description == row.description, netcdf_name
            assert var.units == row.units, netcdf_name
            assert var.stagger == row.stagger == "", netcdf_name
            assert var.MemoryOrder == "XY ", netcdf_name
            assert var.dimensions == (
                "Time", "south_north", "west_east"), netcdf_name


def test_the_two_category_rows_survive_the_round_trip_as_integers(tmp_path):
    """A smoothed or widened category is not a valid surface identity.

    ``read_child_surface_state`` refuses a category field that is not
    exactly integral, so the writer's dtype choice is load-bearing rather
    than cosmetic.
    """

    path = _write_frame(tmp_path / "wrfout_d01_categories")
    with netCDF4.Dataset(path) as dataset:
        for name, expected in (("ISLTYP", 8), ("IVGTYP", 10)):
            value = np.asarray(dataset.variables[name][0])
            assert value.dtype == np.int32, name
            assert np.array_equal(value, np.full((_NY, _NX), expected)), name


def test_our_own_wrfout_is_accepted_as_a_child_surface_source(tmp_path):
    """THE USER-STORY SEAM, proven by the consumer rather than by a list.

    The child's own reader, at the child's own grid, with no special case
    for a gpuwm producer: it either finds the nine fields and the four
    identity attributes or it raises.
    """

    path = _write_frame(tmp_path / "wrfout_d01_surface_source")
    surface = read_child_surface_state(
        path, child_ny=_NY, child_nx=_NX, num_soil_layers=_SOIL)

    assert set(_SURFACE_REQUIRED_FIELDS) <= set(surface.fields)
    assert surface.identity["MMINLU"] == "MODIFIED_IGBP_MODIS_NOAH"
    assert surface.identity["ISOILWATER"] == 14
    np.testing.assert_array_equal(
        surface.fields["ISLTYP"], np.full((_NY, _NX), 8.0, np.float32))
    np.testing.assert_array_equal(
        surface.fields["TMN"], np.full((_NY, _NX), 287.5, np.float32))
    np.testing.assert_array_equal(
        surface.fields["VEGFRA"], np.full((_NY, _NX), 62.5, np.float32))
    # Carried, not defaulted: SEAICE reaches the reader as a real field
    # instead of the zeros it substitutes for an absent one.
    assert "SEAICE" in surface.fields


def test_a_nested_d02_frame_is_accepted_at_the_child_grid(tmp_path):
    """THE OTHER SHAPE THE 2.2.1 CUT PROVED FAILING.

    The legitimate way to get a child-grid surface source out of gpuwm is a
    NESTED domain whose grid is the child's: run d01 12 km + d02 3 km, then
    downscale d01's archive onto d02's grid with d02's own frame as
    ``--child-surface-from``.  That attempt failed for exactly the same
    reason the parent-grid one did -- three missing fields -- and the
    domain-topology globals a nest carries are the only thing that makes
    this file different, so they are what this test adds.
    """

    fields = _driver_fields()
    frame = {
        "TSK": fields["tsk"],
        "LANDMASK": np.ones((_NY, _NX), np.float32),
        "LU_INDEX": np.full((_NY, _NX), 10.0, np.float32),
        **_live_state_history_fields(_state(land_surface=True)),
    }
    attrs = {
        "MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
        "ISWATER": 17, "ISLAKE": 21, "ISICE": 15, "ISOILWATER": 14,
        # A nest, not a root: this is a d02 history frame.
        "GRID_ID": np.int32(2), "PARENT_ID": np.int32(1),
        "I_PARENT_START": np.int32(26), "J_PARENT_START": np.int32(21),
        "PARENT_GRID_RATIO": np.int32(4),
    }
    path = tmp_path / "wrfout_d02_surface_source"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=5, dx=3000.0, dy=3000.0,
                      global_attrs=attrs, soil_layers=_SOIL) as writer:
        writer.write_frame("2020-06-01_12:00:00", frame)

    surface = read_child_surface_state(
        path, child_ny=_NY, child_nx=_NX, num_soil_layers=_SOIL)
    assert set(_SURFACE_REQUIRED_FIELDS) <= set(surface.fields)
    # The receipt names the file, so a run's provenance records WHICH
    # domain's frame seeded the child rather than "a wrfout".
    assert surface.receipt["path"].endswith("wrfout_d02_surface_source")


def test_a_file_missing_the_three_rows_is_still_refused(tmp_path):
    """RED-ON-REVERT, expressed as a property of the reader.

    Reverting the writer produces exactly this file, so this is the failure
    the 2.2.1 cut met -- kept as a test so the fix cannot regress into a
    reader that shrugged instead.
    """

    fields = _driver_fields()
    frame = {
        "TSK": fields["tsk"],
        "LANDMASK": np.ones((_NY, _NX), np.float32),
        "LU_INDEX": np.full((_NY, _NX), 10.0, np.float32),
        "SNOW": fields["snow"], "SNOWH": fields["snowh"],
        "TSLB": fields["tslb"], "SMOIS": fields["smois"],
        "SH2O": fields["sh2o"],
    }
    attrs = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
             "ISWATER": 17, "ISLAKE": 21, "ISICE": 15}
    path = tmp_path / "wrfout_d01_pre_fix"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=5, dx=1000.0, dy=1000.0,
                      global_attrs=attrs, soil_layers=_SOIL) as writer:
        writer.write_frame("2020-06-01_12:00:00", frame)

    with pytest.raises(OfflineChildContractError) as excinfo:
        read_child_surface_state(
            path, child_ny=_NY, child_nx=_NX, num_soil_layers=_SOIL)
    message = str(excinfo.value)
    for name in ("ISLTYP", "TMN", "VEGFRA"):
        assert name in message, message
