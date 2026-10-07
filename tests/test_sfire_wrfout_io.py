"""Refined fire grids and atmospheric exchange reach the real history writer."""
from datetime import datetime

import netCDF4
import numpy as np
import pytest

from gpuwm.io.wrfout import WrfoutWriter
from gpuwm.io.wrf_output_schema import HISTORY_FIELDS_BY_NETCDF_NAME
from gpuwm.io.sfire_schema import SFIRE_REGISTRY_FIELDS


def test_fire_history_preserves_native_dimensions_words_and_metadata(tmp_path):
    ny, nx, nz, rx, ry = 4, 6, 5, 3, 2
    fine = np.arange(ny*ry*nx*rx, dtype=np.float32).reshape(ny*ry, nx*rx)
    fields = {"FIRE_AREA": fine, "NFUEL_CAT": fine + 1,
              "FGRNHFX": fine + 2, "LFN": -fine,
              "FXLAT": fine * np.float32(0.01), "FXLONG": fine * np.float32(0.02),
              "GRNHFX": np.ones((ny, nx), np.float32),
              "UAH": np.ones((ny, nx + 1), np.float32),
              "VAH": np.ones((ny + 1, nx), np.float32),
              "RTHFRTEN": np.ones((nz, ny, nx), np.float32),
              "RQVFRTEN": np.ones((nz, ny, nx), np.float32),
              "FMC_GC": np.ones((5, ny, nx), np.float32),
              "FMC_EQUI": np.ones((5, ny, nx), np.float32),
              "FMC_TEND": np.ones((5, ny, nx), np.float32),
              "FMEP": np.ones((2, ny, nx), np.float32),
              "LFN_HIST": fine + 4,
              "LFN_TIME": np.array([-125.5], np.float32)}
    path = tmp_path / "wrfout_fire.nc"
    writer = WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=90, dy=120,
                         field_schema=fields, engine="rust")
    assert writer.engine == "rust"
    writer.write_frame("2026-10-02_00:00:00", fields)
    writer.close()
    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["west_east_subgrid"]) == nx * rx
        assert len(ds.dimensions["south_north_subgrid"]) == ny * ry
        assert ds.SR_X == rx and ds.SR_Y == ry
        assert ds.variables["FIRE_AREA"].dimensions == ("Time", "south_north_subgrid", "west_east_subgrid")
        assert ds.variables["UAH"].dimensions == ("Time", "south_north", "west_east_stag")
        assert ds.variables["VAH"].dimensions == ("Time", "south_north_stag", "west_east")
        assert ds.variables["RTHFRTEN"].dimensions == ("Time", "bottom_top_stag", "south_north", "west_east")
        assert ds.variables["FMC_GC"].dimensions == ("Time", "fuel_moisture_classes_stag", "south_north", "west_east")
        assert ds.variables["LFN_TIME"].dimensions == ("Time", "i_lfn_history")
        assert ds.variables["FIRE_AREA"].coordinates == "FXLONG FXLAT"
        for name, expected in fields.items():
            variable = ds.variables[name]
            row = SFIRE_REGISTRY_FIELDS[name]
            assert variable.description == row[2] and variable.units == row[3]
            assert variable.stagger == row[1] and variable.FieldType == 104
            got = variable[0].data
            if name in ("RTHFRTEN", "RQVFRTEN"):
                assert not np.any(got[-1])
                got = got[:-1]
            if name in ("FMC_GC", "FMC_EQUI", "FMC_TEND", "FMEP"):
                assert got.shape == expected.shape
            assert got.tobytes() == expected.tobytes()


def test_fire_schema_uses_original_registry_names_and_real_fuel_categories():
    assert HISTORY_FIELDS_BY_NETCDF_NAME["NFUEL_CAT"].dtype == "f4"
    assert HISTORY_FIELDS_BY_NETCDF_NAME["FMC_TEND"].units == "h"
    assert HISTORY_FIELDS_BY_NETCDF_NAME["RQVFRTEN"].units == ""
    assert HISTORY_FIELDS_BY_NETCDF_NAME["UAH"].stagger == "X"
    assert HISTORY_FIELDS_BY_NETCDF_NAME["VAH"].stagger == "Y"


def test_fire_history_planner_prices_native_refinement_and_fixed_class_extents():
    from gpuwm.config import RunConfig
    from gpuwm.io.history_layout import produced_history_shapes
    cfg = RunConfig(nx=16, ny=12, nz=16, ifire=2, sr_x=4, sr_y=3,
                    dx=90.0, dy=120.0, ztop=8000.0, dt=0.25, run_seconds=0.0,
                    moist=True, fire_fuel_read=0, fmoist_run=True, fire_fmc_read=0)
    shapes = produced_history_shapes(cfg)
    assert shapes["FIRE_AREA"] == (36, 64)
    assert shapes["UF"] == (36, 64) and shapes["VF"] == (36, 64)
    assert shapes["UAH"] == (12, 17) and shapes["VAH"] == (13, 16)
    assert shapes["RTHFRTEN"] == (17, 12, 16)
    assert shapes["FMC_TEND"] == (5, 12, 16)
    assert shapes["FMC_GC"] == (5, 12, 16)
    assert shapes["FMEP"] == (2, 12, 16)
    assert shapes["FMOIST_LASTTIME"] == () and shapes["FMOIST_NEXTTIME"] == ()


def test_fractional_fire_refinement_refuses_misplaced_history_grid(tmp_path):
    writer = WrfoutWriter(tmp_path / "broken.nc", nx=6, ny=4, nz=5, dx=90, dy=120)
    with pytest.raises(ValueError, match="refine every atmospheric mass cell"):
        writer.write_frame("2026-10-02_00:00:00", {"FIRE_AREA": np.zeros((8, 19), np.float32)})
    writer.abort()


@pytest.mark.parametrize("mode,units", [("metric", "m"), ("geographic", "degrees")])
def test_fire_coordinate_mode_labels_actual_fine_axes(tmp_path, mode, units):
    from types import SimpleNamespace
    from gpuwm.io.wrfout import fire_history_attrs, carrier_provenance_attrs
    attributes = dict(FIRE_COORDINATE_MODE=mode, IFIRE=2, SR_X=3, SR_Y=2)
    physics = SimpleNamespace(fire=SimpleNamespace(output_attributes=lambda: attributes))
    assert fire_history_attrs(physics) == attributes
    assert carrier_provenance_attrs(physics) == attributes
    fields = {"FXLAT": np.zeros((8, 18), np.float32), "FXLONG": np.ones((8, 18), np.float32)}
    path = tmp_path / (mode + ".nc")
    with WrfoutWriter(path, nx=6, ny=4, nz=5, dx=90, dy=120,
                      global_attrs=attributes, field_schema=fields, engine="rust") as writer:
        writer.write_frame("2026-10-02_00:00:00", fields)
    with netCDF4.Dataset(path) as ds:
        assert ds.FIRE_COORDINATE_MODE == mode
        assert ds.variables["FXLAT"].units == ds.variables["FXLONG"].units == units
        if mode == "metric":
            assert ds.variables["FXLAT"].standard_name == "projection_y_coordinate"
            assert ds.variables["FXLONG"].standard_name == "projection_x_coordinate"
