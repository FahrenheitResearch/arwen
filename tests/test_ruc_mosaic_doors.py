"""RUC fractions must reach physics unchanged by storage or rank windows.

These controls prevent a selected mosaic from silently taking dominant
categories, mixing neighboring columns, or applying real.exe edits twice.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.ruc_mosaic import (
    ruc_mosaic_physics_inputs, wrfinput_ruc_mosaic_inputs)


ATTRS = dict(MMINLU="MODIFIED_IGBP_MODIS_NOAH", ISWATER=17,
             ISLAKE=21, ISICE=15)


def _cfg(**values):
    return SimpleNamespace(**(dict(sf_surface_physics=3, mosaic_lu=1,
                                   mosaic_soil=1) | values))


def _static(ny=3, nx=4):
    land = np.zeros((21, ny, nx), np.float32)
    soil = np.zeros((16, ny, nx), np.float32)
    land[6] = np.float32(.7)
    land[9] = np.float32(.2)
    land[20] = np.float32(.1)
    soil[2] = np.float32(.25)
    soil[5] = np.float32(.75)
    return dict(LANDUSEF=land, SOILCTOP=soil,
                LANDMASK=np.ones((ny, nx), np.float32))


@pytest.mark.parametrize("cfg", [_cfg(mosaic_lu=0, mosaic_soil=0),
                                 _cfg(sf_surface_physics=2)])
def test_off_never_reads_fractions(cfg):
    assert ruc_mosaic_physics_inputs(cfg, None) == {}
    assert wrfinput_ruc_mosaic_inputs(None, cfg) == {}


@pytest.mark.parametrize("option,name", [("mosaic_lu", "LANDUSEF"),
                                         ("mosaic_soil", "SOILCTOP")])
def test_missing_fraction_cannot_silently_select_dominant_category(option, name):
    cfg = _cfg(mosaic_lu=0, mosaic_soil=0)
    setattr(cfg, option, 1)
    with pytest.raises(ValueError, match=name + ".*dominant category"):
        ruc_mosaic_physics_inputs(cfg, {}, processed=True)


def test_processed_fractions_are_not_edited_or_normalized():
    static = _static()
    static["LANDUSEF"][20] = np.float32(.3)
    inputs = ruc_mosaic_physics_inputs(_cfg(), static, processed=True)
    assert inputs["landusef"] is static["LANDUSEF"]
    assert inputs["soilctop"] is static["SOILCTOP"]
    assert inputs["landusef"][20, 0, 0] == np.float32(.3)


@pytest.mark.parametrize("fractional,threshold", [(False, .5), (True, .02)])
def test_raw_fractions_merge_lakes_and_make_seaice_one_hot(fractional, threshold):
    static = _static()
    static["LANDMASK"][0, :3] = 0
    before = {key: value.copy() for key, value in static.items()}
    ice = np.zeros((3, 4), np.float32)
    ice[0, :3] = [threshold, np.nextafter(np.float32(threshold), np.float32(0)), 0]
    ice[1, 0] = 1  # real.exe clears XICE on land before its fraction edits.
    result = ruc_mosaic_physics_inputs(
        _cfg(), static, landuse_attrs=ATTRS, xice=ice,
        fractional_seaice=fractional)
    expected_land = before["LANDUSEF"].copy()
    expected_land[16] = np.float32(.1)
    expected_land[20] = 0
    expected_land[:, 0, 0] = 0
    expected_land[14, 0, 0] = 1
    expected_soil = before["SOILCTOP"].copy()
    expected_soil[:, 0, 0] = 0
    expected_soil[15, 0, 0] = 1
    np.testing.assert_array_equal(result["landusef"], expected_land)
    np.testing.assert_array_equal(result["soilctop"], expected_soil)
    for key in static:
        np.testing.assert_array_equal(static[key], before[key])


def test_soil_only_does_not_require_land_fraction_or_landuse_identity():
    static = _static()
    del static["LANDUSEF"]
    result = ruc_mosaic_physics_inputs(
        _cfg(mosaic_lu=0), static, xice=0)
    assert set(result) == {"soilctop"}
    np.testing.assert_array_equal(result["soilctop"], static["SOILCTOP"])


@pytest.mark.parametrize("name,category", [("LANDUSEF", "land_cat"),
                                         ("SOILCTOP", "soil_cat")])
def test_wrfinput_uses_reader_axes_and_keeps_real_exe_fractions(
        monkeypatch, name, category):
    from gpuwm import netcdf_bridge
    from gpuwm.ingest import wrfinput
    static = _static()
    cfg = _cfg(mosaic_lu=int(name == "LANDUSEF"),
               mosaic_soil=int(name == "SOILCTOP"))
    variable = SimpleNamespace(
        dimensions=("Time", category, "south_north", "west_east"),
        data=static[name])
    monkeypatch.setattr(netcdf_bridge, "open_dataset", lambda path: nullcontext(
        SimpleNamespace(variables={name: variable})))
    monkeypatch.setattr(wrfinput, "_read_numeric", lambda value: value.data)
    restored = SimpleNamespace(raw={}, path="fractions.nc")
    result = wrfinput_ruc_mosaic_inputs(restored, cfg)
    np.testing.assert_array_equal(result[name.lower()], static[name])
    variable.dimensions = ("Time", "south_north", category, "west_east")
    with pytest.raises(ValueError, match="axes.*different columns"):
        wrfinput_ruc_mosaic_inputs(restored, cfg)


def test_four_static_windows_preserve_every_category_and_column():
    from tilestream.real_init import slice_mapping
    from tilestream.spec import plan_tiles
    ny, nx = 8, 10
    static = _static(ny, nx)
    # Distinct fractions in every column detect an accidental vertical or
    # category slice, even where one category count equals the atmosphere nz.
    weights = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx) / np.float32(100)
    static["LANDUSEF"][6] = weights
    static["LANDUSEF"][9] = np.float32(.9) - weights
    static["SOILCTOP"][2] = weights
    static["SOILCTOP"][5] = np.float32(1) - weights
    whole = ruc_mosaic_physics_inputs(
        _cfg(), static, landuse_attrs=ATTRS, xice=0)
    for spec in plan_tiles(nx, ny, 5, 4, 0, periodic=False):
        window = slice_mapping(static, spec, nz=21, ny=ny, nx=nx, label="static")
        split = ruc_mosaic_physics_inputs(
            _cfg(), window, landuse_attrs=ATTRS, xice=0)
        for name in whole:
            expected = whole[name][:, spec.cj0:spec.cj0 + spec.cny,
                                   spec.ci0:spec.ci0 + spec.cnx]
            np.testing.assert_array_equal(split[name].view("u4"), expected.view("u4"))


def test_prepared_initializer_hands_both_fractions_to_selected_physics(monkeypatch):
    from gpuwm.ingest import hrrr_physics
    import gpuwm.core.diagnostics as diagnostics
    import gpuwm.core.landuse as landuse
    import gpuwm.core.physics as physics
    import gpuwm.core.radiation_composition as radiation
    one = np.ones((3, 4), np.float32)
    cfg = _cfg(nx=4, ny=3, num_soil_layers=9, hypsometric_opt=2,
               sf_lake_physics=1)
    static = _static()
    static.update(LU_INDEX=7*one, SCT_DOM=6*one,
                  GREENFRAC=np.full((12, 3, 4), .5, np.float32),
                  LAI12M=np.full((12, 3, 4), 2, np.float32),
                  LAKE_DEPTH=np.full((3, 4), 12, np.float32),
                  LAKE_DEPTH_FLAG=1, LAKEMASK=np.zeros((3, 4), np.float32))
    fields = {name: one.copy() for name in hrrr_physics._CANONICAL_SURFACE_FIELDS}
    fields.update(TSK=280*one, TMN=280*one,
                  TSLB=np.full((9, 3, 4), 280, np.float32),
                  SMOIS=np.full((9, 3, 4), .2, np.float32),
                  SH2O=np.full((9, 3, 4), .2, np.float32),
                  SEAICE=0*one, SNOW=0*one, SNOWH=0*one)
    met = SimpleNamespace(fields=dict(LANDSEA=one, T2=280*one,
                         SKINTEMP=280*one, U10=np.ones((3, 5)), V10=np.ones((4, 4))))
    result = SimpleNamespace(state=object(), surface_pressure=99000*one,
                             surface_qv=.01*one)
    grid = SimpleNamespace(ref_lat=38., latlon_mass=lambda: (38*one, -97*one))
    monkeypatch.setitem(sys.modules, "cupy", np)
    monkeypatch.setattr(diagnostics, "update_diagnostics", lambda *args: None)
    monkeypatch.setattr(landuse, "initialize_landuse", lambda *args, **kwargs: object())
    monkeypatch.setattr(radiation, "make_radiation", lambda *args, **kwargs: None)
    captured = {}
    class ReachedPhysics(Exception):
        pass
    def initialize(*args, **kwargs):
        captured.update(kwargs)
        raise ReachedPhysics
    monkeypatch.setattr(physics, "initialize_physics", initialize)
    with pytest.raises(ReachedPhysics):
        hrrr_physics.initialize_prepared_physics(
            result, cfg, met, SimpleNamespace(fields=fields), static,
            ATTRS, grid, datetime(2026, 7, 20))
    np.testing.assert_array_equal(captured["soilctop"], static["SOILCTOP"])
    assert np.all(captured["landusef"][16] == np.float32(.1))
    assert np.all(captured["landusef"][20] == 0)
    assert captured["lake_depth"] is static["LAKE_DEPTH"]
    assert captured["lakemask"] is static["LAKEMASK"]
    assert captured["lake_depth_flag"] == 1
