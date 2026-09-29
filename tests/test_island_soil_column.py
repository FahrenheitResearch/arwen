"""Land the source holds no land for takes a soil column, not the fill.

An island in a source area of open sea: every soil search from its cells
ends without a source land cell, and WPS writes METGRID.TBL fill_missing
there, a 285 K column saturated at 1.0, which stock real.exe carries into
the forecast.  The skin temperature of such cells already comes from the
source's sea there; the soil column now does too, at every layer, with the
soil's own field capacity as its water, and the count rides the receipt.
On the 2026-09-27 world sweep's ICON tiles over atolls that was 130, 185
and 18 island cells on the 1 km grids and 140 and 4 on the 500 m nests.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.noah import load_tables
from gpuwm.ingest.horiz import interpolate_era5_to_lambert
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
import gpuwm.ingest.soil as soil_module
from gpuwm.ingest.soil import island_soil_columns

from test_skin_temperature_other_surface import (
    _LATITUDE,
    _LONGITUDE,
    _NumpyBackend,
    _grid,
    _snapshot,
)

_TEMPERATURES = ("ST000007", "ST007028", "ST028100", "ST100289")
_MOISTURES = ("SM000007", "SM007028", "SM028100", "SM100289")


def _sea_snapshot():
    """An open-sea source crop that still carries soil fields."""
    snapshot = _snapshot(np.zeros((_LATITUDE.size, _LONGITUDE.size)))
    shape = (_LATITUDE.size, _LONGITUDE.size)
    fields = dict(snapshot.fields)
    for name in _TEMPERATURES:
        fields[name] = np.full(shape, 301.0)
    for name in _MOISTURES:
        fields[name] = np.full(shape, 0.3)
    return type(snapshot)(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields=fields)


def _island(shape):
    island = np.zeros(shape, dtype=bool)
    island[14:16, 15:17] = True
    return island


def _mapped():
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    island = _island(shape)
    met = interpolate_era5_to_lambert(
        _sea_snapshot(), grid, target_landmask=island,
        backend=_NumpyBackend())
    return met, island


def test_the_mapping_names_the_island_cells_and_keeps_the_metgrid_fields():
    met, island = _mapped()
    np.testing.assert_array_equal(met.soil_no_source_land, island)
    for name in _TEMPERATURES:
        assert np.all(np.asarray(met.fields[name])[island] == 285.0)
        repairs = met.masked_field_repairs[name]
        assert repairs["no_source_land"] == 4
        assert repairs.get("fill", 0) == 0
    for name in _MOISTURES:
        assert np.all(np.asarray(met.fields[name])[island] == 1.0)
        assert met.masked_field_repairs[name]["no_source_land"] == 4


def test_a_source_with_land_names_no_cell():
    grid = _grid()
    shape = grid.latlon_mass()[0].shape
    snapshot = _sea_snapshot()
    fields = dict(snapshot.fields)
    landsea = np.zeros((_LATITUDE.size, _LONGITUDE.size))
    landsea[0, 0] = 1.0             # one land cell, far from the island
    fields["LANDSEA"] = landsea
    snapshot = type(snapshot)(
        valid_time=snapshot.valid_time, levels_hpa=snapshot.levels_hpa,
        latitude=snapshot.latitude, longitude=snapshot.longitude,
        fields=fields)
    met = interpolate_era5_to_lambert(
        snapshot, grid, target_landmask=_island(shape),
        backend=_NumpyBackend())
    assert met.soil_no_source_land is None
    island = _island(shape)
    assert np.all(np.asarray(met.fields["SM000007"])[island] == np.float32(0.3))


def _noah(met, island, soil_type, **extra):
    shape = island.shape
    return preprocess_land_surface_soil(
        met.fields, sf_surface_physics=2, soil_type=soil_type,
        deep_soil_temperature=np.full(shape, 296.0),
        landmask=island.astype(np.float64), **extra)


def test_the_noah_column_is_the_skin_temperature_at_field_capacity(capsys):
    met, island = _mapped()
    soil_type = np.full(island.shape, 14.0)        # water, off the island
    soil_type[island] = 1.0                        # sand
    soil_type[15, 16] = 14.0                       # a land cell with water soil
    refsmc = load_tables().refsmc

    before = _noah(met, island, soil_type)
    assert np.all(before.soil_moisture[:, island] == 1.0)
    assert np.all(before.soil_temperature[:, island] == 285.0)

    state = _noah(met, island, soil_type,
                  soil_no_source_land=met.soil_no_source_land)
    tsk = state.tsk[island]
    assert tsk.min() > 280.0
    np.testing.assert_allclose(
        state.soil_temperature[:, island], np.broadcast_to(tsk, (4, 4)),
        rtol=0, atol=1e-4)
    expected = np.where(soil_type[island] == 1.0, refsmc[0], refsmc[7])
    np.testing.assert_allclose(
        state.soil_moisture[:, island], np.broadcast_to(expected, (4, 4)),
        rtol=0, atol=1e-6)
    said = capsys.readouterr().err
    assert "island soil: 4 land cell(s)" in said
    assert "field capacity" in said


def test_the_ruc_levels_are_built_from_the_same_column():
    met, island = _mapped()
    soil_type = np.full(island.shape, 14.0)
    soil_type[island] = 1.0
    state = preprocess_land_surface_soil(
        met.fields, sf_surface_physics=3, num_soil_layers=9,
        soil_type=soil_type, deep_soil_temperature=np.full(island.shape, 296.0),
        landmask=island.astype(np.float64),
        soil_no_source_land=met.soil_no_source_land)
    moisture = np.asarray(state.soil_moisture)[:, island]
    temperature = np.asarray(state.soil_temperature)[:, island]
    assert np.all(moisture < 0.5)
    np.testing.assert_allclose(moisture[0], load_tables().refsmc[0],
                               rtol=0, atol=1e-5)
    assert np.all(temperature[:4] > 280.0) and np.all(temperature != 285.0)


def test_the_column_builder_keeps_lakes_and_other_land_as_they_are():
    met, island = _mapped()
    fields = met.fields
    same, receipt = island_soil_columns(
        fields, no_source_land=np.zeros(island.shape, dtype=bool),
        soil_type=np.ones(island.shape), landmask=island.astype(float))
    assert same is fields and receipt is None

    lake = np.zeros(island.shape, dtype=bool)
    lake[14, 15] = True
    patched, receipt = island_soil_columns(
        fields, no_source_land=met.soil_no_source_land,
        soil_type=np.ones(island.shape), landmask=island.astype(float),
        lake_mask=lake)
    assert receipt["cells"] == 3
    assert receipt["fields"] == sorted(_MOISTURES + _TEMPERATURES)
    assert np.asarray(patched["SM000007"])[14, 15] == 1.0
    assert np.asarray(patched["SM000007"])[15, 15] == np.float32(
        load_tables().refsmc[0])


def test_a_mask_of_another_shape_is_refused():
    met, island = _mapped()
    with pytest.raises(ValueError, match="differ in shape"):
        island_soil_columns(
            met.fields, no_source_land=np.ones((3, 3), dtype=bool),
            soil_type=np.ones((3, 3)), landmask=np.ones((3, 3)))


@pytest.fixture(autouse=True)
def _fresh_announcements(monkeypatch):
    monkeypatch.setattr(soil_module, "_ANNOUNCED_ISLAND_SOIL", set(),
                        raising=False)
