"""Sea-level land is refused only where it lies away from any shore.

The static check refused every domain whose land all sat at exactly 0 m,
to catch a terrain field that never arrived.  The 30 arc-second GMTED2010
in WPS GEOG holds 0 m on atolls and cays that really lie at sea level, so
an ocean domain whose only land is such islands was refused with its
terrain complete (eight GFS runs of the 2026-09-27 world sweep, on ocean
tiles whose only land is atolls, cays or islets).  Ground at sea level is
shoreline ground:
the check now refuses zero-height land only where some of it has land on
all four sides inside the grid, which a missing field over any landmass
still has and an atoll does not.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.native_wrf_contract as native_wrf_contract
from gpuwm.native_wrf_contract import (
    load_native_static_cache,
    validate_native_static_fields,
    write_native_static_cache,
)


def require_land_terrain(*args, **kwargs):
    return native_wrf_contract.require_land_terrain(*args, **kwargs)


def _grid(ny, nx):
    plane = np.ones((ny, nx), dtype=np.float64)
    return SimpleNamespace(
        mapfac_m=lambda: plane,
        mapfac_u=lambda: np.ones((ny, nx + 1)),
        mapfac_v=lambda: np.ones((ny + 1, nx)),
        coriolis_m=lambda: (plane, plane),
        rotation_m=lambda: (plane, plane),
    )


def _fields(landmask, height):
    landmask = np.asarray(landmask, dtype=np.float64)
    mass = landmask.shape
    water = landmask < 0.5
    landuse = np.zeros((21, *mass), dtype=np.float64)
    landuse[16][water] = 1.0          # category 17, water
    landuse[12][~water] = 1.0         # category 13, urban (any land class)
    soil = np.zeros((16, *mass), dtype=np.float64)
    soil[13][water] = 1.0             # category 14, water
    soil[0][~water] = 1.0             # category 1, sand
    return {
        "HGT_M": np.asarray(height, dtype=np.float64),
        "LANDMASK": landmask,
        "LU_INDEX": np.where(water, 17.0, 13.0),
        "SCT_DOM": np.where(water, 14.0, 1.0),
        "SCB_DOM": np.where(water, 14.0, 1.0),
        "SNOALB": np.zeros(mass),
        "SOILTEMP": np.full(mass, 299.0),
        "TMN": np.full(mass, 299.0),
        "GREENFRAC": np.full((12, *mass), 0.3),
        "LAI12M": np.full((12, *mass), 1.0),
        "ALBEDO12M": np.full((12, *mass), 15.0),
        "LANDUSEF": landuse,
        "SOILCTOP": soil,
        "SOILCBOT": soil.copy(),
    }


def _atoll(ny=24, nx=24):
    """A ring of one-cell-wide rim around a lagoon, plus a two-cell cay."""
    land = np.zeros((ny, nx))
    land[6, 6:14] = land[13, 6:14] = 1.0
    land[6:14, 6] = land[6:14, 13] = 1.0
    land[19, 3:5] = 1.0
    return land


def test_an_atoll_only_domain_at_sea_level_prepares_and_says_so_once(
        capsys, monkeypatch):
    monkeypatch.setattr(
        native_wrf_contract, "_ANNOUNCED_SHORELINE_LAND", set(),
        raising=False)
    land = _atoll()
    fields = _fields(land, np.zeros(land.shape))
    grid = _grid(*land.shape)

    exported = validate_native_static_fields(fields, grid, *land.shape)
    assert np.all(exported["HGT_M"] == 0.0)
    # The same grid checked again (at write and at load) says nothing more.
    validate_native_static_fields(fields, grid, *land.shape)
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("terrain:")]
    assert len(lines) == 1
    assert f"{int(land.sum())} land cell(s)" in lines[0]
    assert "shoreline" in lines[0] and "[static.highres]" in lines[0]


def test_an_island_three_cells_across_at_sea_level_is_refused_by_name():
    """A high island the terrain dataset does not carry.

    Measured on one of the sweep's tiles: the staged GMTED2010 holds no
    height anywhere on an island over 1000 m high, whose 93 land cells on
    the 1 km grid include 56 with land on all four sides.
    """
    land = np.zeros((20, 20))
    land[8:13, 7:12] = 1.0
    with pytest.raises(ValueError) as refusal:
        require_land_terrain(np.zeros(land.shape), land)
    text = str(refusal.value)
    assert "identically zero over every land cell" in text
    assert "9 of its 25 land cell(s) have land on all four sides" in text
    assert "[static.highres]" in text and 'fields = "terrain"' in text


def test_a_missing_terrain_field_over_a_landmass_is_still_refused(tmp_path):
    land = np.zeros((30, 40))
    land[:, 18:] = 1.0                 # a coast with land to the east
    fields = _fields(land, np.zeros(land.shape))
    with pytest.raises(ValueError, match="identically zero over every land"):
        validate_native_static_fields(fields, _grid(*land.shape), *land.shape)
    # A route that never validated cannot publish it for a forecast to
    # refuse later: the cache writer holds it to the same check.
    with pytest.raises(ValueError, match="identically zero over every land"):
        write_native_static_cache(tmp_path / "static.npz", fields)
    assert not (tmp_path / "static.npz").exists()


def test_the_check_reads_what_the_caller_defers_it_to():
    """A declared high-resolution terrain overlay replaces the baseline
    height, so the route validates the baseline with the check deferred
    and runs it on what the overlay makes of the fields."""
    land = np.zeros((20, 20))
    land[8:13, 7:12] = 1.0
    fields = _fields(land, np.zeros(land.shape))
    exported = validate_native_static_fields(
        fields, _grid(*land.shape), *land.shape, land_terrain=False)
    overlaid = exported["HGT_M"].copy()
    overlaid[land > 0.5] = 35.0
    require_land_terrain(overlaid, exported["LANDMASK"])


def test_an_edge_sliver_is_not_counted_as_inland_ground():
    """A cell on the grid's edge has an unknown neighbour outside it.

    Measured on one of the sweep's tiles: its only land is a string of
    low cays in two rows along the domain's southern edge, and counting
    the outside as land made one of them inland.
    """
    land = np.zeros((20, 20))
    land[0:2, 5:12] = 1.0
    require_land_terrain(np.zeros(land.shape), land)
    land[0:3, 5:12] = 1.0              # a third row: now some are inland
    with pytest.raises(ValueError, match="have land on all four sides"):
        require_land_terrain(np.zeros(land.shape), land)


def test_land_with_any_height_and_a_grid_with_no_land_pass_silently(capsys):
    land = np.zeros((20, 20))
    land[5:15, 5:15] = 1.0
    height = np.zeros(land.shape)
    height[9, 9] = 4.0
    require_land_terrain(height, land)
    require_land_terrain(np.zeros(land.shape), np.zeros(land.shape))
    assert "terrain:" not in capsys.readouterr().err


def test_an_atoll_cache_round_trips_through_write_and_load(tmp_path):
    land = _atoll(22, 26)
    fields = _fields(land, np.zeros(land.shape))
    grid = _grid(*land.shape)
    exported = validate_native_static_fields(fields, grid, *land.shape)
    path = tmp_path / "static.npz"
    write_native_static_cache(path, exported)
    loaded = load_native_static_cache(path, grid, *land.shape)
    np.testing.assert_array_equal(loaded["LANDMASK"], land)
