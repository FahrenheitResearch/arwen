"""A water cell no provider can give a temperature is filled and counted.

A route can hand the assembly a lake whose skin temperature is a fill
value, zero, where its source holds no water near the lake; the assembly
refused the whole preparation for it ("N water cells have no admissible
water temperature").  Every source reaches the same assembly, so each
case below runs through every provider the assembly chooses between: skin
only, an SST analysis that does not reach the lake, and a lake model that
declines every lake cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.ingest import water_temperature as wt
from gpuwm.ingest.water_temperature import (
    WaterTemperatureStatics,
    assemble_for_route,
    assemble_water_temperature,
    water_temperature_advisory,
)

LAKE = 21
SHAPE = (12, 16)


def _land_with_lakes():
    """Land everywhere but two small lakes, the skin a gradient over land."""
    land = np.ones(SHAPE, dtype=bool)
    land[3:5, 3:5] = False          # a 2x2 lake
    land[8, 9:12] = False           # a 1x3 lake
    rows, cols = np.indices(SHAPE)
    skin = 285.0 + 0.5 * rows + 0.25 * cols
    # The masked chain's fill value where no source water reached a lake.
    skin[~land] = 0.0
    return land, skin


def _statics(land):
    lu = np.where(land, 10, LAKE)
    return WaterTemperatureStatics.for_route(
        route="a regional source crop", policy=None,
        landmask=land.astype(np.float64), lu_index=lu,
        landuse_attrs={"ISLAKE": LAKE})


def _assemble(provider, land, skin):
    """The one assembly, reached the way each kind of source reaches it."""
    statics = _statics(land)
    rows, cols = np.indices(SHAPE)
    if provider == "skin":
        return assemble_for_route(statics, mapped_sst=None, mapped_skin=skin)
    if provider == "analysis":
        # An SST analysis with no value anywhere near the lakes.
        source_sst = np.full(SHAPE, np.nan)
        return assemble_for_route(
            statics, mapped_sst=np.zeros(SHAPE), mapped_skin=skin,
            source_sst=source_sst,
            source_lat=np.arange(SHAPE[0], dtype=np.float64),
            source_lon=np.arange(SHAPE[1], dtype=np.float64),
            target_lat=rows.astype(np.float64),
            target_lon=cols.astype(np.float64))
    # A lake model that declined every lake cell (frozen or unknown ice).
    return assemble_for_route(
        statics, mapped_sst=None, mapped_skin=skin,
        mapped_lake_water=np.full(SHAPE, np.nan))


@pytest.mark.parametrize("provider", ["skin", "analysis", "lake_model"])
def test_a_lake_with_no_source_water_near_takes_the_skin_around_it(provider):
    land, skin = _land_with_lakes()
    water = ~land

    assembly = _assemble(provider, land, skin)

    values, receipt = assembly.values, assembly.receipt
    assert np.isfinite(values[water]).all()
    assert (values[water] >= 170.0).all() and (values[water] <= 400.0).all()
    assert (assembly.provider[water] == wt.SOURCE_SURROUNDING_SKIN).all()
    # Each lake lies inside the range of the skin on its own shore, so the
    # fill came from around it and from nowhere else.
    for lake in (np.s_[3:5, 3:5], np.s_[8, 9:12]):
        shore = np.zeros(SHAPE, dtype=bool)
        body = np.zeros(SHAPE, dtype=bool)
        body[lake] = True
        rows, cols = np.nonzero(body)
        shore[max(rows.min() - 1, 0):rows.max() + 2,
              max(cols.min() - 1, 0):cols.max() + 2] = True
        shore &= land
        assert skin[shore].min() <= values[body].min()
        assert values[body].max() <= skin[shore].max()
    # Land is untouched: the soil router still owns it.
    np.testing.assert_array_equal(values[land], skin[land])
    fill = receipt["water_fill"]
    assert fill["cells"] == int(water.sum()) == 7
    assert fill["surrounding_skin"] == 7
    assert fill["own_body"] == fill["nearest_water"] == 0
    assert fill["cell_indices"] == np.argwhere(water).tolist()
    assert receipt["per_provider"] == {"surrounding_skin": 7}
    advisory = water_temperature_advisory(receipt)
    assert "7 water cell(s) had no water temperature in the source" in advisory
    assert "7 from the skin temperature around them" in advisory
    assert "(3, 3)" in advisory


def test_a_body_with_part_of_its_water_fills_the_rest_from_it():
    land, skin = _land_with_lakes()
    skin[3, 3] = skin[3, 4] = 281.0        # half the 2x2 lake has water
    skin[8, 9:12] = 283.0                  # the other lake is whole

    assembly = _assemble("skin", land, skin)

    np.testing.assert_array_equal(assembly.values[3:5, 3:5], 281.0)
    assert (assembly.provider[4, 3:5] == wt.SOURCE_NEAREST_WATER).all()
    fill = assembly.receipt["water_fill"]
    assert (fill["cells"], fill["own_body"]) == (2, 2)
    assert fill["cell_indices"] == [[4, 3], [4, 4]]


def test_a_body_with_none_takes_the_nearest_source_water_as_one_value():
    land, skin = _land_with_lakes()
    skin[8, 9:12] = 283.0                  # the one lake the source reached
    land[0, 15] = False                    # a farther pond with other water
    skin[0, 15] = 299.0

    assembly = _assemble("skin", land, skin)

    np.testing.assert_array_equal(assembly.values[3:5, 3:5], 283.0)
    assert (assembly.provider[3:5, 3:5] == wt.SOURCE_NEAREST_WATER).all()
    fill = assembly.receipt["water_fill"]
    assert (fill["cells"], fill["nearest_water"]) == (4, 4)


def test_a_complete_assembly_carries_no_fill_record():
    land, skin = _land_with_lakes()
    skin[~land] = 283.0

    assembly = _assemble("skin", land, skin)

    assert "water_fill" not in assembly.receipt
    assert "had no water temperature" not in water_temperature_advisory(
        assembly.receipt)


def test_a_lake_whose_shore_has_no_skin_takes_the_nearest_admissible_one():
    land, skin = _land_with_lakes()
    skin[2:6, 2:6] = 0.0                   # the 2x2 lake and its whole shore
    skin[3:5, 3:5] = 0.0

    values, provider, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=~land)

    # Nearest admissible cells to the 2x2 lake's edge sit two rows or
    # columns out; the first in row-major order of the window is (1, 3).
    np.testing.assert_array_equal(values[3:5, 3:5], skin[1, 3])
    assert (provider[3:5, 3:5] == wt.SOURCE_SURROUNDING_SKIN).all()
    assert receipt["water_fill"]["surrounding_skin"] == 7


def test_only_a_domain_with_no_surface_temperature_at_all_refuses():
    land, skin = _land_with_lakes()
    skin[:] = 0.0

    with pytest.raises(ValueError, match="no cell of this domain carries"):
        _assemble("skin", land, skin)
