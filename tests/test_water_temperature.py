"""The water-temperature assembly: one provider per body, never per cell.

Each test here fails on a revert of gpuwm/ingest/water_temperature.py's
guarantees, and the numbers in the docstrings come from the reproducing
case (Lake Erie, 1985-05-31 12Z, 3 km nest) measured with
tools/water_temperature_probe.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from gpuwm.ingest.soil import preprocess_noah_soil
from gpuwm.ingest.water_temperature import (
    DEFAULT_WATER_TEMPERATURE_POLICY,
    SOURCE_ANALYSIS,
    SOURCE_COMPONENT_SKIN,
    SOURCE_LAND,
    SOURCE_PER_CELL,
    WaterTemperatureStatics,
    assemble_for_route,
    assemble_water_temperature,
    label_surface_components,
    lake_mask_from_landuse,
    require_assembled_water_temperature,
    resolve_water_temperature_policy,
    validate_water_temperature_policy,
    water_temperature_advisory,
)


def _source_grid():
    """A 0.25 degree ERA5-shaped crop: a lake, an ocean, land between."""
    lat = np.arange(44.0, 40.99, -0.25)
    lon = np.arange(-84.0, -76.99, 0.25)
    sst = np.full((lat.size, lon.size), np.nan)
    sst[4:9, 6:14] = 280.0            # the lake's own analysis
    sst[:, 22:] = 295.0               # the ocean, 15 K warmer
    return lat, lon, sst


def _target_grid():
    lat, lon = np.meshgrid(np.linspace(43.2, 41.8, 60),
                           np.linspace(-82.4, -77.2, 130), indexing="ij")
    lake = (lat < 42.9) & (lat > 42.0) & (lon > -82.3) & (lon < -80.7)
    ocean = lon > -78.4
    land = np.ones(lat.shape, dtype=bool)
    land[lake] = False
    land[ocean] = False
    skin = np.where(lake, 289.0, np.where(ocean, 296.0, 291.0))
    return lat, lon, land, lake, ocean, skin


def test_a_lake_never_takes_ocean_water():
    """Component identity, not a search radius, is what keeps basins apart.

    The shipped selector had no notion of a body at all, and the operator
    chains behind it choose donors by distance.  Here the ocean is 15 K
    warmer than the lake's own analysis and three degrees away, so any
    radius wide enough to rescue a starved shoreline cell would show up as
    ocean water inside the lake.
    """
    lat, lon, sst = _source_grid()
    tlat, tlon, land, lake, ocean, skin = _target_grid()

    values, source, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=sst, source_lat=lat, source_lon=lon,
        target_lat=tlat, target_lon=tlon)

    assert values[lake].max() < 285.0, "ocean water reached the lake"
    assert np.allclose(values[ocean], 295.0, atol=1e-6)
    assert receipt["components"] == 2


def test_land_values_are_never_donors_for_water():
    """Only water-defined source cells may feed a water target.

    A source analysis is undefined over land.  Writing a land temperature
    into those cells must not change one water target; if it does, some
    stencil is reading land.
    """
    lat, lon, sst = _source_grid()
    tlat, tlon, land, lake, ocean, skin = _target_grid()
    poisoned = np.where(np.isfinite(sst), sst, 330.0)

    clean, _, _ = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=sst, source_lat=lat, source_lon=lon,
        target_lat=tlat, target_lon=tlon)
    dirty, _, _ = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=poisoned, source_lat=lat,
        source_lon=lon, target_lat=tlat, target_lon=tlon)

    water = ~land
    np.testing.assert_array_equal(clean[water], dirty[water])
    # ... and the analysis genuinely reached the water, so the equality
    # above is a statement about donors rather than about a fallback that
    # never read the source at all.
    assert not np.allclose(clean[water], skin[water])


def test_every_water_cell_carries_a_declared_provider():
    """WATER_TEMP_SOURCE is total over water, and land is marked land."""
    lat, lon, sst = _source_grid()
    tlat, tlon, land, lake, ocean, skin = _target_grid()

    values, source, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=sst, source_lat=lat, source_lon=lon,
        target_lat=tlat, target_lon=tlon)

    water = ~land
    assert not np.any(source[water] == SOURCE_LAND)
    assert np.all(source[land] == SOURCE_LAND)
    assert set(np.unique(source[water]).tolist()) <= {
        SOURCE_ANALYSIS, SOURCE_COMPONENT_SKIN}
    assert sum(receipt["per_provider"].values()) == int(water.sum())
    assert np.isfinite(values[water]).all()


def test_one_provider_per_body_so_no_seam_can_form_inside_a_lake():
    """The defect was the switch itself, so the switch has to be gone.

    Measured on the reproducing case with the shipped per-cell selector:
    46.6 % of the nest's water cells passed the SST validity test and the
    rest fell back to SKINTEMP, and the intra-lake adjacent-cell step
    reached P99 7.56 K, with 88 of the 97 steps above 1 K sitting exactly
    on that validity boundary.  Under one provider per body the same
    statistic is P99 0.12 K.
    """
    lat, lon, sst = _source_grid()
    tlat, tlon, land, lake, ocean, skin = _target_grid()
    # Starve the lake's western third of analysis, exactly the coastal
    # geometry that made the shipped selector flip provider mid-lake.
    starved = sst.copy()
    starved[:, :9] = np.nan

    values, source, _ = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=starved, source_lat=lat, source_lon=lon,
        target_lat=tlat, target_lon=tlon)

    labels, classes = label_surface_components(land, lake)
    for label in classes:
        body = labels == label
        assert len(set(source[body].tolist())) == 1, (
            "a body was painted by more than one provider")

    # The starved lake still keeps its own analysis, so this is coherence
    # ACROSS a provider decision, not the flat answer a total fallback
    # would also produce.
    assert np.all(source[lake] == SOURCE_ANALYSIS)
    assert values[lake].max() < 285.0

    body = lake
    steps = np.concatenate([
        np.abs(np.diff(values, axis=0))[body[:-1] & body[1:]],
        np.abs(np.diff(values, axis=1))[body[:, :-1] & body[:, 1:]]])
    assert steps.max() < 1.0


def test_a_lake_with_no_analysis_of_its_own_takes_the_component_fallback():
    """No global search: a donorless body takes its coherent skin field."""
    lat, lon, sst = _source_grid()
    tlat, tlon, land, lake, ocean, skin = _target_grid()
    no_lake_analysis = sst.copy()
    no_lake_analysis[4:9, 6:14] = np.nan

    values, source, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=no_lake_analysis, source_lat=lat,
        source_lon=lon, target_lat=tlat, target_lon=tlon)

    assert np.all(source[lake] == SOURCE_COMPONENT_SKIN)
    np.testing.assert_allclose(values[lake], skin[lake])
    assert receipt["components_on_skin"] == 1


def test_wrf_compat_is_the_shipped_expression_byte_for_byte():
    """The certification arm has to BE the old code, not resemble it."""
    tlat, tlon, land, lake, ocean, skin = _target_grid()
    rng = np.random.default_rng(20260810)
    mapped_sst = np.where(rng.random(skin.shape) < 0.5, 0.0,
                          280.0 + 10.0 * rng.random(skin.shape))

    values, source, receipt = assemble_water_temperature(
        mapped_sst=mapped_sst, mapped_skin=skin, target_land=land,
        target_lake=lake, policy="wrf_compat")

    expected = np.where(
        np.isfinite(mapped_sst) & (mapped_sst >= 170.0)
        & (mapped_sst <= 400.0), mapped_sst, skin)
    np.testing.assert_array_equal(values, expected)
    assert np.all(source[~land] == SOURCE_PER_CELL)
    assert receipt["policy"] == "wrf_compat"


def test_the_soil_seam_takes_the_assembled_field_and_refuses_a_broken_one():
    """preprocess_noah_soil publishes the assembled water temperature."""
    shape = (6, 7)
    land = np.zeros(shape, dtype=bool)
    land[:, :3] = True
    fields = {
        "LANDSEA": land.astype(np.float64),
        "SKINTEMP": np.full(shape, 291.0),
        "SST": np.zeros(shape),
        "ST000007": np.full(shape, 288.0),
        "ST007028": np.full(shape, 287.0),
        "ST028100": np.full(shape, 286.0),
        "ST100289": np.full(shape, 285.0),
        "SM000007": np.full(shape, 0.3),
        "SM007028": np.full(shape, 0.3),
        "SM028100": np.full(shape, 0.3),
        "SM100289": np.full(shape, 0.3),
        "TMN": np.full(shape, 285.0),
    }
    assembled = np.where(land, 291.0, 283.5)

    state = preprocess_noah_soil(
        fields, soil_type=np.full(shape, 3.0),
        landmask=land.astype(np.float64), water_temperature=assembled)
    np.testing.assert_allclose(np.asarray(state.tsk)[~land], 283.5)

    broken = assembled.copy()
    broken[~land] = 0.0
    with pytest.raises(ValueError, match="assembled water_temperature"):
        preprocess_noah_soil(
            fields, soil_type=np.full(shape, 3.0),
            landmask=land.astype(np.float64), water_temperature=broken)


def test_silence_selects_the_coherent_policy_and_a_declaration_wins():
    """Declaration doctrine: silence is a choice, and it is the safe one."""

    class Case:
        water_temperature_policy = None
        water_temperature_overlay = None

    case = Case()
    assert resolve_water_temperature_policy(case) == "era5_class_coherent"
    assert DEFAULT_WATER_TEMPERATURE_POLICY == "era5_class_coherent"

    case.water_temperature_overlay = "analysis.nc"
    assert resolve_water_temperature_policy(case) == "external_overlay"

    case.water_temperature_policy = "wrf_compat"
    assert resolve_water_temperature_policy(case) == "wrf_compat"

    with pytest.raises(ValueError, match="is not one of"):
        validate_water_temperature_policy("coherent-ish")


def test_the_advisory_names_the_policy_and_the_higher_accuracy_option():
    """Warn, never block, and say what would be better."""
    lat, lon, sst = _source_grid()
    tlat, tlon, land, lake, ocean, skin = _target_grid()
    _, _, receipt = assemble_water_temperature(
        mapped_sst=None, mapped_skin=skin, target_land=land,
        target_lake=lake, source_sst=sst, source_lat=lat, source_lon=lon,
        target_lat=tlat, target_lon=tlon)

    line = water_temperature_advisory(receipt)
    assert "era5_class_coherent" in line
    assert "water_temperature_overlay" in line
    assert str(receipt["water_cells"]) in line
    assert "\n" not in line

    quiet = dict(receipt, policy="wrf_compat")
    assert water_temperature_advisory(quiet) is None


# ---------------------------------------------------------------------------
# The seam, not the route: a source added later cannot reintroduce the fuse
# ---------------------------------------------------------------------------
def _soil_fields(shape, land):
    """The smallest complete ERA5-shaped soil mapping, SST included."""
    return {
        "LANDSEA": land.astype(np.float64),
        "SKINTEMP": np.full(shape, 291.0),
        "SST": np.where(land, 0.0, 283.5),
        "ST000007": np.full(shape, 288.0),
        "ST007028": np.full(shape, 287.0),
        "ST028100": np.full(shape, 286.0),
        "ST100289": np.full(shape, 285.0),
        "SM000007": np.full(shape, 0.3),
        "SM007028": np.full(shape, 0.3),
        "SM028100": np.full(shape, 0.3),
        "SM100289": np.full(shape, 0.3),
        "TMN": np.full(shape, 285.0),
    }


def test_a_new_forcing_route_that_skips_the_assembly_is_refused_by_name():
    """The future-proofing bar, and the reason this lives at the ROUTER.

    Closing this defect route by route is what would guarantee its return
    with forcing source five.  So the refusal sits at the one seam every
    route already crosses.  This synthetic caller is that fifth source: it
    hands the router a raw SST beside its SKINTEMP, forwards no assembled
    field, and declares nothing.  It must be refused, by its own name, and
    must NOT quietly get the historical per-cell fuse.
    """
    shape = (6, 7)
    land = np.zeros(shape, dtype=bool)
    land[:, :3] = True
    fields = _soil_fields(shape, land)

    with pytest.raises(ValueError) as caught:
        preprocess_noah_soil(
            fields, soil_type=np.full(shape, 3.0),
            landmask=land.astype(np.float64),
            route="the imaginary source-five route")
    message = str(caught.value)
    assert "the imaginary source-five route" in message
    assert "assemble_for_route" in message
    assert "wrf_compat" in message

    # ... and an unnamed caller is still refused, just less helpfully.
    with pytest.raises(ValueError, match="an unnamed forcing route"):
        preprocess_noah_soil(
            fields, soil_type=np.full(shape, 3.0),
            landmask=land.astype(np.float64))


def test_the_declared_historical_selector_is_the_one_way_through():
    """``wrf_compat`` reopens the fuse; nothing else does."""
    shape = (6, 7)
    land = np.zeros(shape, dtype=bool)
    land[:, :3] = True
    fields = _soil_fields(shape, land)

    state = preprocess_noah_soil(
        fields, soil_type=np.full(shape, 3.0),
        landmask=land.astype(np.float64),
        route="a stock-WRF certification",
        water_temperature_policy="wrf_compat")
    # The fuse took SST on water, exactly as the shipped selector did.
    np.testing.assert_allclose(np.asarray(state.tsk)[~land], 283.5)

    for policy in ("era5_class_coherent", "external_overlay", None):
        with pytest.raises(ValueError, match="raw SST"):
            preprocess_noah_soil(
                fields, soil_type=np.full(shape, 3.0),
                landmask=land.astype(np.float64),
                route="a route with no decision",
                water_temperature_policy=policy)


def test_a_mapping_without_an_sst_is_the_identity_and_passes():
    """Why the bar is "a raw SST beside SKINTEMP" and not "any water".

    Absent an SST the selector reduces to SKINTEMP on every water cell,
    which is exactly what the assembly returns for a body with no donors.
    The two are the same array, so refusing those routes would buy nothing
    and cost a connected-components pass per domain.  This test is the
    proof of that equality, so the narrower bar stays honest.
    """
    shape = (6, 7)
    land = np.zeros(shape, dtype=bool)
    land[:, :3] = True
    fields = _soil_fields(shape, land)
    del fields["SST"]

    through = preprocess_noah_soil(
        fields, soil_type=np.full(shape, 3.0),
        landmask=land.astype(np.float64),
        route="a skin-only source")

    statics = WaterTemperatureStatics(
        route="a skin-only source", land=land,
        lake=np.zeros(shape, dtype=bool), lake_category=21,
        policy="era5_class_coherent")
    assembly = assemble_for_route(
        statics, mapped_sst=None, mapped_skin=fields["SKINTEMP"])
    assembled = preprocess_noah_soil(
        fields, soil_type=np.full(shape, 3.0),
        landmask=land.astype(np.float64),
        route="a skin-only source", water_temperature=assembly.values)
    np.testing.assert_array_equal(
        np.asarray(through.tsk), np.asarray(assembled.tsk))


def test_the_guard_is_reachable_on_its_own_terms():
    """The predicate, without a soil column around it."""
    fields = {"SST": np.zeros((2, 2)), "SKINTEMP": np.zeros((2, 2))}
    with pytest.raises(ValueError, match="route four"):
        require_assembled_water_temperature(
            route="route four", fields=fields, water_temperature=None,
            policy=None)
    # Assembled, declared, or SST-less: all three pass.
    require_assembled_water_temperature(
        route="route four", fields=fields,
        water_temperature=np.zeros((2, 2)), policy=None)
    require_assembled_water_temperature(
        route="route four", fields=fields, water_temperature=None,
        policy="wrf_compat")
    require_assembled_water_temperature(
        route="route four", fields={"SKINTEMP": np.zeros((2, 2))},
        water_temperature=None, policy=None)


# ---------------------------------------------------------------------------
# The lake category is the land-use table's, never a constant
# ---------------------------------------------------------------------------
def test_the_lake_category_comes_from_islake_and_a_table_without_one_is_loud():
    """MODIS names 21; USGS 24-category names no lake class at all.

    A hard-coded 21 reads USGS's LU_INDEX and finds no lakes, which is
    indistinguishable from a domain that genuinely has none.  ISLAKE below
    1 is WPS's way of saying the table has no inland-water class, and the
    mask that comes back is DECLARED absent rather than silently empty.
    """
    lu = np.array([[21, 1], [17, 21]])
    modis = {"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISWATER": 17,
             "ISLAKE": 21, "ISICE": 15}
    mask, category = lake_mask_from_landuse(lu, modis, route="a route")
    assert category == 21
    np.testing.assert_array_equal(
        mask, np.array([[True, False], [False, True]]))

    usgs = {"MMINLU": "USGS", "ISWATER": 16, "ISLAKE": -1, "ISICE": 24}
    mask, category = lake_mask_from_landuse(lu, usgs, route="a route")
    assert category is None
    assert not mask.any()

    # Not resolving the table at all is a refusal, not an empty mask.
    with pytest.raises(ValueError, match="ISLAKE"):
        lake_mask_from_landuse(lu, {"MMINLU": "USGS"}, route="a route")
    with pytest.raises(ValueError, match="a route"):
        lake_mask_from_landuse(lu, None, route="a route")


def test_an_absent_lake_class_is_stated_in_the_advisory():
    """Declared and loud, never silently all-False: the user-facing half."""
    shape = (4, 6)
    land = np.zeros(shape, dtype=bool)
    land[:, :2] = True
    lu = np.full(shape, 17)

    for attrs, expected in (({"ISLAKE": 21}, False), ({"ISLAKE": -1}, True)):
        statics = WaterTemperatureStatics.for_route(
            route="a route", policy=None,
            landmask=land.astype(np.float64), lu_index=lu,
            landuse_attrs=attrs)
        assembly = assemble_for_route(
            statics, mapped_sst=None, mapped_skin=np.full(shape, 288.0))
        advisory = water_temperature_advisory(assembly.receipt)
        assert ("names no lake category" in advisory) is expected
        assert "a route" in advisory


def test_the_lake_override_routes_put_their_lakes_on_the_water_side():
    """GFS hands the soil router a lake_mask/lake_skin pair, which makes
    every land-use lake cell water INSIDE the router.  The assembly has to
    agree, or the router would read a land value on exactly those cells."""
    shape = (3, 4)
    landmask = np.ones(shape)
    landmask[0, 0] = 0.0
    lu = np.full(shape, 1)
    lu[2, 3] = 21
    attrs = {"ISLAKE": 21}

    plain = WaterTemperatureStatics.for_route(
        route="a route", policy=None, landmask=landmask, lu_index=lu,
        landuse_attrs=attrs)
    assert plain.land[2, 3]

    override = WaterTemperatureStatics.for_route(
        route="a route", policy=None, landmask=landmask, lu_index=lu,
        landuse_attrs=attrs, lake_override=True)
    assert not override.land[2, 3]


def test_a_lake_override_route_forwards_without_moving_a_number():
    """The GFS ordering property, pinned.

    Forwarding an assembled field on a lake-override route is only safe if
    the assembly saw the SAME water surface the router will.  Assemble on
    the mapping's landmask instead, which calls GEOG lakes land because they
    are smaller than a source cell, and every lake would take a LAND skin
    value through `tsk = where(terrestrial | sea_ice, skin, water_temperature)`
    -- a silent regression exactly where this lane was trying to help.

    So the route builds its statics with ``lake_override=True`` and
    substitutes the nearest-source-water lake skin before assembling.  The
    result has to be bit-identical to what the route published before it
    forwarded anything, and that is what this asserts.
    """
    shape = (5, 6)
    landmask = np.ones(shape)
    landmask[:, 4:] = 0.0                      # ocean on the right
    lu = np.full(shape, 1)
    lu[1:3, 1:3] = 21                          # a lake inside the land
    lake = lu == 21
    attrs = {"ISLAKE": 21}

    # SKINTEMP as the GFS mapping produces it: lakes were interpolated as
    # LAND, so they carry a land value here and only the override knows
    # better.
    skin = np.where(landmask >= 0.5, 291.0, 288.0)
    lake_skin = np.where(lake, 279.5, np.nan)

    fields = {
        "LANDSEA": landmask.copy(),
        "SKINTEMP": skin,
        "GFS_ST000010": np.full(shape, 288.0),
        "GFS_ST010040": np.full(shape, 287.0),
        "GFS_ST040100": np.full(shape, 286.0),
        "GFS_ST100200": np.full(shape, 285.0),
        "GFS_SM000010": np.full(shape, 0.3),
        "GFS_SM010040": np.full(shape, 0.3),
        "GFS_SM040100": np.full(shape, 0.3),
        "GFS_SM100200": np.full(shape, 0.3),
        "TMN": np.full(shape, 285.0),
    }
    soil_type = np.full(shape, 3.0)

    before = preprocess_noah_soil(
        fields, soil_type=soil_type, landmask=landmask,
        lake_mask=lake, lake_skin_temperature=lake_skin,
        route="the GFS direct route")

    statics = WaterTemperatureStatics.for_route(
        route="the GFS direct route", policy=None, landmask=landmask,
        lu_index=lu, landuse_attrs=attrs, lake_override=True)
    assembly_skin = skin.copy()
    usable = lake & np.isfinite(lake_skin)
    assembly_skin[usable] = lake_skin[usable]
    assembly = assemble_for_route(
        statics, mapped_sst=None, mapped_skin=assembly_skin)

    after = preprocess_noah_soil(
        fields, soil_type=soil_type, landmask=landmask,
        lake_mask=lake, lake_skin_temperature=lake_skin,
        water_temperature=assembly.values, route="the GFS direct route")

    np.testing.assert_array_equal(
        np.asarray(after.tsk), np.asarray(before.tsk),
        err_msg="forwarding the assembly moved TSK on a lake-override route")
    # ... and the lake really is on the water side of both, so the equality
    # above is not two land values agreeing.
    np.testing.assert_allclose(np.asarray(after.tsk)[lake], 279.5)
    np.testing.assert_allclose(np.asarray(after.tsk)[landmask < 0.5], 288.0)

    # The trap this guards: statics built WITHOUT the override class the
    # lake as land, and the router then reads a land value there.
    naive = WaterTemperatureStatics.for_route(
        route="the GFS direct route", policy=None, landmask=landmask,
        lu_index=lu, landuse_attrs=attrs)
    naive_assembly = assemble_for_route(
        naive, mapped_sst=None, mapped_skin=skin)
    wrong = preprocess_noah_soil(
        fields, soil_type=soil_type, landmask=landmask,
        lake_mask=lake, lake_skin_temperature=lake_skin,
        water_temperature=naive_assembly.values,
        route="the GFS direct route")
    assert not np.allclose(np.asarray(wrong.tsk)[lake], 279.5)


# ---------------------------------------------------------------------------
# Component labels are invariant statics, so they are computed once
# ---------------------------------------------------------------------------
def test_component_labels_are_memoized_on_the_statics_they_come_from():
    """Launch-to-first-plot: the labels are a pure function of two masks,
    and the assembly runs once per forcing time.  Recomputing the union-find
    for every boundary time cost 0.19 s at 550 squared and 0.61 s at 1000
    squared, for an answer that cannot have changed."""
    from gpuwm.ingest import water_temperature as wt

    land = np.zeros((40, 40), dtype=bool)
    land[:, :10] = True
    lake = np.zeros((40, 40), dtype=bool)
    lake[20:25, 20:25] = True

    wt._LABEL_CACHE.clear()
    first_labels, first_classes = label_surface_components(land, lake)
    assert len(wt._LABEL_CACHE) == 1
    # A fresh, EQUAL pair of masks hits: the key is content, not identity,
    # so a caller that rebuilds its statics each time still reuses this.
    again_labels, again_classes = label_surface_components(
        land.copy(), lake.copy())
    assert again_labels is first_labels
    assert again_classes is first_classes
    assert len(wt._LABEL_CACHE) == 1

    # Different statics are a different answer and a different entry.
    other = lake.copy()
    other[30:32, 30:32] = True
    other_labels, _ = label_surface_components(land, other)
    assert other_labels is not first_labels
    assert len(wt._LABEL_CACHE) == 2

    # Shared, therefore never writable by a caller.
    assert not first_labels.flags.writeable
    with pytest.raises(ValueError):
        first_labels[0, 0] = 7


# ---------------------------------------------------------------------------
# Receipt implies consumption, on every route at once
# ---------------------------------------------------------------------------
#: The three spellings of the seam every forcing route crosses.
_SOIL_ROUTERS = ("preprocess_land_surface_soil", "preprocess_ruc_soil",
                 "preprocess_noah_soil")
#: What "this module assembled a water temperature" looks like in source.
_ASSEMBLES = ("assemble_for_route", "water_temperature_statics")


def _soil_router_calls(tree):
    """Every call to a soil router in one parsed module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(
            node.func, "attr", None)
        if name in _SOIL_ROUTERS:
            yield name, node


def test_a_route_that_assembles_must_be_the_route_that_consumes():
    """The false-receipt gate, stated once for every route there will be.

    The worst version of this defect is not a wrong number, it is a right
    one that nobody used: the GFS route triggered the assembly, printed the
    era5_class_coherent advisory, and then called soil preprocessing WITHOUT
    the assembled field, so the per-cell fuse ran behind a receipt saying it
    had not.  A reader had no way to tell.

    So the bar is structural rather than per route: a module that pays for
    an assembly must forward the result into every soil-router call it
    makes, and must name itself while doing it.  Reverting the forwarding on
    ANY route turns this red, including routes that do not exist yet.
    """
    root = Path(__file__).resolve().parents[1] / "gpuwm"
    checked = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if not any(marker in text for marker in _ASSEMBLES):
            continue
        calls = list(_soil_router_calls(ast.parse(text)))
        if not calls:
            continue
        relative = path.relative_to(root.parent).as_posix()
        checked.append(relative)
        for name, node in calls:
            keywords = {keyword.arg for keyword in node.keywords}
            assert "water_temperature" in keywords, (
                f"{relative}:{node.lineno} assembles a water temperature "
                f"somewhere in this module but calls {name} without "
                "forwarding it; that is the shape of a receipt describing "
                "a field nobody consumed")
            assert "route" in keywords, (
                f"{relative}:{node.lineno} calls {name} without naming its "
                "route, so a missing assembly could not be attributed")

    # The enumeration itself, so a route that quietly stops assembling is
    # visible here rather than silently dropping out of the gate above.
    assert checked == [
        "gpuwm/era5_direct.py",
        "gpuwm/gfs_direct.py",
        "gpuwm/ingest/nest_init.py",
        "gpuwm/mapped_direct.py",
        "gpuwm/runtime.py",
    ], checked


def test_every_soil_router_caller_in_the_tree_is_accounted_for():
    """The other half: every caller in the shipped package, not just the
    assembling ones.

    Scope is the ``gpuwm`` package, because that is what a forecast runs;
    ``tools/`` has three callers of its own and they are diagnostics, not
    routes.  A caller that does not appear in the gate above is one that
    hands the router no assembled field, which is allowed only where the
    mapping carries no SST beside its SKINTEMP, because there the selector
    and the assembly are the same array.  Those callers are listed below
    with that reason, so a new one has to be added deliberately.
    """
    root = Path(__file__).resolve().parents[1] / "gpuwm"
    callers = set()
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if list(_soil_router_calls(ast.parse(text))):
            callers.add(path.relative_to(root.parent).as_posix())
    assert callers == {
        # Assemble and consume (the gate above).
        "gpuwm/era5_direct.py",
        "gpuwm/gfs_direct.py",
        "gpuwm/ingest/nest_init.py",
        "gpuwm/mapped_direct.py",
        "gpuwm/runtime.py",
        # Native HRRR: the mapping carries SKINTEMP and no SST, so the
        # historical selector reduces to SKINTEMP on every water cell,
        # which is what the assembly returns for a donorless body.  The
        # router lets these through for that reason and no other; add an
        # SST to the HRRR inventory and it refuses until they assemble.
        "gpuwm/hrrr_hierarchy_direct.py",
        "gpuwm/ingest/hrrr_physics.py",
        # The router's own internal dispatch.
        "gpuwm/ingest/ruc_soil.py",
    }, sorted(callers)
