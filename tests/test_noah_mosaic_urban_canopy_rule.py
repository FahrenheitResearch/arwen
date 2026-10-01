"""Where Noah mosaic runs the urban canopy: WRF's rule by default, the town
rule by name (``mosaic_urban_canopy``), on the CPU.

WRF v4.7.1 runs the single-layer canopy inside every urban tile of
lsm_mosaic (module_sf_noahdrv.F:3733-4057) and blends it with ONE grid
fraction, FRC_URB2D, which urban_var_init sets to zero wherever the cell's
dominant category is not urban (module_sf_urban.F:2811-2821).  So the
dominant-urban rule lives entirely in the fraction the initialization leaves.

The proof of the default path is the ``ucm_wrfinit`` and ``ucm_lcz_wrfinit``
oracle families (tools/noah_mosaic_wrf471_oracle/run_mosaic_ucm.F90, families
3 and 4): byte-unmodified WRF with the fraction urban_var_init wrote,
dominant-urban and urban-secondary cells side by side.  gpuwm's production
initialization reproduces those fractions word for word here, and the GPU
replay (tests/test_noah_mosaic_ucm_wrf471_parity.py) runs the tile loop on
them against every WRF output word.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.noah_mosaic import (MosaicCategories, every_tile_urban_fraction,
                                    load_mosaic_categories,
                                    mosaic_tile_utype_lookup)
from gpuwm.core.urban_state import urban_var_init_host
from gpuwm.core.urban_tables import load_urban_params, urban_category_set
from gpuwm.verify.noah_mosaic_oracle import load, wrf_to_gpuwm

ROOT = Path(__file__).resolve().parents[1] / "gpuwm/data/noah_mosaic/oracle"
WRF_INIT_FAMILIES = ("ucm_wrfinit", "ucm_lcz_wrfinit")
MODIS = load_mosaic_categories("MODIFIED_IGBP_MODIS_NOAH", isurban=13,
                               iswater=17, isice=15)


def _first_step(family):
    steps = sorted(load(ROOT / family).values(), key=lambda f: int(f["itimestep"]))
    return steps[0]


def _production_init(fixture):
    """gpuwm's urban_var_init on the fixture's dominant categories, as the
    physics driver runs it before the door attaches the tiles.

    The LCZ family's only dominant class is LCZ 1, and WRF's physics_init
    stops such a domain after urban_var_init ("USING URBPARM_LCZ.TBL WITH
    OLD 3 URBAN CLASSES", module_physics_init.F:3349-3356), which gpuwm
    raises first.  The oracle calls urban_var_init directly, so it has the
    fractions anyway.  One extra LCZ 4 column, cut off again below, lifts
    the domain past that check; every fraction is set cell by cell, so the
    fixture's own cells are unchanged by it.
    """
    lcz = int(fixture["use_wudapt_lcz"])
    ivgtyp = wrf_to_gpuwm(fixture["ivgtyp_in"]).astype(np.int32)
    ny, nx = ivgtyp.shape
    if lcz:
        ivgtyp = np.concatenate([ivgtyp, np.full((ny, 1), 54, np.int32)], axis=1)
    shape = ivgtyp.shape
    host = urban_var_init_host(
        option=1, use_wudapt_lcz=lcz, params=load_urban_params(1, lcz),
        categories=urban_category_set("MODIFIED_IGBP_MODIS_NOAH", isurban=13),
        ivgtyp=ivgtyp, tsk=np.full(shape, 290.0, np.float32),
        tslb=np.full((4, *shape), 288.0, np.float32),
        tmn=np.full(shape, 285.0, np.float32),
        smois=np.full((4, *shape), 0.3, np.float32))
    return {name: np.ascontiguousarray(host[name][..., :nx])
            for name in ("frc_urb2d", "utype_urb2d")}


def _tiles(fixture):
    cat = np.ascontiguousarray(
        np.transpose(fixture["mosaic_cat_index_in"][:, :3, :], (1, 2, 0)))
    weight = np.ascontiguousarray(
        np.transpose(fixture["landusef2_in"][:, :3, :], (1, 2, 0)))
    return cat, weight


@pytest.mark.parametrize("family", WRF_INIT_FAMILIES)
def test_the_default_rule_is_wrfs_initialized_fraction_word_for_word(family):
    fixture = _first_step(family)
    host = _production_init(fixture)
    wrf_frc = wrf_to_gpuwm(fixture["frc_urb2d_in"]).astype(np.float32)
    wrf_utype = wrf_to_gpuwm(fixture["utype_urb2d_in"]).astype(np.int32)
    assert host["frc_urb2d"].view(np.uint32).tobytes() == wrf_frc.view(np.uint32).tobytes()
    assert np.array_equal(host["utype_urb2d"], wrf_utype)
    # The family really holds both kinds of cell, and WRF zeroed the town
    # ones: the rule is exercised, not vacuous.
    cat, weight = _tiles(fixture)
    lookup = mosaic_tile_utype_lookup(MODIS, int(fixture["use_wudapt_lcz"]))
    has_urban_tile = ((lookup[cat] > 0) & (weight > 0)).any(axis=0)
    town = has_urban_tile & (wrf_utype == 0)
    assert town.sum() >= 12 and (wrf_utype > 0).sum() >= 12
    assert np.all(wrf_frc[town] == 0.0) and np.all(wrf_frc[wrf_utype > 0] > 0.0)


@pytest.mark.parametrize("family", WRF_INIT_FAMILIES)
def test_the_town_rule_moves_only_town_cells_to_their_tiles_table_fraction(family):
    fixture = _first_step(family)
    lcz = int(fixture["use_wudapt_lcz"])
    host = _production_init(fixture)
    cat, weight = _tiles(fixture)
    lookup = mosaic_tile_utype_lookup(MODIS, lcz)
    table = load_urban_params(1, lcz).FRC_URB_TBL
    land = np.ones(cat.shape[1:], bool)
    town_frc = every_tile_urban_fraction(
        host["frc_urb2d"], utype_urb2d=host["utype_urb2d"], mosaic_cat_index=cat,
        landusef2=weight, utype_lookup=lookup, frc_table=table, land=land)
    dominant = host["utype_urb2d"] > 0
    # WRF's cells keep WRF's words.
    assert town_frc[dominant].view(np.uint32).tobytes() == \
        host["frc_urb2d"][dominant].view(np.uint32).tobytes()
    # Independent of the vectorised fill: walk each cell's tiles in WRF's
    # descending order and take the first urban tile of positive weight.
    expected = host["frc_urb2d"].copy()
    for j, i in zip(*np.nonzero(~dominant)):
        for t in range(cat.shape[0]):
            utype = int(lookup[cat[t, j, i]])
            if utype and weight[t, j, i] > 0:
                expected[j, i] = table[utype - 1]
                break
    assert town_frc.view(np.uint32).tobytes() == expected.view(np.uint32).tobytes()
    assert np.count_nonzero(town_frc != host["frc_urb2d"]) >= 12


def _grid(categories, weights):
    cat = np.asarray(categories, np.int32)[:, None, :]
    weight = np.asarray(weights, np.float32)[:, None, :]
    return cat, weight


def test_the_fill_skips_water_dominant_urban_and_weightless_tiles():
    # Columns: 0 town tile 30%; 1 dominant urban (input fraction kept);
    # 2 urban tile at zero weight; 3 sea; 4 no urban tile; 5 two urban
    # tiles, the larger (LCZ 6, 25%) first; 6 town tile 5%.
    cat, weight = _grid(
        [[10, 13, 10, 17, 10, 56, 12],
         [13, 10, 12, 13, 12, 59, 10],
         [12, 12, 13, 10, 7, 10, 13]],
        [[.7, .6, .8, 1., .6, .5, .6],
         [.3, .4, .2, 0., .4, .25, .35],
         [0., 0., 0., 0., 0., .25, .05]])
    categories = MosaicCategories(13, 15, 17, 5, tuple(range(51, 62)))
    frc = np.array([[0., .42, 0., 0., 0., 0., 0.]], np.float32)
    utype = np.array([[0, 2, 0, 0, 0, 0, 0]], np.int32)
    land = np.array([[1, 1, 1, 0, 1, 1, 1]], bool)
    lookup = mosaic_tile_utype_lookup(categories, 1)
    table = np.linspace(0.05, 0.55, 11).astype(np.float32)
    out = every_tile_urban_fraction(frc, utype_urb2d=utype, mosaic_cat_index=cat,
                                    landusef2=weight, utype_lookup=lookup,
                                    frc_table=table, land=land)
    assert out.dtype == np.float32
    assert out[0, 0] == table[4]          # ISURBAN under the LCZ table: type 5
    assert out[0, 1] == np.float32(.42)   # WRF's dominant cell, input kept
    assert out[0, 2] == 0.0               # weightless urban tile: no canopy
    assert out[0, 3] == 0.0               # water: no land tile loop
    assert out[0, 4] == 0.0               # no urban tile at all
    assert out[0, 5] == table[5]          # LCZ 6 (cat 56) before LCZ 9
    assert out[0, 6] == table[4]          # a 5% town tile still runs
    assert np.array_equal(frc, np.array([[0., .42, 0., 0., 0., 0., 0.]], np.float32))


def test_a_town_type_past_the_table_is_refused_by_name():
    cat, weight = _grid([[10], [56]], [[.7], [.3]])
    categories = MosaicCategories(13, 15, 17, 5, tuple(range(51, 62)))
    with pytest.raises(ValueError, match="no row in the 3-row urban table.*use_wudapt_lcz = 1"):
        every_tile_urban_fraction(
            np.zeros((1, 1), np.float32), utype_urb2d=np.zeros((1, 1), np.int32),
            mosaic_cat_index=cat, landusef2=weight,
            utype_lookup=mosaic_tile_utype_lookup(categories, 0),
            frc_table=np.array([.5, .9, .95], np.float32), land=np.ones((1, 1), bool))


@pytest.mark.parametrize("lcz", [0, 1])
def test_the_tile_type_is_the_kernels_and_urban_var_inits(lcz):
    # noah_mosaic.cu: utype = ivgtyp == isurban ? (lcz ? 5 : 2) : 0, then
    # LCZ_n -> n; urban_var_init gives a dominant cell the same type.
    lookup = mosaic_tile_utype_lookup(MODIS, lcz)
    urban = urban_category_set("MODIFIED_IGBP_MODIS_NOAH", isurban=13).utype_lookup(lcz)
    for category in range(min(lookup.size, urban.size)):
        assert lookup[category] == urban[category], category
    assert lookup[13] == (5 if lcz else 2)
    assert [int(lookup[c]) for c in MODIS.lcz] == list(range(1, 12))
