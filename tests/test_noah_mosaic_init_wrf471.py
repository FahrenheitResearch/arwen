"""CPU oracle gate for WRF v4.7.1 mosaic tile initialisation."""
from pathlib import Path
import numpy as np
import pytest

from gpuwm.core.noah_mosaic import (
    lsm_mosaic_init, _lsm_mosaic_init_reference, real_exe_landusef, attach_noah_mosaic,
    load_mosaic_categories, mosaic_array_shapes, landless_tile_cells,
)
from gpuwm.verify.noah_mosaic_oracle import load, wrf_to_gpuwm

ROOT = Path(__file__).resolve().parents[1] / "tests/data/oracles/noah_mosaic/mosaic_init"
INPUTS = ("ivgtyp xland xice tsk tslb smois sh2o snow snowc snowh canwat "
          "albedo albbck emiss embck znt").split()


#: The two category outputs the landless-cell definition changes.
LANDLESS_FIELDS = ("landusef2", "mosaic_cat_index", "landusef2_full",
                   "mosaic_cat_index_full")


def _assert_equal_outside_landless(actual, expected, landless, label):
    """Bitwise equality, the landless cells' first tile excepted."""
    actual = np.array(actual, copy=True)
    expected = np.array(expected, copy=True)
    assert actual.shape == expected.shape, label
    actual[0][landless] = 0
    expected[0][landless] = 0
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), label


def _assert_landless_definition(result, reference, ivgtyp, landless, label):
    """WRF (the reference and the fixture) leaves every tile weight zero on a
    landless tiled cell; the port gives the first tile the cell's own
    category at weight one and leaves the other tiles as WRF has them."""
    ref_f = np.asarray(reference["landusef2"])
    assert np.all(ref_f[:, landless] == 0), label
    assert np.all(result["landusef2"][0][landless] == np.float32(1)), label
    assert np.array_equal(result["mosaic_cat_index"][0][landless],
                          np.asarray(ivgtyp)[landless]), label
    assert np.array_equal(result["landusef2"][1:].view(np.uint32),
                          ref_f[1:].view(np.uint32)), label


@pytest.mark.parametrize("name,fixture", list(load(ROOT).items()))
def test_init_oracle(name, fixture):
    inputs = {n: wrf_to_gpuwm(fixture[n]) for n in INPUTS}
    result = lsm_mosaic_init(wrf_to_gpuwm(fixture["landusef"]), **inputs,
                             mosaic_cat=int(fixture["mosaic_cat"]),
                             fractional_seaice=bool(fixture["fractional_seaice"]),
                             iswater=17, isice=15, full=True)
    reference = _lsm_mosaic_init_reference(wrf_to_gpuwm(fixture["landusef"]), **inputs,
                             mosaic_cat=int(fixture["mosaic_cat"]),
                             fractional_seaice=bool(fixture["fractional_seaice"]),
                             iswater=17, isice=15, full=True)
    landless = landless_tile_cells(
        reference["landusef2"], xland=inputs["xland"], xice=inputs["xice"],
        fractional_seaice=bool(fixture["fractional_seaice"]))
    # The all-zero-LANDUSEF column of every fixture is exactly such a cell.
    assert landless.any(), name
    for field in result:
        if field in LANDLESS_FIELDS:
            _assert_equal_outside_landless(result[field], reference[field],
                                           landless, (name, field))
        else:
            assert np.array_equal(result[field].view(np.uint32), reference[field].view(np.uint32)), field
    _assert_landless_definition(result, reference, inputs["ivgtyp"], landless, name)
    outputs = {n for n in fixture if n.endswith("_mosaic") or n.endswith("_full")}
    assert outputs <= result.keys()
    for field in sorted(outputs):
        actual, expected = result[field], wrf_to_gpuwm(fixture[field])
        if field in LANDLESS_FIELDS:
            _assert_equal_outside_landless(actual, expected, landless, (name, field))
        else:
            assert actual.shape == expected.shape
            assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), (name, field)
    # CONTROL: WRF's own words on those cells are the zero weights the
    # reference transcribes, so the difference above is the definition,
    # not a transcription error.
    wrf_f = wrf_to_gpuwm(fixture["landusef2_full"])
    assert np.all(wrf_f[:int(fixture["mosaic_cat"])][:, landless] == 0), name


def test_real_edits():
    """Lake merge and the sea-ice one-hot; no surface_input_source=1 edit."""
    f = np.zeros((21, 1, 6), dtype=np.float32)
    f[0] = .5; f[16] = .5
    f[20, 0, 2] = .25
    mask = np.array([[0, 1, .5, 0, 0, 0]], dtype=np.float32)
    ice = np.array([[0, 0, 0, .5, .02, np.nextafter(np.float32(.02), np.float32(0))]])
    result = real_exe_landusef(f, landmask=mask, xice=ice, iswater=17,
                              islake=21, isice=15, fractional_seaice=False)
    # process_percent_cat_new's exact-50% fix belongs to
    # surface_input_source=1 only (module_initialize_real.F:3033-3050).
    for column in (0, 1):
        assert result[0, 0, column] == np.float32(.5)
        assert result[16, 0, column] == np.float32(.5)
    assert result[16, 0, 2] == np.float32(.75)
    assert result[20, 0, 2] == 0
    assert result[14, 0, 3] == 1 and np.sum(result[:, 0, 3]) == 1
    result = real_exe_landusef(f, landmask=mask, xice=ice, iswater=17,
                              islake=-1, isice=15, fractional_seaice=True)
    assert result[20, 0, 2] == np.float32(.25)
    assert result[16, 0, 2] == np.float32(.5)
    assert result[14, 0, 4] == 1 and np.sum(result[:, 0, 4]) == 1
    assert result[14, 0, 5] == 0


def test_shapes_categories_and_attach_refusals():
    categories = load_mosaic_categories("MODIFIED_IGBP_MODIS_NOAH", iswater=17, isice=15, isurban=13)
    assert categories.lcz == tuple(range(51, 62))
    fixture = next(iter(load(ROOT).values()))
    state = {n: wrf_to_gpuwm(fixture[n]) for n in INPUTS}
    f = wrf_to_gpuwm(fixture["landusef"])
    kw = dict(categories=categories, fractional_seaice=False, lucats=20)
    for mc in (0, 22):
        with pytest.raises(ValueError, match="LANDUSEF2"):
            attach_noah_mosaic(state.copy(), landusef=f, mosaic_cat=mc, **kw)
    with pytest.raises(ValueError, match="category indexing"):
        attach_noah_mosaic(state.copy(), landusef=f[0], mosaic_cat=3, **kw)
    invalid = np.zeros_like(f);invalid[-1] = 1
    with pytest.raises(ValueError, match="past its rows"):
        attach_noah_mosaic(state.copy(), landusef=invalid, mosaic_cat=3, **kw)
    # Use LUCATS=21 for this fixture's real lake category.
    kw["lucats"] = 21
    attach_noah_mosaic(state, landusef=f, mosaic_cat=3, **kw)
    for name, (shape, dtype) in mosaic_array_shapes(3, *state["tsk"].shape).items():
        assert state[name].shape == shape
        assert state[name].dtype == np.dtype(dtype)
    with pytest.raises(ValueError, match="double attach"):
        attach_noah_mosaic(state, landusef=f, mosaic_cat=3, **kw)


def test_stray_land_seaice():
    f = np.zeros((21, 1, 3), dtype=np.float32);f[6] = 1
    for fractional in (False, True):
        result = real_exe_landusef(f, landmask=np.array([[1, .5, 0]], np.float32),
                                  xice=np.ones((1, 3), np.float32), iswater=17,
                                  islake=-1, isice=15, fractional_seaice=fractional)
        assert np.array_equal(result[:, 0, 0].view(np.uint32), f[:, 0, 0].view(np.uint32))
        assert result[14, 0, 1] == 1 and result[14, 0, 2] == 1


def test_random_vectorized_reference():
    rng = np.random.default_rng(174)
    ny, nx, nlcat = 5, 600, 21
    f = rng.choice(np.array([0., -0., .1, .25, 1/3, .5, .7, 1.], np.float32),
                   size=(nlcat, ny, nx))
    state = dict(ivgtyp=rng.choice([7, 15, 17], size=(ny, nx)).astype(np.int32),
                 xland=rng.choice([1., 2.], size=(ny, nx)).astype(np.float32),
                 xice=rng.choice([0., .02, .5, 1.], size=(ny, nx)).astype(np.float32))
    for n in ("tsk snow snowc snowh canwat albedo albbck emiss embck znt").split():
        state[n] = rng.random((ny, nx), dtype=np.float32)
    for n in ("tslb", "smois", "sh2o"):
        state[n] = rng.random((4, ny, nx), dtype=np.float32)
    for mc in range(1, nlcat+1):
        for fractional in (False, True):
            kw = dict(state, mosaic_cat=mc, iswater=17, isice=15,
                      fractional_seaice=fractional, full=True)
            actual, expected = lsm_mosaic_init(f, **kw), _lsm_mosaic_init_reference(f, **kw)
            landless = landless_tile_cells(
                expected["landusef2"], xland=state["xland"], xice=state["xice"],
                fractional_seaice=fractional)
            for n in actual:
                if n in LANDLESS_FIELDS:
                    _assert_equal_outside_landless(actual[n], expected[n], landless,
                                                   (mc, fractional, n))
                else:
                    assert np.array_equal(actual[n].view(np.uint32), expected[n].view(np.uint32)), (mc, fractional, n)
            _assert_landless_definition(actual, expected, state["ivgtyp"], landless,
                                        (mc, fractional))
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN LANDUSEF has no category order"):
        lsm_mosaic_init(f, **kw)
