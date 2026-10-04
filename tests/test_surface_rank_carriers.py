"""Surface persistence must be horizontally addressable by the rank copier."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.streaming import _surface_tile_initialization_inputs, StreamingRefused
from gpuwm.core.preflight import physics_array_shapes
from gpuwm.core.lake_schema import LAKE_STATE_WORDS, LAKE_STATIC_WORDS
from test_ruc_mosaic_options import _config
from tilestream.gather import classify


def test_fraction_buffer_uses_source_category_counts_and_preserves_source():
    source = dict(landusef=np.full((21, 3, 5), .125, np.float32),
                  soilctop=np.full((16, 3, 5), .0625, np.float32))
    driver = SimpleNamespace(fields=source,
                             ruc_params=SimpleNamespace(dataset_identifier="MODIFIED_IGBP_MODIS_NOAH"))
    cfg = _config(mosaic_lu=1, mosaic_soil=1)
    result = _surface_tile_initialization_inputs(driver, cfg)
    for name, categories in (("landusef", 21), ("soilctop", 16)):
        assert result[name].shape == (categories, cfg.ny, cfg.nx)
        np.testing.assert_array_equal(result[name].sum(axis=0), 1.)
    assert np.all(source["landusef"] == .125)
    assert result["landuse_dataset"] == driver.ruc_params.dataset_identifier


def test_rank_cannot_invent_missing_source_fractions():
    with pytest.raises(StreamingRefused, match="no prepared landusef carrier"):
        _surface_tile_initialization_inputs(SimpleNamespace(fields={}),
                                            _config(mosaic_lu=1))


def test_lake_restart_planes_fit_existing_horizontal_copier():
    cfg = _config(sf_lake_physics=1, mosaic_lu=1, mosaic_soil=1)
    shapes = physics_array_shapes(cfg, lake_columns=3)
    assert shapes["fields/lake_columns"] == (LAKE_STATE_WORDS, cfg.ny, cfg.nx)
    assert shapes["fields/lake_static"] == (LAKE_STATIC_WORDS, cfg.ny, cfg.nx)
    for name in ("lake_columns", "lake_static", "lake_latitude", "landusef", "soilctop"):
        assert classify(shapes["fields/" + name], cfg.nz, cfg.ny, cfg.nx,
                        layers_ok=True) == "mass"
    assert shapes["lake/forcing"][-1] == 3
    assert shapes["lake/gather_work"] == (LAKE_STATE_WORDS, 3)
