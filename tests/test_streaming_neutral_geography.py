"""Neutral tile geography obeys the requested base-profile dimensions."""
import numpy as np
import pytest

from gpuwm.core.grid import make_base_state, make_vertical_coord
from tilestream import harness


@pytest.mark.parametrize("terrain", [0, 1])
@pytest.mark.parametrize("placeholder_height", [0., 123.])
def test_neutral_base_shape_matches_configured_terrain(terrain, placeholder_height):
    cfg = harness.make_config(12, 10, 8, terrain_opt=terrain)
    geo = harness.neutral_geography(cfg, terrain_height=placeholder_height)
    assert geo.msft.shape == (10, 12)
    assert geo.msfu.shape == (10, 13)
    assert geo.msfv.shape == (11, 12)
    if terrain:
        np.testing.assert_array_equal(geo.terrain, np.full((10, 12), placeholder_height))
    else:
        assert geo.terrain is None
    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac)
    base = make_base_state(coord, lambda z: np.full_like(z, 300.),
                           p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=geo.terrain)
    expected_rank = 3 if terrain else 1
    for name in ("thb", "pb", "alb", "phb"):
        assert getattr(base, name).ndim == expected_rank
