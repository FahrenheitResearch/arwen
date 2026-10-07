"""The radial-velocity operator evaluated only where the batch observes.

assimilate_radar_grid used to evaluate every member's Vr over the whole
domain for every radar and then crop to the radar's window, which cost a
member cube per radar (0.7 GB each on the recent 241 x 241 x 49 case, held
alive by the cropped view).  It now evaluates at the batch's own observed
points.  The breakage these tests prevent: an observed-points operator that
reads a different point, beam or wind than the whole-domain one, which
would change H(x) and so the analysis.  The comparison is bitwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.da.letkf import LetkfConfig, Localization, analyze
from gpuwm.da.obs_radar import (beam_unit_vectors, letkf_grid_geometry,
                                observed_radial_velocity,
                                radar_grid_to_gridded_obs,
                                simulated_radial_velocity)
from gpuwm.obs.radar_grid import read_radar_grid

from test_obs_radar_grid_v2 import PLACED, _grid, _merged, _params, _write


def _winds(rng, members, shape):
    return [tuple(rng.standard_normal(shape) for _ in range(3))
            for _ in range(members)]


def _whole_domain(document, winds):
    def simulated(index, radar):
        beam = beam_unit_vectors(document, index)
        return np.stack([simulated_radial_velocity(*w, beam) for w in winds])
    return simulated


def _observed_points(document, winds):
    def simulated(index, radar):
        return observed_radial_velocity(document, index, winds)
    return simulated


@pytest.mark.parametrize("dense", [False, True], ids=["v2-windowed", "v1-dense"])
def test_observed_points_operator_is_the_whole_domain_operator_bitwise(
        tmp_path, monkeypatch, dense):
    grid, params = _grid(), _params()
    observations = _merged(grid, params, dense=dense, monkeypatch=monkeypatch)
    _write(tmp_path / "radar.nc", observations, grid, params)
    document = read_radar_grid(tmp_path / "radar.nc")
    rng = np.random.default_rng(11)
    shape = (grid.nz, grid.ny, grid.nx)
    winds = _winds(rng, 4, shape)

    old, _ = radar_grid_to_gridded_obs(
        document, expected_grid=grid,
        velocity_simulated=_whole_domain(document, winds))
    new, _ = radar_grid_to_gridded_obs(
        document, expected_grid=grid,
        velocity_simulated=_observed_points(document, winds))
    assert len(new) == len(PLACED)
    observed = 0
    for a, b in zip(old, new):
        assert a.name == b.name and a.window == b.window
        np.testing.assert_array_equal(a.mask, b.mask)
        mask = np.asarray(a.mask, bool)
        observed += int(mask.sum())
        np.testing.assert_array_equal(np.asarray(a.simulated)[:, mask],
                                      np.asarray(b.simulated)[:, mask])
        assert not np.any(np.asarray(b.simulated)[:, ~mask])
    assert observed > 0

    prior = {"u": rng.standard_normal((4,) + shape)}
    cfg = LetkfConfig(Localization(3.0 * float(grid.dx_m), 1500.0), ("u",),
                      0.0, chunk_points=64)
    geometry = letkf_grid_geometry(grid)
    np.testing.assert_array_equal(analyze(prior, new, geometry, cfg)["u"],
                                  analyze(prior, old, geometry, cfg)["u"])
