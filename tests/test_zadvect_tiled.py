"""IEVA must cross rank seams with live implicit flux and byte identity."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np

import pytest

from conftest import requires_gpu


@pytest.mark.parametrize("variant", ("wrf_471", "wrf_legacy"))
def test_implicit_option_reaches_every_tile_config(variant):
    # Losing this option in the config copy silently turns rank stepping
    # back into explicit advection while resident integration stays implicit.
    from tilestream.harness import tile_config
    from tilestream.ieva_gate import gate_config

    cfg = gate_config(variant=variant)
    tile = tile_config(cfg, 96, 84)
    assert tile == replace(cfg, nx=96, ny=84)
    assert tile.zadvect_implicit == 1
    assert tile.zadvect_implicit_variant == variant
    assert tile.specified and tile.moist and tile.hybrid_opt == 2


def test_experiment_grid_keeps_coordinate_and_projection(tmp_path):
    # The full-grid proof must not silently substitute the idealized eta
    # ladder or derive a different pressure top from its sounding height.
    from tilestream.ieva_gate import experiment_grid

    payload = b"""[projection]
map_proj = "lambert"
ref_lat = 38.5
ref_lon = -97.5
truelat1 = 38.5
truelat2 = 38.5
stand_lon = -97.5
[shared]
nz = 4
ztop = 20000.0
p_top = 5000.0
hybrid_opt = 2
etac = 0.2
eta_levels = [1.0, 0.85, 0.5, 0.2, 0.0]
[[domain]]
parent_id = 0
nx = 160
ny = 132
dx = 3000.0
time_step = 15.0
"""
    path = tmp_path / "experiment.toml"
    path.write_bytes(payload)
    cfg, projection, p_top, digest = experiment_grid(path, "wrf_legacy")
    assert (cfg.nx, cfg.ny, cfg.nz) == (160, 132, 4)
    assert cfg.eta_levels == (1.0, 0.85, 0.5, 0.2, 0.0)
    assert cfg.dt == 15.0 and p_top == 5000.0 and cfg.ztop == 20000.0
    assert projection["ref_lat"] == projection["truelat1"] == 38.5
    assert cfg.mp_physics == cfg.bl_pbl_physics == cfg.sf_surface_physics == 0
    assert digest == hashlib.sha256(payload).hexdigest()


@requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("variant", ("wrf_471", "wrf_legacy"))
def test_implicit_forced_tiled_dycore_matches_resident(variant):
    # A cold or low-Courant comparison never exercises the implicit solve.
    # The gate requires nonzero wwI in both dynamics and scalar transport,
    # all persisted bytes equal across 1/2/4 ranks, and live negative controls.
    from tilestream.ieva_gate import gate
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() >= 1
    result = gate(devices=(0,), variant=variant)
    assert result["reference"]["finite"]
    assert result["controls"]["poison_seam_matches"]
    assert result["controls"]["explicit_differs"]
    for row in result["runs"].values():
        assert row["match"], row.get("diff")
        assert row["finite"]
        for values in row["implicit_points"].values():
            assert values and max(values) > 0
    assert result["ok"]


@requires_gpu
@pytest.mark.slow
@pytest.mark.parametrize("variant", ("wrf_471", "wrf_legacy"))
def test_implicit_streamed_tiles_match_across_workers(variant):
    # The out-of-core full-grid proof must keep the same answer when two
    # independent workers share a device.  This covers slab initialization
    # and the host-store path that a resident rank test cannot exercise.
    from tilestream.ieva_gate import stream_gate
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() >= 1
    # Eight steps also cover the existing short-halo visibility floor.
    result = stream_gate(160, 132, 50, 8, devices=(0, 0),
                         tile_nx=64, tile_ny=48, variant=variant,
                         resident_reference=True)
    assert set(result["runs"]) == {"1", "2"}
    for row in result["runs"].values():
        assert row["match"] and row["finite"] and row["implicit_active"]
    assert result["ok"]


@requires_gpu
@pytest.mark.gpu
def test_stream_initial_state_is_independent_of_slab_height():
    # Per-slab random draws or local-coordinate phases produce artificial
    # seams.  Both row partitions must build the same global initial state.
    from tilestream.ieva_gate import _stream_start, gate_config
    import cupy as cp

    assert cp.cuda.runtime.getDeviceCount() >= 1
    cfg = replace(gate_config(64, 60, 8), specified=False)
    first, first_geo, first_setup = _stream_start(cfg, slab_rows=17)
    second, second_geo, second_setup = _stream_start(cfg, slab_rows=32)
    assert first_setup == second_setup
    for left, right in ((first, second), (first_geo, second_geo)):
        assert left.keys() == right.keys()
        for key in left:
            assert np.array_equal(left[key].view(np.uint8),
                                  right[key].view(np.uint8)), key
