"""Actual dycore, acoustic-window, CFL and graph transport differentials."""
from dataclasses import replace

import numpy as np
import pytest
from types import SimpleNamespace as NS

cp = pytest.importorskip("cupy")


@pytest.mark.parametrize("graphs", [False, "require"])
@pytest.mark.parametrize("varying", [False, True])
@pytest.mark.parametrize("recording", [False, True])
def test_live_step_and_domain_cfl_match_resident_across_tiles(graphs, varying, recording):
    from gpuwm.core import dycore
    from tilestream import driver, gather, harness, physics_inventory

    cfg = harness.make_config(96, 80, 12, dt=1., dx=1000., dy=1000., grid_id=21)
    tiled_cfg = replace(cfg, grid_id=22)
    resident = harness.make_state(cfg)
    initial = harness.make_state(tiled_cfg)
    # Localized feature away from the last tile; its CFL cannot be represented
    # by reading that last tile's three RK stages.
    for state in (resident, initial):
        state.w[2:8, 25:35, 25:35] = 30.
    store = {k: gather.pinned_copy(v) for k, v in gather.inventory(initial).items()}
    run = driver.TiledRun(
        store, tiled_cfg, 24, 20, 22, 2,
        scalars=physics_inventory.carrier_scalars(initial), use_graph=graphs)
    dycore.reset_wrf_cfl_recording()
    if recording:
        dycore.enable_wrf_cfl_recording()
    try:
        sequence = ((1., 4), (2., 6), (3., 8)) if varying else ((1., 4),) * 3
        for index, (dt, sounds) in enumerate(sequence):
            live = replace(cfg, dt=dt, time_step_sound=sounds)
            dycore.step(resident, live)
            run.sweep(1, live_config=replace(live, grid_id=22))
            run.drain()
            for name, value in gather.inventory(resident).items():
                np.testing.assert_array_equal(cp.asnumpy(value), store[name], err_msg=name)
            if recording:
                assert dycore.take_wrf_cfl(22) == dycore.take_wrf_cfl(21)
                # One row per domain step, all three stages, no duplicated halos.
                assert dycore._WRF_CFL_CALLS[22] == 3 * (index + 1)
                left = cp.asnumpy(dycore._WRF_CFL_STAT[21][index])
                right = cp.asnumpy(dycore._WRF_CFL_STAT[22][index])
                np.testing.assert_array_equal(left, right)
                assert int(right[2]) == 3 * (cfg.nz - 1) * cfg.ny * cfg.nx
            assert all(tile.elapsed_seconds == resident.elapsed_seconds for tile in run.tiles)
        assert run.cfg.dt == sequence[-1][0]
        assert run.tile_cfg.time_step_sound == sequence[-1][1]
        if graphs:
            assert run.graph_steppers is not None
            assert all(g.captures == 3 for g in run.graph_steppers)
    finally:
        dycore.reset_wrf_cfl_recording()


def test_cancelled_fold_is_not_a_step_and_halo_values_are_excluded():
    from gpuwm.core import dycore
    from tilestream import harness
    cfg = harness.make_config(8, 4, 4, dt=1., dx=1., dy=1., grid_id=29)
    state = NS(mup=cp.zeros((4, 8), dtype=cp.float32),
               mub2d=cp.ones((4, 8), dtype=cp.float32),
               c1f=cp.ones(5, dtype=cp.float32), c2f=cp.zeros(5, dtype=cp.float32),
               rdnw=cp.ones(4, dtype=cp.float32),
               u=cp.full((4, 4, 9), 2., dtype=cp.float32),
               v=cp.full((4, 5, 8), 3., dtype=cp.float32),
               msfu=cp.ones((4, 9), dtype=cp.float32),
               msfv=cp.ones((5, 8), dtype=cp.float32))
    window = NS(i0=2, i1=6, ci0=0, j0=1, j1=3, cj0=0)
    ww = cp.full((5, 4, 8), 99., dtype=cp.float32)
    dycore.reset_wrf_cfl_recording()
    dycore.enable_wrf_cfl_recording()
    try:
        dycore.begin_wrf_cfl_domain_step(cfg)
        dycore.set_wrf_cfl_tile_window(cfg.grid_id, window)
        dycore.record_wrf_vertical_cfl(state, cfg, ww)
        dycore.finish_wrf_cfl_tile(cfg.grid_id)
        dycore.finish_wrf_cfl_domain_step(cfg.grid_id, commit=False)
        assert dycore.take_wrf_cfl(cfg.grid_id) == (0., 0.)
        ww[1:4, 1:3, 2:6] = .25
        dycore.begin_wrf_cfl_domain_step(cfg)
        dycore.set_wrf_cfl_tile_window(cfg.grid_id, window)
        for _ in range(3):
            dycore.record_wrf_vertical_cfl(state, cfg, ww)
        dycore.finish_wrf_cfl_tile(cfg.grid_id)
        dycore.finish_wrf_cfl_domain_step(cfg.grid_id)
        assert dycore.take_wrf_cfl(cfg.grid_id) == (.25, 3.)
        words = cp.asnumpy(dycore._WRF_CFL_STAT[cfg.grid_id][0])
        assert int(words[2]) == 3 * 3 * 8
        assert int(words[4:].sum()) == 3 * 3 * 8
    finally:
        dycore.reset_wrf_cfl_recording()


def test_graph_cadence_uses_the_actual_domain_epoch():
    from tilestream import harness
    from tilestream.graphcap import cadence_key
    cfg = harness.make_config(16, 16, 8, dt=3., ra_lw_physics=1,
                              ra_sw_physics=1, cu_physics=1)
    driver = NS(radt_minutes=1., cudt_minutes=1., stepbl=20,
                surface_enabled=False, radiation_due_override=None,
                cumulus_due_override=None)
    key = cadence_key(NS(physics=driver, elapsed_seconds=12.,
                         domain_start_offset=12.), cfg)
    assert key[1] is True and key[2] is True and key[4] is True
