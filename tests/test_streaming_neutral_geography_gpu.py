"""Actual prepared tile buffers step both flat and terrain-enabled domains."""
import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("terrain", [0, 1])
def test_prepared_neutral_buffers_match_resident_steps(terrain):
    from gpuwm.core.streaming import prepared_tile_state_factory
    from tilestream import driver, gather, harness, physics_inventory

    cfg = harness.make_config(64, 64, 12, dt=1., dx=12000., dy=12000.,
        time_step_sound=4, terrain_opt=terrain, map_proj=1,
        moist=True, mp_physics=6, ra_physics=0, cu_physics=0,
        bl_pbl_physics=0, sf_sfclay_physics=0, sf_surface_physics=0)
    geo = harness.make_geography(cfg, terrain=bool(terrain))
    state, _ = harness.make_physics_state(cfg, 4242, geography=geo)
    factory = prepared_tile_state_factory(state, cfg)
    inventory = physics_inventory.carrier_inventory
    store = {name: gather.pinned_copy(cp.asnumpy(value))
             for name, value in inventory(state).items()}
    kwargs = driver.geography_run_kwargs(cfg, state, host=True, warmup=0)
    kwargs["tile_state_factory"] = factory
    run = driver.TiledRun(store, cfg, 32, 32, harness.halo_radius(cfg), 2,
                          periodic=True, **kwargs)
    for _ in range(2):
        harness.run_steps(state, cfg, 1)
        run.sweep(1)
        cp.cuda.runtime.deviceSynchronize()
        expected = inventory(state)
        assert set(store) == set(expected)
        for name, values in expected.items():
            actual, wanted = np.asarray(store[name]), cp.asnumpy(values)
            assert np.isfinite(actual).all(), name
            np.testing.assert_array_equal(actual, wanted, err_msg=name)
