"""Coordinate diffusion keeps its original reference across reused tiles."""
import numpy as np
import pytest

from conftest import requires_gpu

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("km", [2, 4])
@pytest.mark.parametrize("mix", [False, True])
def test_coordinate_diffusion_reused_buffers_match_resident(km, mix):
    from gpuwm.core.streaming import prime_lazy_carriers, prepared_tile_state_factory
    from tilestream import driver, gather, harness, physics_inventory

    cfg = harness.make_config(
        64, 48, 12, dt=1.0, dx=12000.0, dy=12000.0, time_step_sound=4,
        km_opt=km, diff_opt=1, mix_full_fields=mix, terrain_opt=0,
        map_proj=0, moist=False, mp_physics=0, ra_physics=0, cu_physics=0,
        bl_pbl_physics=0, sf_sfclay_physics=0, sf_surface_physics=0,
        nwp_diagnostics=0)
    state, _ = harness.make_physics_state(cfg, 4242)
    prime_lazy_carriers(state, cfg)
    inventory = physics_inventory.carrier_inventory
    original = cp.asnumpy(state._scratch["diff1_theta_initial"])
    assert np.any(original != original[:, :1, :1])
    store = {name: gather.pinned_copy(cp.asnumpy(value))
             for name, value in inventory(state).items()}
    assert "scratch/diff1_theta_initial" in store
    kwargs = driver.geography_run_kwargs(cfg, state, host=True, warmup=0)
    kwargs["tile_state_factory"] = prepared_tile_state_factory(state, cfg)
    # Six tiles occupy two buffers, so every sweep reuses each buffer.
    run = driver.TiledRun(store, cfg, 32, 16, harness.halo_radius(cfg), 2,
                          periodic=True, **kwargs)
    try:
        for _ in range(3):
            harness.run_steps(state, cfg, 1)
            run.sweep(1)
            cp.cuda.runtime.deviceSynchronize()
            expected = inventory(state)
            assert set(store) == set(expected)
            for name, values in expected.items():
                actual, wanted = np.asarray(store[name]), cp.asnumpy(values)
                assert np.isfinite(actual).all(), name
                np.testing.assert_array_equal(actual.view("u4"), wanted.view("u4"),
                                              err_msg=name)
            np.testing.assert_array_equal(store["scratch/diff1_theta_initial"], original)
        # The reference is read by the perturbation operator after theta has
        # changed, rather than passing only because both fields stayed fixed.
        assert np.any(cp.asnumpy(state.thp) + cp.asnumpy(state.thb)[:, None, None]
                      - np.float32(300.0) != original)
    finally:
        run.close()
