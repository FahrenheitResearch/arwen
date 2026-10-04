"""Actual physics rates, spectra and every carrier versus the ordinary run."""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _cupy():
    if os.environ.get("GPUWM_NO_LOCAL_GPU"):
        pytest.skip("local GPU use is forbidden")
    return pytest.importorskip("cupy")


def _binding(cfg, *, skebs):
    from gpuwm.ensemble.stochastic import StochasticConfig, StochasticTimestepHook
    from gpuwm.ensemble.stochastic_execution import StochasticPhysicsBinding
    hook = StochasticTimestepHook((cfg.ny + 1, cfg.nx + 1), dx=cfg.dx, dy=cfg.dy, dt=cfg.dt,
        member_seed=97531, sppt=StochasticConfig.wrf_reference("sppt"),
        skebs_psi=StochasticConfig.wrf_reference("skebs_psi") if skebs else None,
        skebs_theta=StochasticConfig.wrf_reference("skebs_theta") if skebs else None)
    return StochasticPhysicsBinding(hook, member_id=19, recipe_sha256="stream-identity-fixture")


@pytest.mark.parametrize("skebs", [None, False, True], ids=["off", "sppt", "sppt-skebs"])
def test_actual_resident_and_periodic_tiled_stochastic_physics_are_word_identical(skebs):
    cp = _cupy()
    from tilestream import driver, harness, physics_inventory as inventory
    from tilestream.test_gate import physics_cfg
    from gpuwm.ensemble.stochastic_streaming import attach_stochastic_sweep_lease
    cfg = physics_cfg("+Noah LSM", nx=96, ny=80, nz=12)
    ordinary, _ = inventory.default_builder(cfg, 31)
    original_binding = None if skebs is None else _binding(cfg, skebs=skebs)
    if original_binding is not None:
        ordinary._ensemble_stochastic = original_binding
    # Warm with the actual stochastic hook so spectra and physical carriers
    # describe the same absolute model step at the checkpoint seam.
    harness.run_steps(ordinary, cfg, 1)
    start = {name: value.copy() for name, value in inventory.carrier_inventory(ordinary).items()}
    start_scalars = dict(inventory.carrier_scalars(ordinary))
    stream_binding = None if skebs is None else _binding(cfg, skebs=skebs)
    if stream_binding is not None:
        stream_binding.restore(original_binding.snapshot())
    kwargs = driver.physics_run_kwargs(cfg, ordinary, seed=31, warmup=1)
    kwargs["scalars"] = start_scalars
    tiled = driver.TiledRun(start, cfg, tile_nx=48, tile_ny=40, halo=harness.halo_radius(cfg),
                            nbuffers=2, periodic=True, **kwargs)
    try:
        attach_stochastic_sweep_lease(tiled, stream_binding)
        for _ in range(3):
            harness.run_steps(ordinary, cfg, 1)
            tiled.sweep(1)
            tiled.drain()
            cp.cuda.get_current_stream().synchronize()
            expected = inventory.carrier_inventory(ordinary)
            assert set(expected) == set(tiled.store)
            differing = [name for name in expected
                         if cp.asnumpy(expected[name]).tobytes() != cp.asnumpy(tiled.store[name]).tobytes()]
            assert not differing, differing
            if stream_binding is None:
                assert original_binding is None
                continue
            assert stream_binding.applied_steps == original_binding.applied_steps
            for name in ("sppt", "skebs"):
                reference = original_binding.snapshot()["hook"][name]
                actual = stream_binding.snapshot()["hook"][name]
                if reference is None:
                    assert actual is None
                elif name == "sppt":
                    assert cp.asnumpy(reference["spectrum"]).tobytes() == cp.asnumpy(actual["spectrum"]).tobytes()
                else:
                    for component in ("psi", "theta"):
                        assert cp.asnumpy(reference[component]["spectrum"]).tobytes() == cp.asnumpy(actual[component]["spectrum"]).tobytes()
    finally:
        tiled.close()
