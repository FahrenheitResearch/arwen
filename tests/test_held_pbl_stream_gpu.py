"""Measured resident/tiled continuation with held, spatially varying PBL rates."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from test_gf_pbl_forcing_lanes import _driver_for

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, requires_gpu]


def _snapshot(state):
    from tilestream.physics_inventory import carrier_inventory
    return {key: cp.asnumpy(value) for key, value in carrier_inventory(state).items()}


def _pinned(arrays, *, poison=False):
    from tilestream.gather import pinned_copy
    return {key: pinned_copy(np.full_like(value, np.nan if value.dtype.kind == "f" else -1) if poison else value.copy())
            for key, value in arrays.items()}


def _equal(left, right):
    assert set(left) == set(right)
    bad = [key for key in left if not np.array_equal(left[key], right[key])]
    assert not bad, bad


def _builder(cfg, seed=0):
    return _driver_for(cfg)


def _stream(store, cfg, scalars, nsteps):
    from tilestream import driver, harness
    kwargs = driver.physics_run_kwargs(cfg, None, builder=_builder, warmup=0)
    kwargs["scalars"] = scalars
    report = {}
    driver.run_tiled(store, cfg, 16, 20, halo=harness.halo_radius(cfg),
                     nsteps=nsteps, nbuffers=2, report=report, **kwargs)
    cp.cuda.runtime.deviceSynchronize()
    return report


@pytest.mark.parametrize("cu", [3, 16])
def test_between_pbl_calls_all_checkpoint_transports_match_uninterrupted(tmp_path, cu):
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.io import restart
    from tilestream import checkpoint, harness, restart_stream
    from tilestream.physics_inventory import carrier_scalars

    cfg = validate_run_config(RunConfig(
        nx=48, ny=40, nz=40, dx=12000., dy=12000., ztop=18000.,
        dt=20., time_step_sound=4, run_seconds=0., moist=True, mp_physics=10,
        bl_pbl_physics=1, sf_sfclay_physics=91, bldt=2.,
        cu_physics=cu, cudt_minutes=0.))
    assert cfg.bldt * 60 / cfg.dt == 6
    state, physics = _builder(cfg)
    # The real PBL producer makes a different forcing at every horizontal
    # location. Reusing a tile buffer without gathering these lanes then
    # feeds the preceding tile's values, not merely uniform zeros.
    j, i = cp.indices((cfg.ny, cfg.nx), dtype=cp.float32)
    physics.fields["tsk"][...] += i * cp.float32(.02) + j * cp.float32(.03)
    harness.run_steps(state, cfg, 2)
    assert physics.call_counts["ysu"] == 1
    assert physics.call_counts["cumulus"] == 2
    for name in restart.DRIVER_HELD_FORCING_ATTRS:
        held = cp.asnumpy(getattr(physics, name))
        assert np.isfinite(held).all() and np.max(np.abs(held)) > 0
        assert np.any(held != held[:, :1, :1])
    start = _snapshot(state)
    start_scalars = carrier_scalars(state)
    setup = checkpoint.DomainSetup.capture(state, cfg)
    stream_setup = restart_stream.capture_domain_setup(state)
    path = restart.write_restart(tmp_path / "resident.npz", state, cfg)

    harness.run_steps(state, cfg, 2)
    reference = _snapshot(state)
    assert physics.call_counts["ysu"] == 1
    assert physics.call_counts["cumulus"] == 4

    resident, _ = _builder(cfg)
    restart.restore_restart(path, resident, cfg)
    harness.run_steps(resident, cfg, 2)
    _equal(reference, _snapshot(resident))

    continuous = _pinned(start)
    continuous_scalars = dict(start_scalars)
    _stream(continuous, cfg, continuous_scalars, 2)
    _equal(reference, continuous)

    interrupted = _pinned(start)
    clock = dict(start_scalars)
    _stream(interrupted, cfg, clock, 1)
    template, _ = _builder(harness.tile_config(cfg, 48, 52))
    store_path = checkpoint.write_store_restart(
        tmp_path / "store.npz", interrupted, clock, setup, cfg)
    stream_path = tmp_path / "stream.npz"
    restart_stream.write_streamed_restart(
        stream_path, interrupted, cfg, scalars=clock,
        setup=stream_setup, template_state=template)
    for transport, checkpoint_path in [("store", store_path), ("stream", stream_path)]:
        restored = _pinned(start, poison=True)
        if transport == "store":
            restored_clock = checkpoint.read_store_restart(
                checkpoint_path, restored, setup, cfg)
        else:
            restored_clock = {}
            restart_stream.read_streamed_restart(
                checkpoint_path, restored, cfg, setup=stream_setup,
                template_state=template, scalars=restored_clock)
        _equal(interrupted, restored)
        _stream(restored, cfg, restored_clock, 1)
        _equal(reference, restored)
        assert restored_clock == continuous_scalars

    # Recreate the historical loss while keeping every other carrier and
    # the clock exact. The comparison must change a forecast field; merely
    # noticing the two injected zeros in the manifest is no evidence.
    withheld = _pinned(start)
    for name in restart.DRIVER_HELD_FORCING_ATTRS:
        withheld[f"held/{name}"].fill(0)
    _stream(withheld, cfg, dict(start_scalars), 2)
    changed = [key for key in reference if key.startswith("state/")
               and not np.array_equal(reference[key], withheld[key])]
    assert changed, "held forcing reached no forecast consumer"
    print(f"cu={cu}: six tiles/two reused buffers, resident + both streamed "
          f"checkpoint transports exact; withheld rates change {changed}")
