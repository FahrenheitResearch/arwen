"""Coupled fire crosses atmospheric tile seams and continues through disk."""
import numpy as np
import pytest

from conftest import requires_gpu


def _state(cp):
    from test_restart import _physics_state
    result = _physics_state(cp, nx=96, ny=72, nz=16, dx=90.0, dy=120.0, dt=0.25,
        mp_physics=1, cu_physics=0, ra_physics=0, ifire=2, sr_x=4, sr_y=3,
        fire_fuel_read=0, fire_fuel_cat=1, fire_fmc_read=0, fire_boundary_guard=2,
        fmoist_run=True, fmoist_interp=True, fmoist_freq=1,
        fire_num_ignitions=1, fire_ignition_start_x1=2880.0,
        fire_ignition_start_y1=2880.0, fire_ignition_radius1=160.0,
        fire_ignition_ros1=280.0)
    from gpuwm.core.streaming import prime_lazy_carriers
    from gpuwm.core.sfire_coupler import FireCoupler
    state,cfg,physics=result
    physics.fire=FireCoupler(state,cfg,physics.fields,{
        "LFN_HIST":cp.full(physics.fire.fine_shape,-17.5,cp.float32),
        "LFN_TIME":cp.asarray([-125.5],cp.float32)})
    prime_lazy_carriers(result[0], result[1])
    return result


def _stream(state, cfg, mode):
    from gpuwm.core import streaming as st
    from tilestream import driver
    decision = st.StreamingDecision(True, "SFIRE seam proof", 32, 24, 2, 16,
                                    store="host", write_mode=mode)
    return st.attach(state, cfg, decision,
        tile_state_factory=st.prepared_tile_state_factory(state, cfg),
        geography=driver.geography_store(state, host=True),
        inventory_fn=st.streamed_store_inventory(), observe_stability=False)


def _assert_words(cp, arrays, store):
    assert set(arrays) == set(store)
    for name, device in arrays.items():
        actual = np.asarray(store[name])
        expected = cp.asnumpy(device)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), (
            name, int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))))


@requires_gpu
@pytest.mark.parametrize("mode", ["shadow", "ring"])
def test_coupled_fire_refined_host_tiles_and_restart_are_word_exact(tmp_path, mode):
    import cupy as cp
    from gpuwm.core.dycore import step
    from gpuwm.core.streaming import streamed_store_inventory
    from gpuwm.io.wrfout import _device_state_frame
    reference, cfg, _ = _state(cp)
    tiled, tiled_cfg, _ = _state(cp)
    stream = _stream(tiled, tiled_cfg, mode)
    for number in range(8):
        step(reference, cfg)
        stream(tiled, tiled_cfg)
        _assert_words(cp, streamed_store_inventory()(reference, None), stream.store)
        if number == 3:
            checkpoint = tmp_path / "tiled-mid.npz"
            stream.write_restart(checkpoint, tiled_cfg)
            tiled, tiled_cfg, _ = _state(cp)
            stream = _stream(tiled, tiled_cfg, mode)
            stream.restore_restart(checkpoint, tiled_cfg)
    fire = reference.physics.fire
    assert bool(cp.any(fire.grid.data["fire_area"][1:73, 128:132] > 0))
    resident_frame = _device_state_frame(reference)
    tiled_frame = stream.history_fields()
    assert set(resident_frame) == set(tiled_frame)
    for name, expected in resident_frame.items():
        value = cp.asnumpy(expected)
        assert np.array_equal(value.view(np.uint32), tiled_frame[name].view(np.uint32)), name
    native = fire.metadata()["grid"]
    stored = stream.scalars["fire_header"]["fire"]["grid"]
    for name in ("time_seconds", "step_count", "moisture_lasttime", "moisture_nexttime",
                 "last_cfl_bound", "last_cfl_exceeded", "last_ignited_counts"):
        assert stored[name] == native[name], name
