"""One firebrand owner with bounded completed-atmosphere column windows."""
import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
def test_mapped_host_fire_fields_keep_exact_bytes_without_device_allocation():
    import cupy as cp
    from tilestream.hoststore import alloc_pinned_array
    from tilestream.sfire_spotting import mapped_host_array
    host = alloc_pinned_array((9, 13), np.float32)
    host[:] = np.arange(host.size, dtype=np.float32).reshape(host.shape)
    mapped = mapped_host_array(host)
    assert mapped.data.ptr != 0
    assert cp.asnumpy(mapped).tobytes() == host.tobytes()
    cp.copyto(mapped, cp.full_like(mapped, 17.5))
    cp.cuda.get_current_stream().synchronize()
    assert np.all(host == np.float32(17.5))


@requires_gpu
@pytest.mark.parametrize("host", [False, True])
def test_bounded_completed_columns_match_resident_particle_transport(host):
    import cupy as cp
    from gpuwm.core.sfire_spotting import (advect_particles, prepare_atmosphere,
                                          properties)
    from tilestream.sfire_spotting import (BoundedAtmosphereColumns, ATMOSPHERE_FIELDS,
                                          MOISTURE_FIELDS, VERTICAL_FIELDS)
    from tilestream.gather import pinned_copy
    from test_sfire_restart_coupled_gpu import _coupled_state
    state, cfg, _ = _coupled_state(cp, mapped=True)
    fields = prepare_atmosphere(state, cfg)
    sources = {name: getattr(state, name) for name in
               (*ATMOSPHERE_FIELDS, *MOISTURE_FIELDS, *VERTICAL_FIELDS)
               if getattr(state, name, None) is not None}
    if host:
        sources = {name: pinned_copy(value) for name, value in sources.items()}
    columns = BoundedAtmosphereColumns(sources, cfg, p_top=state.p_top,
                                      block_shape=(4, 4))
    coords = cp.array([[1.6, 4.9, 8.05, 12.99, 15.4],
                       [1.6, 4.05, 8.9, 10.2, 11.4],
                       [12., 25., 100., 150., 500.]], cp.float32)
    life = cp.array([0, 1, 2, 3, 4], cp.int32)
    prop = properties(cp.array([[10.]*5, [10.]*5, [900.]*5, [0.]*5], cp.float32))
    kwargs = dict(origin=(-3, -3), tile=(1, cfg.nx, 1, cfg.ny),
                  momentum_steps=1, land_height=.15)
    resident = advect_particles(coords, life, prop.copy(), fields, .25, **kwargs)
    streamed = columns(coords, life, prop.copy(), {"dt": .25}, .25, **kwargs)
    for got, expected in zip(streamed, resident):
        assert cp.asnumpy(got).tobytes() == cp.asnumpy(expected).tobytes()
    assert len(columns.windows) > 1
    assert max(columns.max_window_shape) <= 16
    assert all(x1-x0 <= 16 and y1-y0 <= 16 for x0, x1, y0, y1 in columns.windows)
    assert columns.max_window_bytes > 0


@requires_gpu
def test_public_fire_tiles_match_actual_coupled_single_tile():
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.io.restart import STATE_SERIALIZED_ATTRS
    from test_sfire_restart_coupled_gpu import _coupled_state
    resident, cfg, resident_driver = _coupled_state(cp)
    tiled, tiled_cfg, tiled_driver = _coupled_state(cp)
    tiles = ((1, 32, 1, 18), (33, 64, 1, 18),
             (1, 32, 19, 36), (33, 64, 19, 36))
    tiled_driver.fire = FireCoupler(tiled, tiled_cfg, tiled_driver.fields, fire_tiles=tiles)
    assert tiled_driver.fire.grid.tiles == tiles
    assert tiled_driver.fire.setup_identity()["geometry"]["fire_tiles"] == [list(t) for t in tiles]
    run_steps(resident, cfg, 3)
    run_steps(tiled, tiled_cfg, 3)
    for name in STATE_SERIALIZED_ATTRS:
        value = getattr(resident, name, None)
        if value is not None:
            assert cp.asnumpy(getattr(tiled, name)).tobytes() == cp.asnumpy(value).tobytes(), name
    for name, value in resident_driver.fire.arrays().items():
        assert cp.asnumpy(tiled_driver.fire.arrays()[name]).tobytes() == cp.asnumpy(value).tobytes(), name


@requires_gpu
@pytest.mark.parametrize("mode",["shadow","ring"])
def test_one_global_firebrand_owner_survives_actual_streamed_sweeps_and_disk(tmp_path,mode,record_property):
    import cupy as cp
    from dataclasses import replace
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.core.dycore import step
    from gpuwm.core.streaming import streamed_store_inventory
    from test_sfire_tilestream_gpu import _state,_stream,_assert_words
    from gpuwm.io.wrfout import _device_state_frame
    def initial():
        state,cfg,driver=_state(cp)
        cfg=replace(cfg,fs_array_maxsize=256,fs_firebrand_gen_lim=7,fs_firebrand_gen_dt=2,
            fs_firebrand_gen_levels=3,fs_firebrand_gen_mom3d_dt=1,
            fs_firebrand_gen_levrand=True,trackember=True)
        driver.fire=FireCoupler(state,cfg,driver.fields)
        return state,cfg,driver
    reference,cfg,_=initial()
    tiled,tiled_cfg,_=initial()
    stream=_stream(tiled,tiled_cfg,mode)
    stream._spotting_owner.columns.block_shape=(16,16)
    checked_words=0
    checked_arrays=0
    assert len({id(tile.physics.fire.spotting) for tile in stream.tiled_run.tiles})==1
    for number in range(8):
        due=(number+1)%4==0
        step(reference,cfg,fire_history_due=due)
        stream(tiled,tiled_cfg,fire_history_due=due)
        arrays=streamed_store_inventory()(reference,None)
        checked_words+=sum(value.nbytes//4 for value in arrays.values())
        checked_arrays+=len(arrays)
        _assert_words(cp,arrays,stream.store)
        native=reference.physics.fire.spotting.metadata()
        assert stream.scalars["fire_header"]["fire"]["spotting"]==native
        if number==3:
            checkpoint=stream.write_restart(tmp_path/"tiled-spotting-mid.npz",tiled_cfg).path
            stream.tiled_run.close()
            tiled,tiled_cfg,_=initial()
            stream=_stream(tiled,tiled_cfg,mode)
            stream._spotting_owner.columns.block_shape=(16,16)
            stream.restore_restart(checkpoint,tiled_cfg)
    assert int(reference.physics.fire.spotting.control[0].item())>0
    resident_frame=_device_state_frame(reference)
    streamed_frame=stream.history_fields()
    assert set(resident_frame)==set(streamed_frame)
    for name,value in resident_frame.items():
        assert cp.asnumpy(value).tobytes()==np.asarray(streamed_frame[name]).tobytes(),name
    assert max(stream._spotting_owner.columns.max_window_shape)<=28
    record_property("graded_32bit_words",checked_words)
    record_property("graded_arrays",checked_arrays)
    record_property("completed_steps",8)
    record_property("different_words",0)
    record_property("native_particle_steps",stream._spotting_owner.spotting.step_count)
    record_property("max_column_window",str(stream._spotting_owner.columns.max_window_shape))
