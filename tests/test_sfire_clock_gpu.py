"""Actual fractional nested clock images through coupled fire and disk."""
import json
import numpy as np

from conftest import requires_gpu


@requires_gpu
def test_real_fractional_clock_coupled_fire_disk_is_byte_exact(tmp_path):
    import cupy as cp
    from test_restart import _physics_state,_assert_restart_equal
    from test_sfire_clock import fractional_clock
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.state import refresh_model_time
    from gpuwm.io import restart
    dt=float(fractional_clock().dt_fp32)
    def make():
        return _physics_state(cp,nx=16,ny=12,nz=16,dx=90.0,dy=120.0,dt=dt,
            time_step_sound=32,
            mp_physics=1,cu_physics=0,ra_physics=0,ifire=2,sr_x=4,sr_y=3,
            fire_smoke=True,fire_fuel_read=0,fire_topo_from_atm=True,
            fire_num_ignitions=1,fire_ignition_start_x1=720.0,fire_ignition_start_y1=720.0,
            fire_ignition_end_x1=720.0,fire_ignition_end_y1=720.0,
            fire_ignition_start_time1=0.0,fire_ignition_end_time1=0.0,
            fire_ignition_radius1=70.0,fire_ignition_ros1=10.0)
    def advance(state,cfg,clock,count):
        for _ in range(count):
            refresh_model_time(state,clock,kernel_launch=True)
            run_steps(state,cfg,1)
            refresh_model_time(state,clock,after_step=True);clock.advance()
            assert state.physics.fire.grid.time_seconds==clock.elapsed_seconds
            assert bool(cp.all(state.chemdiag_sfire_source_lasttime==clock.elapsed_seconds))
    full,cfg,_=make();clock=fractional_clock();advance(full,cfg,clock,6)
    reference=restart.write_restart(tmp_path/'full.npz',full,cfg)
    split,_,_=make();clock=fractional_clock();advance(split,cfg,clock,3)
    midpoint=restart.write_restart(tmp_path/'mid.npz',split,cfg)
    fresh,_,_=make();restart.restore_restart(midpoint,fresh,cfg)
    advance(fresh,cfg,clock,3)
    restored=restart.write_restart(tmp_path/'resumed.npz',fresh,cfg)
    _assert_restart_equal(reference,restored)
    assert bool(cp.any(full.chem_fire_smoke>0))
    print(json.dumps({'control':'fractional-clock-six-vs-three-disk-three',
        'dt':dt,'elapsed_seconds':full.elapsed_seconds,'different_payload_bytes':0,
        'first_chained_dt_word':'3fd55555','second_launch_time_word':'3fd55556'}))
