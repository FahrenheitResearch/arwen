"""Actual coupled SFIRE continuation through disk checkpoints and history."""
import json

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.io import restart
from test_restart import (_physics_state, _assert_restart_equal,
                          _rewrite_restart_archive)


def _coupled_state(cp, mapped=False, spotting=False):
    spotting_options=dict(fs_array_maxsize=256,fs_firebrand_gen_lim=7,fs_firebrand_gen_dt=2,
        fs_firebrand_gen_levels=3,fs_firebrand_gen_levrand=True,fs_firebrand_gen_mom3d_dt=1,
        trackember=True) if spotting else {}
    state, cfg, driver = _physics_state(cp, nx=16, ny=12, nz=16, dx=90.0, dy=120.0,
        dt=0.25, mp_physics=1, cu_physics=0, ra_physics=0, ifire=2,
        sr_x=4, sr_y=3, fire_fuel_read=0, fire_topo_from_atm=True,
        fire_fuel_cat=1, fire_fmc_read=0, fire_boundary_guard=2,
        fmoist_run=True, fmoist_interp=True, fmoist_freq=1,
        fire_num_ignitions=1, fire_ignition_start_x1=720.0,
        fire_ignition_start_y1=720.0, fire_ignition_end_x1=720.0,
        fire_ignition_end_y1=720.0, fire_ignition_radius1=70.0,
        fire_ignition_start_time1=0.0, fire_ignition_end_time1=0.0,
        fire_ignition_ros1=280.0,**spotting_options)
    if mapped:
        state.has_msf = True
        state.msft[...] = cp.linspace(0.82, 1.15, cfg.nx*cfg.ny, dtype=cp.float32).reshape(cfg.ny, cfg.nx)
        state.msfu[...] = cp.linspace(0.85, 1.12, (cfg.nx+1)*cfg.ny, dtype=cp.float32).reshape(cfg.ny, cfg.nx+1)
        state.msfv[...] = cp.linspace(0.87, 1.16, cfg.nx*(cfg.ny+1), dtype=cp.float32).reshape(cfg.ny+1, cfg.nx)
    return state, cfg, driver


@requires_gpu
@pytest.mark.parametrize("nfmc", [5,7])
def test_native_history_inputs_persist_without_activating_perimeter(tmp_path,nfmc):
    import cupy as cp
    import netCDF4
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.io.sfire_schema import sfire_history_shapes
    from gpuwm.io.wrfout import WrfoutWriter
    from gpuwm.io.restart import STATE_SERIALIZED_ATTRS
    from dataclasses import replace
    def initial(provided=True):
        state, cfg, driver = _coupled_state(cp)
        cfg=replace(cfg,nfmc=nfmc)
        static={}
        if nfmc>5:
            for name,value in (("FMC_GC",.9375),("FMC_EQUI",.625),("FMC_TEND",31.25)):
                static[name]=cp.zeros((nfmc,cfg.ny,cfg.nx),cp.float32)
                static[name][5:].fill(value)
        if provided:
            static.update(LFN_HIST=cp.full(driver.fire.fine_shape,-17.5,cp.float32),
                          LFN_TIME=cp.asarray([-125.5],cp.float32))
        driver.fire = FireCoupler(state,cfg,driver.fields,static)
        return state, cfg, driver
    reference, cfg, source = initial()
    plain, _, baseline = initial(False)
    run_steps(reference, cfg, 6)
    run_steps(plain, cfg, 6)
    for name in STATE_SERIALIZED_ATTRS:
        value = getattr(reference, name, None)
        if value is not None:
            assert cp.asnumpy(value).tobytes() == cp.asnumpy(getattr(plain, name)).tobytes(), name
    for name, value in source.fire.arrays().items():
        if name not in ("lfn_time", "grid.lfn_hist"):
            assert cp.asnumpy(value).tobytes() == cp.asnumpy(baseline.fire.arrays()[name]).tobytes(), name
    split, _, _ = initial()
    run_steps(split, cfg, 3)
    mid = restart.write_restart(tmp_path/"history-mid.npz", split, cfg)
    resumed, _, target = initial()
    restart.restore_restart(mid, resumed, cfg)
    run_steps(resumed, cfg, 3)
    full = restart.write_restart(tmp_path/"history-full.npz", reference, cfg)
    continued = restart.write_restart(tmp_path/"history-resumed.npz", resumed, cfg)
    _assert_restart_equal(full, continued)
    fields = {name:cp.asnumpy(value) for name,value in target.fire.output_fields().items()}
    assert {name:value.shape for name,value in fields.items()} == sfire_history_shapes(cfg)
    assert bool(np.all(fields["LFN_HIST"] == -17.5))
    assert fields["LFN_TIME"].tobytes() == np.asarray([-125.5], np.float32).tobytes()
    assert fields["FMC_GC"].shape==(nfmc,cfg.ny,cfg.nx)
    if nfmc>5:
        for name,value in (("FMC_GC",.9375),("FMC_EQUI",.625),("FMC_TEND",31.25)):
            assert bool(np.all(fields[name][5:]==np.float32(value))),name
    path = tmp_path/"retained-history.nc"
    writer = WrfoutWriter(path,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,
        field_schema=fields,engine="rust",global_attrs=target.fire.output_attributes())
    writer.write_frame("2026-10-02_00:00:00",fields);writer.close()
    with netCDF4.Dataset(path) as ds:
        assert ds.variables["LFN_TIME"].shape == (1,1)
        assert ds.variables["LFN_TIME"][0].data.tobytes() == fields["LFN_TIME"].tobytes()
        assert ds.variables["LFN_HIST"][0].data.tobytes() == fields["LFN_HIST"].tobytes()
        for name in ("FMC_GC","FMC_EQUI","FMC_TEND"):
            assert ds.variables[name].shape==(1,nfmc,cfg.ny,cfg.nx)
            assert ds.variables[name][0].data.tobytes()==fields[name].tobytes(),name


@requires_gpu
@pytest.mark.parametrize("mapped", [False, True])
def test_actual_coupled_fire_disk_continuation_is_bit_exact(tmp_path, mapped):
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    full, cfg, driver = _coupled_state(cp, mapped)
    run_steps(full, cfg, 6)
    reference = restart.write_restart(tmp_path / "reference.npz", full, cfg)
    assert bool(cp.any(driver.fire.grid.data["fire_area"] > 0))
    assert bool(cp.any(driver.fire.data["grnhfx"] > 0))
    assert bool(cp.any(driver.fire_tendencies.rtheta > 0))
    assert driver.fire.grid.moisture_initialized
    assert driver.call_counts["fire"] == 6
    for value in driver.fire.output_fields().values():
        assert bool(cp.all(cp.isfinite(value)))
    for name in ("u", "v", "w", "thp", "php", "p", "qv", "qc", "qr"):
        assert bool(cp.all(cp.isfinite(getattr(full, name))))

    split, split_cfg, _ = _coupled_state(cp, mapped)
    run_steps(split, split_cfg, 3)
    mid = restart.write_restart(tmp_path / "mid.npz", split, split_cfg)
    resumed, resumed_cfg, resumed_driver = _coupled_state(cp, mapped)
    restart.restore_restart(mid, resumed, resumed_cfg)
    assert all(isinstance(a, cp.ndarray) for a in resumed_driver.fire.arrays().values())
    run_steps(resumed, resumed_cfg, 3)
    result = restart.write_restart(tmp_path / "resumed.npz", resumed, resumed_cfg)
    _assert_restart_equal(result, reference)
    assert restart.read_restart_header(result)["driver"]["fire"] == restart.read_restart_header(reference)["driver"]["fire"]


@requires_gpu
def test_actual_fire_clock_corruption_refuses_before_mutation(tmp_path):
    import cupy as cp
    from gpuwm.core.dycore import run_steps
    source, cfg, _ = _coupled_state(cp)
    run_steps(source, cfg, 3)
    path = restart.write_restart(tmp_path / "source.npz", source, cfg)

    def corrupt(payload, header):
        header["driver"]["fire"]["grid"]["moisture_nexttime"] = "bad-clock"

    broken = _rewrite_restart_archive(path, tmp_path / "broken.npz", corrupt)
    live, live_cfg, driver = _coupled_state(cp)
    state_bytes = {name: cp.asnumpy(getattr(live, name)).tobytes()
                   for name in restart.STATE_SERIALIZED_ATTRS if getattr(live, name, None) is not None}
    fire_bytes = {name: cp.asnumpy(value).tobytes() for name, value in driver.fire.arrays().items()}
    with pytest.raises(restart.RestartMismatchError, match="continuation metadata is malformed"):
        restart.restore_restart(broken, live, live_cfg)
    assert {name: cp.asnumpy(getattr(live, name)).tobytes() for name in state_bytes} == state_bytes
    assert {name: cp.asnumpy(value).tobytes() for name, value in driver.fire.arrays().items()} == fire_bytes


@requires_gpu
def test_actual_fire_output_fields_reach_rust_history(tmp_path):
    import cupy as cp
    import netCDF4
    from gpuwm.core.dycore import run_steps
    from gpuwm.io.wrfout import WrfoutWriter
    state, cfg, driver = _coupled_state(cp)
    run_steps(state, cfg, 3)
    fields = {name: cp.asnumpy(value) for name, value in driver.fire.output_fields().items()}
    path = tmp_path / "fire.nc"
    writer = WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                         dx=cfg.dx, dy=cfg.dy, field_schema=fields, engine="rust",
                         global_attrs=driver.fire.output_attributes())
    writer.write_frame("2026-10-02_00:00:00", fields)
    writer.close()
    with netCDF4.Dataset(path) as ds:
        assert ds.variables["FIRE_AREA"].shape == (1, cfg.ny*cfg.sr_y, cfg.nx*cfg.sr_x)
        assert ds.variables["UAH"].shape == (1, cfg.ny, cfg.nx+1)
        assert ds.variables["VAH"].shape == (1, cfg.ny+1, cfg.nx)
        assert ds.variables["FMC_TEND"].shape == (1, cfg.nfmc, cfg.ny, cfg.nx)
        assert ds.FIRE_COORDINATE_MODE == "metric"
        assert ds.variables["FXLAT"].units == ds.variables["FXLONG"].units == "m"
        assert set(fields) <= set(ds.variables)
        assert ds.variables["FIRE_AREA"][0].data.tobytes() == fields["FIRE_AREA"].tobytes()


@requires_gpu
@pytest.mark.parametrize("nfmc",[1,5,7])
def test_inactive_moisture_and_spotting_registry_fields_survive_disk_and_rust(tmp_path,nfmc):
    import cupy as cp
    import netCDF4
    from dataclasses import replace
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.sfire_coupler import FireCoupler
    from gpuwm.core.sfire_spotting import COARSE_REAL_FIELDS,COARSE_INTEGER_FIELDS
    from gpuwm.io.wrfout import WrfoutWriter
    def initial():
        state,cfg,driver=_coupled_state(cp)
        cfg=replace(cfg,fmoist_run=False,fmoist_interp=False,nfmc=nfmc)
        driver.fire=FireCoupler(state,cfg,driver.fields)
        return state,cfg,driver
    reference,cfg,driver=initial()
    run_steps(reference,cfg,6)
    path=restart.write_restart(tmp_path/"reference.npz",reference,cfg)
    split,split_cfg,_=initial()
    run_steps(split,split_cfg,3)
    mid=restart.write_restart(tmp_path/"mid.npz",split,split_cfg)
    resumed,resumed_cfg,resumed_driver=initial()
    restart.restore_restart(mid,resumed,resumed_cfg)
    run_steps(resumed,resumed_cfg,3)
    result=restart.write_restart(tmp_path/"resumed.npz",resumed,resumed_cfg)
    _assert_restart_equal(path,result)
    assert not resumed_driver.fire.grid.moisture_initialized
    assert resumed_driver.fire.grid.moisture_lasttime==resumed_driver.fire.grid.moisture_nexttime==0.
    assert all(bool(cp.all(value==0)) for value in resumed_driver.fire.grid.moisture.arrays().values())
    fields={key:cp.asnumpy(value) for key,value in driver.fire.output_fields().items()}
    assert fields["FMC_GC"].shape==(cfg.nfmc,cfg.ny,cfg.nx)
    for key in (*COARSE_REAL_FIELDS,*COARSE_INTEGER_FIELDS):
        assert fields[key.upper()].shape==(cfg.ny,cfg.nx)
        assert np.count_nonzero(fields[key.upper()])==0
    for key in COARSE_INTEGER_FIELDS:
        assert fields[key.upper()].dtype==np.int32
    output=tmp_path/"inactive.nc"
    writer=WrfoutWriter(output,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,
        field_schema=fields,engine="rust",global_attrs=driver.fire.output_attributes())
    writer.write_frame("2026-10-02_00:00:00",fields);writer.close()
    with netCDF4.Dataset(output) as ds:
        assert set(fields)<=set(ds.variables)
        for key in (*COARSE_REAL_FIELDS,*COARSE_INTEGER_FIELDS):
            assert ds.variables[key.upper()][0].data.tobytes()==fields[key.upper()].tobytes()
        for key,value in fields.items():
            actual=ds.variables[key][0].data
            if key in ("RTHFRTEN","RQVFRTEN"):
                # Native fire tendency history retains its unused W top slot.
                assert actual.shape==(cfg.nz+1,cfg.ny,cfg.nx),key
                assert actual[:cfg.nz].tobytes()==value.tobytes(),key
                assert np.count_nonzero(actual[cfg.nz])==0,key
            else:
                assert actual.shape==value.shape,key
                assert actual.tobytes()==value.tobytes(),key
        assert ds.variables["FMC_GC"].shape==(1,cfg.nfmc,cfg.ny,cfg.nx)


@requires_gpu
def test_actual_firebrand_disk_continuation_preserves_typed_particles(tmp_path):
    import cupy as cp
    from gpuwm.core.dycore import step
    def advance(state,cfg,first,last):
        for count in range(first,last):
            step(state,cfg,fire_history_due=(count+1)%4==0)
    full,cfg,driver=_coupled_state(cp,spotting=True)
    advance(full,cfg,0,8)
    reference=restart.write_restart(tmp_path/"reference.npz",full,cfg)
    assert driver.fire.spotting.step_count==8
    assert int(driver.fire.spotting.control[0].item())>0
    assert bool(cp.any(driver.fire.spotting.identifiers[0]>0))
    assert driver.fire.spotting.count_reset
    assert all(bool(cp.all(cp.isfinite(value))) for value in driver.fire.spotting.arrays().values())
    split,split_cfg,split_driver=_coupled_state(cp,spotting=True)
    advance(split,split_cfg,0,4)
    midpoint=restart.write_restart(tmp_path/"mid.npz",split,split_cfg)
    resumed,resumed_cfg,resumed_driver=_coupled_state(cp,spotting=True)
    restart.restore_restart(midpoint,resumed,resumed_cfg)
    assert resumed_driver.fire.spotting.metadata()==split_driver.fire.spotting.metadata()
    advance(resumed,resumed_cfg,4,8)
    result=restart.write_restart(tmp_path/"resumed.npz",resumed,resumed_cfg)
    _assert_restart_equal(result,reference)
    with np.load(result,allow_pickle=False) as archive:
        assert archive["fire/spotting.fs_p_id"].dtype==np.int32
        assert archive["fire/spotting.fs_gen_inst"].dtype==np.int32
        assert archive["fire/spotting.fs_p_mass"].dtype==np.float32


@requires_gpu
def test_actual_firebrand_native_fields_reach_rust_history(tmp_path):
    import cupy as cp
    import netCDF4
    from gpuwm.core.dycore import step
    from gpuwm.io.wrfout import WrfoutWriter
    state,cfg,driver=_coupled_state(cp,spotting=True)
    for count in range(4):
        step(state,cfg,fire_history_due=count==3)
    fields={name:cp.asnumpy(value) for name,value in driver.fire.output_fields().items()}
    path=tmp_path/"firebrand.nc"
    writer=WrfoutWriter(path,nx=cfg.nx,ny=cfg.ny,nz=cfg.nz,dx=cfg.dx,dy=cfg.dy,
        field_schema=fields,engine="rust",global_attrs=driver.fire.output_attributes())
    writer.write_frame("2026-10-02_00:00:00",fields)
    writer.close()
    with netCDF4.Dataset(path) as ds:
        assert ds.variables["FS_P_ID"].dimensions==("Time","fs_maxsize")
        assert ds.variables["FS_P_ID"].dtype==np.int32
        assert ds.variables["FS_P_Z"].units=="m"
        assert ds.variables["FS_LAST_GEN_DT"].dimensions==("Time",)
        for name,value in fields.items():
            if name.startswith("FS_"):
                assert ds.variables[name][0].data.tobytes()==value.tobytes()
