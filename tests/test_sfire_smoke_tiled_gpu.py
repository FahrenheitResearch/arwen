"""Native whole-domain smoke accounting over device and mapped host fields."""
from dataclasses import replace
import numpy as np

from conftest import requires_gpu
import pytest


@requires_gpu
def test_external_whole_domain_ledger_is_resident_exact():
    import cupy as cp
    from test_restart import _physics_state
    from gpuwm.core.dycore import run_steps
    from gpuwm.core.chem_driver import _masses
    from tilestream.chem_mass import global_masses
    from tilestream.hoststore import alloc_pinned_array
    from tilestream.sfire_smoke import (GlobalSmokeLedger, GLOBAL_KEYS,
                                       configure_tile, transport_fields)
    options=dict(nx=16,ny=12,nz=16,dx=90.0,dy=120.0,dt=.25,
        mp_physics=1,cu_physics=0,ra_physics=0,ifire=2,sr_x=4,sr_y=3,
        fire_smoke=True,fire_fuel_read=0,fire_topo_from_atm=True,
        fire_num_ignitions=1,fire_ignition_start_x1=720.0,fire_ignition_start_y1=720.0,
        fire_ignition_end_x1=720.0,fire_ignition_end_y1=720.0,
        fire_ignition_start_time1=0.0,fire_ignition_end_time1=0.0,
        fire_ignition_radius1=70.0,fire_ignition_ros1=280.0)
    resident,cfg,_=_physics_state(cp,**options)
    external,_,_=_physics_state(cp,**options)
    configure_tile(external)
    store={'state/mup':external.mup,**transport_fields(external)}
    for row in external.chem.transported:
        store['state/'+row.state_attr]=getattr(external,row.state_attr)
    for key in GLOBAL_KEYS:
        store[key]=getattr(external,key.removeprefix('state/'))
    ledger=GlobalSmokeLedger(cfg,{},external)
    for _ in range(3):
        ledger.begin(store)
        run_steps(external,cfg,1)
        ledger.end(store)
        run_steps(resident,cfg,1)
        for key in GLOBAL_KEYS:
            name=key.removeprefix('state/')
            assert cp.asnumpy(store[key]).tobytes()==cp.asnumpy(getattr(resident,name)).tobytes(),key
    assert cp.asnumpy(external.chem_fire_smoke).tobytes()==cp.asnumpy(resident.chem_fire_smoke).tobytes()
    pinned={}
    for key in ('state/mup','state/chem_fire_smoke'):
        value=cp.asnumpy(store[key]);pinned[key]=alloc_pinned_array(value.shape,value.dtype)
        pinned[key][...] = value
    geography={}
    for name in ('mub2d','msft'):
        value=cp.asnumpy(getattr(external,name));geography['setup/'+name]=alloc_pinned_array(value.shape,value.dtype)
        geography['setup/'+name][...] = value
    direct=_masses(external,cfg,external.chem.transported,external.chem.fields(external),external.mup)
    mapped=global_masses(pinned,('state/chem_fire_smoke',),'state/mup',cfg,geography,external)
    assert cp.asnumpy(mapped).tobytes()==cp.asnumpy(direct).tobytes()
    print('global smoke ledger: 3 steps, 11 vectors, byte exact; mapped pinned mass byte exact')


@requires_gpu
@pytest.mark.parametrize('mode', ['ring', 'shadow'])
@pytest.mark.parametrize('spotting', [False, True])
def test_bulk_smoke_tiles_full_state_history_and_cold_disk(tmp_path, mode, spotting, record_property):
    import cupy as cp
    from test_restart import _physics_state
    from test_sfire_tilestream_gpu import _stream, _assert_words
    from gpuwm.core.streaming import prime_lazy_carriers, streamed_store_inventory, publish_store
    from gpuwm.core.dycore import step
    from gpuwm.io.wrfout import _device_state_frame
    from gpuwm.core.chem_driver import chem_ledger_receipt
    def state():
        result = _physics_state(cp, nx=96, ny=72, nz=16, dx=90., dy=120., dt=.25,
            mp_physics=1, cu_physics=0, ra_physics=0, ifire=2, sr_x=4, sr_y=3,
            fire_smoke=True, fire_fuel_read=0, fire_fuel_cat=1, fire_fmc_read=0,
            fire_boundary_guard=0, fmoist_run=True, fmoist_interp=True, fmoist_freq=1,
            fire_num_ignitions=1, fire_ignition_start_x1=2880.,
            fire_ignition_start_y1=2880., fire_ignition_radius1=160.,
            fire_ignition_ros1=280., fs_firebrand_gen_lim=7 if spotting else 0,
            fs_array_maxsize=256, fs_firebrand_gen_dt=2, fs_firebrand_gen_levels=3,
            fs_firebrand_gen_mom3d_dt=1, fs_firebrand_gen_levrand=True, trackember=True)
        prime_lazy_carriers(result[0], result[1])
        return result
    resident, cfg, _ = state()
    tiled, tcfg, _ = state()
    stream = _stream(tiled, tcfg, mode)
    publish_store(tiled, stream)
    carrier_words = history_words = 0
    carrier_arrays = history_arrays = 0
    for number in range(8):
        step(resident, cfg, fire_history_due=True)
        stream(tiled, tcfg, fire_history_due=True)
        carriers = streamed_store_inventory()(resident, None)
        _assert_words(cp, carriers, stream.store)
        carrier_words += sum(value.nbytes // 4 for value in carriers.values())
        carrier_arrays += len(carriers)
        assert chem_ledger_receipt(tiled) == chem_ledger_receipt(resident)
        native_frame = _device_state_frame(resident)
        history_words += sum(value.nbytes // 4 for value in native_frame.values())
        history_arrays += len(native_frame)
        tiled_frame = stream.history_fields()
        assert set(native_frame) == set(tiled_frame)
        for name, expected in native_frame.items():
            expected = cp.asnumpy(expected)
            assert expected.shape == tiled_frame[name].shape, name
            assert expected.tobytes() == tiled_frame[name].tobytes(), name
        if number == 3:
            checkpoint = tmp_path / 'bulk-tiles-mid.npz'
            stream.write_restart(checkpoint, tcfg)
            stream.tiled_run.close()
            tiled, tcfg, _ = state()
            stream = _stream(tiled, tcfg, mode)
            publish_store(tiled, stream)
            stream.restore_restart(checkpoint, tcfg)
            restored = stream.history_fields()
            for name, expected in native_frame.items():
                assert cp.asnumpy(expected).tobytes() == restored[name].tobytes(), name
    assert bool(cp.any(resident.chem_fire_smoke > 0))
    if spotting:
        assert int(resident.physics.fire.spotting.control[0].item()) > 0
    record_property('carrier_32bit_words', carrier_words)
    record_property('history_32bit_words', history_words)
    record_property('carrier_arrays', carrier_arrays)
    record_property('history_arrays', history_arrays)
    record_property('different_words', 0)
    record_property('completed_sweeps', 8)
    record_property('cold_disk_split', '4+4')
    print(f'bulk smoke {mode} spotting={spotting}: 8 sweeps, all carriers/history/ledger byte exact, fresh disk 4+4')
