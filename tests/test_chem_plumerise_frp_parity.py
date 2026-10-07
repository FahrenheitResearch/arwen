"""Pinned GSL full-solver and driver-subset parity."""
import hashlib

import numpy as np
import pytest

from gpuwm.verify.chem_oracle import load, ulp_table, ORACLE_ROOT
from gpuwm.verify.chem_plumerise_ref import frp_driver_override, ebu_distribute
from conftest import requires_gpu

ROOT = ORACLE_ROOT / "smoke" / "ebu_distribute_gsl"


def test_fixture_pins():
    assert hashlib.sha256((ROOT / 'oracle-sha256sums.txt').read_bytes()).hexdigest() == (
        '4f2024d02c8d6453b6d62c1ca42ca8a70f016221c9b2f51906bb0a8b35af1708')
    for line in (ROOT / "oracle-sha256sums.txt").read_text().splitlines():
        digest, relative = line.split(None, 1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_cpu_driver_parity():
    cases = load(ROOT)
    assert len(cases) == 12
    for case in cases.values():
        k1, k2, fraction = frp_driver_override(
            case['frp_inst'], case['plume_k_min'], case['plume_k_max'],
            case['kpbl'], case['uspdavg2d'], case['hpbl2d'],
            wind_eff_opt=int(case['wind_eff_opt']))
        for name, actual in [('k_min', np.int32(k1)), ('k_max', np.int32(k2)),
                             ('flam_frac', fraction),
                             ('ebu', ebu_distribute(k1, k2, fraction,
                                                    case['ebu_in'], case['z_at_w']))]:
            assert ulp_table(actual, case[name]) == {
                'max_ulp': 0, 'n_nonzero': 0, 'n': np.asarray(actual).size}, name


def test_ref_invalid_bounds():
    with pytest.raises(ValueError, match='divide by zero'):
        ebu_distribute(2, 2, .9, 1., np.arange(6, dtype=np.float32))
    with pytest.raises(ValueError, match='undefined emissions'):
        ebu_distribute(2, 3, .9, 1., np.zeros(6, dtype=np.float32))


def test_call_cadence():
    from gpuwm.core.chem_plumerise_cache import plume_call_due
    assert plume_call_due('frp', 36, 2, 60)
    assert plume_call_due('frp', 3600.9, 101, 60)
    assert not plume_call_due('frp', 3599.9, 100, 60)
    assert plume_call_due('frp',3599.9999,100,60)
    assert not plume_call_due('frp', 0, 2, 0)
    assert plume_call_due('landuse', 0, 1, 180, stepfirepl=5)
    assert plume_call_due('landuse', 180, 5, 180, stepfirepl=5)
    assert not plume_call_due('landuse', 144, 4, 180, stepfirepl=5)
    assert plume_call_due('landuse', 10764, 300, 180, adaptive=True, dt=36)
    assert not plume_call_due('landuse', 10727, 299, 180, adaptive=True, dt=36)
    assert plume_call_due('landuse', 36, 2, 0, adaptive=True, dt=36)
    assert not plume_call_due('frp', 0, 2, 60, enabled=False)


def test_local_gpu_refusal(monkeypatch):
    from gpuwm.core.chem_plumerise_cache import distribute_cached
    monkeypatch.setenv('GPUWM_NO_LOCAL_GPU', '1')
    with pytest.raises(RuntimeError, match='protected local device'):
        distribute_cached(None, None, None, None, None, None, None)


FULL_ROOT=ORACLE_ROOT/'smoke/plumerise_frp_gsl'
CPU_CASES=('nz49_s1_f03','nz59_s4_f05')


def test_cpu_solver_parity():
    from gpuwm.verify.chem_plumerise_ref import reference_column
    cases=load(FULL_ROOT)
    assert len(cases)==132
    for name in CPU_CASES:
        actual=reference_column(cases[name],'frp')
        for key,value in actual.items():
            ref=cases[name][key]
            assert ulp_table(value,ref)==dict(max_ulp=0,n_nonzero=0,n=np.asarray(value).size),(name,key)
            assert np.array_equal(np.asarray(value).view(np.uint32),np.asarray(ref).view(np.uint32)),(name,key)


def test_full_fixture_pins():
    assert hashlib.sha256((FULL_ROOT/'oracle-sha256sums.txt').read_bytes()).hexdigest()=='4fe9a8e4bf85f2cd720dde055247e7e657c5dd6a9002aa78803741ff421c145c'
    for line in (FULL_ROOT/'oracle-sha256sums.txt').read_text().splitlines():
        digest,name=line.split(None,1)
        assert hashlib.sha256((FULL_ROOT/name).read_bytes()).hexdigest()==digest


@pytest.mark.gpu
@requires_gpu
def test_full_kernel_all_cases():
    cp=pytest.importorskip('cupy')
    from gpuwm.core.chem_plumerise import launch
    for name,case in load(FULL_ROOT).items():
        met={k:cp.asarray(case[k][:,None,None]) for k in
             ('t_phy','p_phy','rho','u_phy','v_phy','z','z_at_w')}
        met['qv']=cp.asarray(case['qv_in'][:,None,None])
        shape=(1,1)
        result=launch(met,cp.asarray(case['ebu_in'][:,None,None]),frp_inst=cp.full(shape,case['frp_inst'],dtype=cp.float32),
            kpbl=cp.full(shape,case['kpbl'],dtype=cp.int32),uspdavg2d=cp.full(shape,case['uspdavg2d'],dtype=cp.float32),
            hpbl2d=cp.full(shape,case['hpbl2d'],dtype=cp.float32),audit=True)
        if not result['columns'].size:
            assert name.endswith(('f01','f02')); continue
        actual=dict(ebu=cp.asnumpy(result['ebu'][0]).T,
            k_min=cp.asnumpy(result['bounds'][0,0]),k_max=cp.asnumpy(result['bounds'][0,1]),
            flam_frac=cp.asnumpy(result['diagnostics'][0,0]),
            ztopmax=cp.asnumpy(result['diagnostics'][0,1:9]).reshape(4,2).T,
            steps=cp.asnumpy(result['steps'][0]).reshape(4,2).T)
        for f,key in enumerate(('w','t','qv','qc','qh','qi','radius')): actual[key]=cp.asnumpy(result['profiles'][0,f])
        for key,value in actual.items():
            assert ulp_table(value,case[key])==dict(max_ulp=0,n_nonzero=0,n=np.asarray(value).size),(name,key)


@pytest.mark.gpu
@requires_gpu
def test_kernel_driver_parity_and_nonfire_scatter():
    # Import through pytest so CPU collection never opens a device.
    cp = pytest.importorskip('cupy')
    from gpuwm.core.chem_plumerise_cache import distribute_cached
    for case in load(ROOT).values():
        z = case['z_at_w']
        nz = len(z)-1
        columns = cp.array([1], dtype=cp.int32)
        k1 = cp.full((1,3), case['k_min'], dtype=cp.int32)
        k2 = cp.full((1,3), case['k_max'], dtype=cp.int32)
        frac = cp.full((1,3), case['flam_frac'], dtype=cp.float32)
        incoming = cp.full((2,1,3), case['ebu_in'], dtype=cp.float32)
        heights = cp.asarray(np.repeat(z[:,None,None], 3, axis=2))
        output = cp.full((2,nz,1,3), -123.25, dtype=cp.float32)
        distribute_cached(columns, k1, k2, frac, incoming, heights, output)
        actual = cp.asnumpy(output)
        for row in range(2):
            assert ulp_table(actual[row,:,0,1], case['ebu']) == {
                'max_ulp': 0, 'n_nonzero': 0, 'n': nz}
        assert np.all(actual[:,:,:,0] == np.float32(-123.25))
        assert np.all(actual[:,:,:,2] == np.float32(-123.25))
