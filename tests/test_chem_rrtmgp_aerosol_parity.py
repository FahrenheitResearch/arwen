"""Pinned RTE Fortran -O0 grading. No radiation hook is applied here."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.core.chem_rrtmgp_aerosol import AerosolTables, TABLE_PINS, pack_rows
from gpuwm.verify.chem_oracle import load_case, ulp_table

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / 'tests/data/oracles/chem/rrtmgp_aerosol'
OUTPUTS = ('tau', 'ssa', 'g', 'absorption')
MANIFEST_SHA256 = '309091948cb844872c299ceb04ec9e071772a2e7ce30f68802c3248619cc83c4'


def table(kind):
    # This file is a hash-pinned Rust-decoded copy of the pinned NetCDF,
    # not an independent Python decode. It makes GPU grading self contained.
    with np.load(FIX / f'packed-{kind}.npz') as a:
        return AerosolTables(kind, *(a[k].copy() for k in
                          ('packed', 'limits', 'rh', 'bands')))


def test_fixture_pins():
    assert hashlib.sha256((FIX / 'oracle-sha256sums.txt').read_bytes()).hexdigest() == MANIFEST_SHA256
    for line in (FIX / 'oracle-sha256sums.txt').read_text().splitlines():
        digest, name = line.split(None, 1)
        assert hashlib.sha256((FIX / name.strip()).read_bytes()).hexdigest() == digest
    for kind, digest in TABLE_PINS.items():
        path = ROOT / f'gpuwm-data/gpuwm_data/data/rrtmgp/rrtmgp-aerosols-merra-{kind}.nc'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_row_packing():
    rows = [{'optics': {}}, {'optics': {'merra_type': 2, 'merra_bin': 2},
                            'units': 'ug kg-1'}]
    p = pack_rows(rows, table('sw'))
    assert p.indices == (1,)
    assert p.types.tolist() == [2]
    assert p.scales.dtype == np.float32
    with pytest.raises(ValueError, match='mass_scale'):
        pack_rows([{'units': 'ppmv', 'optics': {'merra_type': 3}}], table('sw'))


@requires_gpu
@pytest.mark.parametrize('kind', ['sw', 'lw'])
@pytest.mark.parametrize('family', ['types', 'mixture'])
def test_parity(kind, family):
    import cupy as cp
    from gpuwm.core.chem_rrtmgp_aerosol import optics, launch
    ref = load_case(FIX / 'sp' / f'{kind}_{family}')
    t = table(kind)
    def field(a):
        return cp.asarray(np.ascontiguousarray(a.T[:, None, :]))
    if family == 'types':
        port = optics(t, field(ref['types']), field(ref['sizes']),
                      field(ref['mass']), field(ref['rh']))
    else:
        rows = [{'units': 'ug kg-1', 'optics': {
                 'merra_type': int(tp), 'merra_bin': int(b),
                 'merra_mass_scale': float(scale)}}
                for tp, b, scale in zip(ref['rowtypes'], ref['rowbins'], ref['scales'])]
        port = launch(rows, [field(ref['q'][:,:,r]) for r in range(3)],
                      field(ref['dp']), field(ref['rh']), t)
    measured = {name: ulp_table(a.get(), ref[name]) for name, a in zip(OUTPUTS, port)}
    print(kind, family, json.dumps(measured, sort_keys=True))
    for name in OUTPUTS:
        assert measured[name]['max_ulp'] == 0
    assert np.all(ref['tau'][:, -1, :] == 0)
    print('CuPy pool bytes', cp.get_default_memory_pool().total_bytes())
    assert cp.get_default_memory_pool().total_bytes() < 2_000_000_000
    if family == 'types':
        # Above-top RH must remain constant, not extrapolate or fail.
        assert np.all(ref['ssa'][:10, 0, :] != 0)
        assert np.any(ref['tau'][5:10, 0, :] != ref['tau'][5:10, 3, :])


@requires_gpu
def test_validity_and_empty_rows():
    import cupy as cp
    from gpuwm.core.chem_rrtmgp_aerosol import optics, launch
    t = table('sw'); shape = (3, 1, 2)
    tp=cp.ones(shape,cp.int32); size=cp.ones(shape,cp.float32)
    mass=cp.full(shape,1.e-5,cp.float32); rh=cp.full(shape,.5,cp.float32)
    for a,value,message in [(tp,8,'type=1'),(size,11,'size=2'),(rh,1.01,'RH=4')]:
        old=a.copy(); a.fill(value)
        with pytest.raises(ValueError,match=message): optics(t,tp,size,mass,rh)
        a[...] = old
    tp.fill(0); size.fill(0); rh.fill(2)
    assert all(bool(cp.all(a==0)) for a in optics(t,tp,size,mass,rh))
    rh.fill(.5)
    outputs=launch([{'optics':{}}],[mass],mass,rh,t)
    assert all(bool(cp.all(a==0)) for a in outputs)


@requires_gpu
def test_rh_top_and_size_edge():
    import cupy as cp
    from gpuwm.core.chem_rrtmgp_aerosol import optics
    t=table('sw'); shape=(1,1,3)
    tp=cp.full(shape,2,cp.int32); size=cp.full(shape,1.,cp.float32)
    mass=cp.full(shape,1.e-5,cp.float32)
    rh=cp.asarray(np.array([t.rh[-1],1.,.5],np.float32).reshape(shape))
    tau,ssa,g,absorb=optics(t,tp,size,mass,rh)
    for a in (tau,ssa,g,absorb): assert bool(cp.all(a[0]==a[1]))
    assert bool(cp.any(tau[0]!=tau[2]))


def test_precision_consequence():
    limits = {'sw_types': [0,126820179,173789128,3400910],
              'lw_types': [0,136088791,243269631,61],
              'sw_mixture': [2,2,2,470], 'lw_mixture': [2,3,3,29]}
    for case, expected in limits.items():
        sp=load_case(FIX / 'sp' / case); dp=load_case(FIX / 'dp' / case)
        assert [ulp_table(sp[n],dp[n])['max_ulp'] for n in OUTPUTS] == expected
