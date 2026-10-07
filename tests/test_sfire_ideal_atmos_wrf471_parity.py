"""Byte-extracted compiled native ideal atmospheric initialization controls."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import numpy as np
import pytest
from conftest import requires_gpu
from tools.sfire_wrf471_oracle.fixture import words

ROOT=Path(__file__).resolve().parents[1]/'tools/sfire_coupled_ideal/initializer_atmos/fixtures'


def load(name):
    receipt=json.loads((ROOT/'receipt.json').read_text());p=ROOT/(name+'.npz')
    assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['cases'][name]['sha256']
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def raw():return json.loads((ROOT/'receipt.json').read_text())['raw_sounding']


def check(actual,expected,name):
    import cupy as cp
    r=words(cp.asnumpy(actual),expected);print(json.dumps({name:r}))
    assert r['different_words']==0 and r['max_ulp']==0 and r['nonfinite_differences']==0,(name,r)


@requires_gpu
@pytest.mark.parametrize('dry',[False,True])
def test_native_prepared_sounding(dry):
    from gpuwm.core.sfire_ideal_atmos import prepare_sounding
    got=prepare_sounding(raw(),dry=dry);ref=load('sounding_dry'+str(int(dry)))
    for name in ref:check(got[name],ref[name],'sounding/'+str(dry)+'/'+name)


@requires_gpu
@pytest.mark.parametrize('profile',['elevated','two_levels'])
@pytest.mark.parametrize('dry',[False,True])
def test_native_first_height_hydrostatic_iterations(profile,dry):
    from gpuwm.core.sfire_ideal_atmos import prepare_sounding
    ref=load('sounding_'+profile+'_dry'+str(int(dry)))
    data={name:ref['input_'+name] for name in ('height','theta','qv','u','v')}
    data.update(psurf=ref['input_surface'][0],theta_surface=ref['input_surface'][1])
    got=prepare_sounding(data,dry=dry)
    for name in got:check(got[name],ref[name],'sounding/'+profile+'/'+str(dry)+'/'+name)


@requires_gpu
@pytest.mark.parametrize('case',[1,2])
def test_native_interpolation_all_branches(case):
    from gpuwm.core.sfire_ideal_atmos import interpolate_sounding
    ref=load('interpolation'+str(case))
    got=interpolate_sounding(ref['values'],ref['coordinates'],ref['targets'])
    check(got,ref['out'],'interp/'+str(case))


@requires_gpu
@pytest.mark.parametrize('case',range(1,7))
def test_native_columns_and_corrected_skin(case):
    import cupy as cp
    from gpuwm.core.sfire_ideal_atmos import prepare_sounding,initialize_atmosphere
    ref=load('columns'+str(case)+'_corrected1');original=load('columns'+str(case)+'_corrected0')
    settings=ref['settings'];flags=ref['flags'];nz,ny,nx=ref['T'].shape
    keys=('dx','dy','ztop','delt_perturbation','xrad_perturbation','yrad_perturbation','zrad_perturbation','hght_perturbation')
    cfg=SimpleNamespace(nx=nx,ny=ny,nz=nz,sfc_full_init=bool(flags[1]),**dict(zip(keys,map(float,settings))))
    vertical={key:cp.asarray(ref[key]) for key in ('p_top','cf1','cf2','cf3','dnw','rdnw','rdn','c1f','c2f','c1h','c2h','c3h','c4h')}
    sounding={'dry':prepare_sounding(raw(),dry=True),'moist':prepare_sounding(raw(),dry=False)}
    got=initialize_atmosphere(cfg,ref['terrain'],vertical,sounding,native_use_theta_m=int(flags[0]))
    for name in got:check(got[name],ref[name],'columns/'+str(case)+'/'+name)
    for name in ('U','V','W','PH','PHB','MU','MUB','T_DRY','T','T_INIT','P','PB','AL','ALB','ALT','QVAPOR'):
        assert original[name].tobytes()==ref[name].tobytes()
    if case==2:
        assert np.ptp(original['TSK'])==0
        assert np.ptp(ref['TSK'])>0
        assert original['TSK'].tobytes()!=ref['TSK'].tobytes()
