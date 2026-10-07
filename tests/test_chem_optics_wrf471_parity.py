"""WRF-Chem 4.7.1 optics, ordered sums and RRTMG feedback parity."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from conftest import requires_gpu
from gpuwm.verify.chem_oracle import load,load_case,ulp_table
from gpuwm.core.chem_optics import data_dir

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/data/oracles/chem/gocart/optics'
# The tables ship in the gpuwm-data companion (gpuwm.data_assets.COMPANION_TREES);
# read them where the loader reads them.
DATA=data_dir()
PINS=json.loads((DATA/'parity-pins.json').read_text())
TABLE_MANIFEST_SHA256='d39e1c78ddd3465f0bf3a5e4b329e626be508ac21847ca94fa3a5a11f50486af'
FIXTURE_MANIFEST_SHA256='195c1e6fd226a31c13c55cc393ea34f11f439614296e4315a6e001f88197ee25'
CASES=sorted(p.name for p in FIX.iterdir() if (p/'chem.bin').is_file())
ROWS=json.loads((DATA/'gocart_simple_optics_rows.json').read_text())

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _verify_list(directory,name):
    for line in (directory/name).read_text().splitlines():
        digest,path=line.split(maxsplit=1)
        assert _sha(directory/path)==digest,path

def test_fixture_and_table_sha256_pins():
    assert _sha(DATA/'table-sha256sums.txt')==TABLE_MANIFEST_SHA256
    assert _sha(FIX/'oracle-sha256sums.txt')==FIXTURE_MANIFEST_SHA256
    for path,digest in PINS.items():assert _sha(ROOT/path)==digest,path
    _verify_list(FIX,'oracle-sha256sums.txt')
    _verify_list(DATA,'table-sha256sums.txt')
    # The WRF sources themselves are checked by build_optics.sh against
    # source-sha256sums.txt (pinned above) before anything is compiled.

def _field(v):
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(v.transpose(1,2,0)))
def _bands(v):
    if v.ndim==2:v=v[:,:,None]
    return np.ascontiguousarray(v[:,:,None,:])
def _assert_words(a,b):
    import cupy as cp
    p=cp.asnumpy(a)
    assert ulp_table(p,b)['max_ulp']==0
    np.testing.assert_array_equal(p.view(np.uint32),b.view(np.uint32))
def _launch(f,rows=ROWS,indices=None,diagnostics=True):
    from gpuwm.core.chem_optics import gocart_optics
    if indices is None:indices=range(19)
    fields=[_field(f['chem'][:,:,:,i+1]) for i in indices]
    return gocart_optics(rows,fields,_field(f['alt']),_field(f['relhum']),_field(f['dz8w']),diagnostics=diagnostics)

@requires_gpu
@pytest.mark.parametrize('case',CASES)
def test_gocart_every_output(case):
    f=load_case(FIX/case);o=_launch(f)
    for name in ['tauaer','extaer','waer','gaer','bscoef','tauaerlw','extaerlw']:
        _assert_words(o[name],_bands(f[name]))
    _assert_words(o['EXTCOF55'],f['EXTCOF55'].transpose(1,2,0).copy())
    _assert_words(o['AOD5502D'],f['AOD5502D'].reshape(1,-1))
    for name in ['radius','number']:_assert_words(o[name],f[name].transpose(3,1,2,0).copy())
    for wave,offset in [('sw',0),('lw',4)]:
        ref=np.stack([f[wave+'ri_real'],f[wave+'ri_imag']],axis=0).transpose(3,1,0,2,4)[:,:,:,:,None,:].copy()
        _assert_words(o['refindex'][offset:offset+ref.shape[0]],ref)
    _assert_words(o['moments'],f['moments'].transpose(2,0,1,3)[:,:,:,None,:].copy())

@requires_gpu
@pytest.mark.parametrize('case',CASES+['rrtmg_edges'])
def test_rrtmg_isolated(case):
    import cupy as cp
    from gpuwm.core.chem_optics import rrtmg_sw_bands
    f=load_case(FIX/case)
    b=rrtmg_sw_bands(*[cp.asarray(_bands(f[n])) for n in ['tauaer','waer','gaer']])
    for name in ['tau','ssa','asm']:_assert_words(b[name],f['band_'+name].transpose(1,2,0)[:,:,None,:].copy())

@requires_gpu
@pytest.mark.parametrize('case',['lite_zero','lite_negative_zero','negative_zero'])
def test_lite_absent_rows_signed_zero(case):
    import cupy as cp
    f=load_case(FIX/case)
    # WRF registry positions of dms, msa, p25, p10. Case-only identifiers.
    indices=[i for i in range(19) if i not in (2,3,4,18)]
    subset=[ROWS[i] for i in indices]
    full=_launch(f);lite=_launch(f,subset,indices)
    for name in full:
        if name=='_rows':continue
        a=cp.asnumpy(full[name]);b=cp.asnumpy(lite[name])
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32),err_msg=name)
    for name in ['tauaer','extaer','waer','gaer','bscoef','tauaerlw','extaerlw']:
        _assert_words(lite[name],_bands(f[name]))

@requires_gpu
def test_six_columns_x_fastest_and_no_diagnostic_workspace():
    import cupy as cp
    from gpuwm.core.chem_optics import gocart_optics,rrtmg_sw_bands
    names=['dust_haboob','salt_030','salt_095','carbon_sulfate','all_species','threshold']
    fs=[load_case(FIX/n) for n in names]
    def combine(name):
        return cp.asarray(np.stack([f[name][0,:,0] for f in fs],axis=1).reshape(50,2,3).copy())
    fields=[cp.asarray(np.stack([f['chem'][0,:,0,i+1] for f in fs],axis=1).reshape(50,2,3).copy()) for i in range(19)]
    # The row name is deliberately changed. It must have no process effect.
    rows=[{**r,'name':f'arbitrary_{i}'} for i,r in enumerate(ROWS)]
    o=gocart_optics(rows,fields,combine('alt'),combine('relhum'),combine('dz8w'))
    assert 'radius' not in o
    for name in ['tauaer','extaer','waer','gaer','bscoef','tauaerlw','extaerlw']:
        ref=np.stack([f[name][:,:,0] for f in fs],axis=2).reshape(fs[0][name].shape[0],50,2,3)
        _assert_words(o[name],ref)
    b=rrtmg_sw_bands(o['tauaer'],o['waer'],o['gaer'])
    for name in ['tau','ssa','asm']:
        ref=np.stack([f['band_'+name][0] for f in fs],axis=2).reshape(50,14,2,3)
        _assert_words(b[name],ref)

def test_fixtures_reach_threshold_rescale_and_rh_cap():
    f=load(FIX)
    np.testing.assert_array_equal(f['salt_095']['tauaer'],f['salt_099']['tauaer'])
    assert not np.array_equal(f['salt_030']['tauaer'],f['salt_080']['tauaer'])
    assert not np.array_equal(f['dry_mix']['tauaer'],f['all_species']['tauaer'])
    assert np.all(f['threshold']['band_tau']==0)
    # The oracle warnings record the unscaled band sums above 6.
    assert np.any(np.sum(f['dust_haboob']['band_tau'],axis=1)>5.99999)
    e=f['rrtmg_edges']
    assert np.all(e['band_tau'][:,0:2,:]==0)
    assert np.any(e['band_ssa']==.4) and np.any(e['band_ssa']==1.)
    assert np.any(e['band_asm']==.5) and np.any(e['band_asm']==1.)

@requires_gpu
def test_one_word_mutation_is_rejected():
    import cupy as cp
    f=load_case(FIX/'all_species');o=_launch(f,diagnostics=False)
    o['tauaer'][0,0,0,0]=cp.nextafter(o['tauaer'][0,0,0,0],cp.float32(np.inf))
    with pytest.raises(AssertionError):_assert_words(o['tauaer'],_bands(f['tauaer']))

@requires_gpu
def test_rrtmg_negative_column_is_fatal():
    import cupy as cp
    from gpuwm.core.chem_optics import rrtmg_sw_bands
    f=load_case(FIX/'rrtmg_edges')
    tau=cp.asarray(_bands(f['tauaer']));tau[1]=-0.3
    ssa=cp.asarray(_bands(f['waer']));asm=cp.asarray(_bands(f['gaer']))
    with pytest.raises(ValueError,match='negative total optical depth'):
        rrtmg_sw_bands(tau,ssa,asm)
    o=rrtmg_sw_bands(tau,ssa,asm,check_negative=False)
    assert cp.asnumpy(o['negative_tau'])[0,0]==(1<<14)-1

@requires_gpu
def test_mie_refractive_index_diagnostic_is_fatal():
    import cupy as cp
    from gpuwm.core.chem_optics import gocart_optics
    f=load_case(FIX/'all_species')
    fields=[cp.zeros((50,1,3),cp.float32) for r in ROWS]
    fields[3].fill(1.e-4)  # MSA-only case from the Fortran --badindex probe.
    rh=cp.full((50,1,3),.05,cp.float32)
    with pytest.raises(ValueError,match='invalid aerosol refractive index'):
        gocart_optics(ROWS,fields,_field(f['alt']),rh,_field(f['dz8w']))
    o=gocart_optics(ROWS,fields,_field(f['alt']),rh,_field(f['dz8w']),check_refindex=False)
    assert np.all(cp.asnumpy(o['invalid_refindex'])!=0)

@requires_gpu
def test_stored_subset_is_the_full_run_word_for_word():
    """The forecast process stores only tauaer and the 550 nm fields; a null
    output pointer must skip the store and change no other word."""
    import cupy as cp
    from gpuwm.core.chem_optics import gocart_optics
    f=load_case(FIX/'all_species')
    fields=[_field(f['chem'][:,:,:,i+1]) for i in range(19)]
    args=(ROWS,fields,_field(f['alt']),_field(f['relhum']),_field(f['dz8w']))
    full=gocart_optics(*args)
    nz,ny,nx=fields[0].shape
    held=cp.full((4,nz,ny,nx),cp.float32(7.),cp.float32)
    ext=cp.zeros((nz,ny,nx),cp.float32);aod=cp.zeros((ny,nx),cp.float32)
    part=gocart_optics(*args,outputs=('tauaer',),moments=False,
                       out={'tauaer':held,'EXTCOF55':ext,'AOD5502D':aod})
    assert part['tauaer'] is held and part['EXTCOF55'] is ext and part['AOD5502D'] is aod
    assert set(part)=={'tauaer','EXTCOF55','AOD5502D','invalid_refindex','_rows'}
    for name in ('tauaer','EXTCOF55','AOD5502D'):
        np.testing.assert_array_equal(cp.asnumpy(part[name]).view(np.uint32),
                                      cp.asnumpy(full[name]).view(np.uint32),err_msg=name)
    _assert_words(part['tauaer'],_bands(f['tauaer']))
    with pytest.raises(ValueError,match='computed from tauaer'):
        gocart_optics(*args,outputs=('waer',))
