"""WRF v4.7.1 chem_prep identity; no local GPU is required."""
from types import SimpleNamespace
import numpy as np
import pytest
from gpuwm.core.chem_prep import ChemPrep
from gpuwm.verify.chem_prep_ref import chem_prep_reference
from gpuwm.verify.chem_oracle import ORACLE_ROOT,load,ulp_table

CASES=load(ORACLE_ROOT/'core'/'chem_prep')
OUTPUTS=('p_phy','t_phy','rho','u_phy','v_phy','z_at_w','dz8w','z','rh','p8w','t8w')


def _state(a):
    def cv(key):return np.ascontiguousarray(a[key].transpose(1,2,0))
    return SimpleNamespace(p=cv('p')[:-1]+cv('pb')[:-1],
        thp=cv('t')[:-1],thb=np.full(a['t'].shape[1]-1,300,dtype=np.float32),
        alt=cv('alt')[:-1],php=cv('ph'),phb=cv('phb'),
        u=cv('u')[:-1],v=cv('v')[:-1],qv=cv('qv')[:-1],
        qc=None,qr=None,fnm=a['fnm'][:-1],fnp=a['fnp'][:-1])


def _expected(a,key):
    result=np.ascontiguousarray(a[key].transpose(1,2,0))
    return result if key in ('z_at_w','p8w','t8w') else result[:-1]


def _tile(s):
    """Six identical columns with alternating staggered wind faces."""
    nz=s.p.shape[0]
    for key in ("p","thp","alt","qv","php"):
        setattr(s,key,np.tile(getattr(s,key),(1,2,3)))
    s.phb=s.phb[:,0,0].copy()
    u=s.u.copy();v=s.v.copy()
    s.u=np.tile(u,(1,2,2))
    s.v=np.concatenate((v,v[:,:1,:]),axis=1)
    s.v=np.tile(s.v,(1,1,3))
    return s


def _identity(out,a):
    for key in OUTPUTS:
        reference=_expected(a,key)
        reference=np.tile(reference,(1,out[key].shape[1],out[key].shape[2]))
        measured=ulp_table(out[key],reference)
        assert measured=={'max_ulp':0,'n_nonzero':0,'n':reference.size},(key,measured)
        np.testing.assert_array_equal(out[key].view(np.uint32),reference.view(np.uint32))


@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('tiled',(False,True))
def test_cpu_oracle(case,tiled):
    a=CASES[case];s=_state(a)
    if tiled:s=_tile(s)
    out=chem_prep_reference(s.p,s.thp+s.thb[:,None,None],s.alt,s.php,s.phb[:,None,None] if s.phb.ndim==1 else s.phb,
                            s.u,s.v,s.qv,s.fnm,s.fnp)
    _identity(out,a)
    np.testing.assert_array_equal(out['dryrho'],np.float32(1)/s.alt)
    with pytest.raises(TypeError, match="host pointers"):
        ChemPrep(s)


@pytest.mark.gpu
@pytest.mark.parametrize('case',CASES)
@pytest.mark.parametrize('tiled',(False,True))
def test_gpu_oracle(case,tiled):
    import cupy as cp
    s=_state(CASES[case])
    if tiled:s=_tile(s)
    for key,value in vars(s).items():
        if isinstance(value,np.ndarray):setattr(s,key,cp.asarray(value))
    prep=ChemPrep(s)
    _identity({key:cp.asnumpy(prep.get(key)) for key in OUTPUTS},CASES[case])
    addresses={key:prep.get(key).data.ptr for key in OUTPUTS}
    prep.refresh(s)
    assert addresses=={key:prep.get(key).data.ptr for key in OUTPUTS}
