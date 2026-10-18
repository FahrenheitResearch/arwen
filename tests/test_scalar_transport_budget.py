"""Independent directional flux checks for the optional shadow observer."""
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import requires_gpu


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('mapped,bounded', [(False,False),(True,False),(True,True)])
def test_directional_fluxes_and_unchanged_production(mapped,bounded):
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.core import moist
    from gpuwm.verify.scalar_process_budget import transport_kernel
    if wrf_exact.ENABLED or wrf_exact.ADVECTION_ENABLED:
        pytest.skip('observer refuses strict WRF')
    nz,ny,nx=3,4,5
    q=cp.ones((nz,ny,nx),cp.float32)
    mu=cp.full((ny,nx),100.,cp.float32)
    fx=cp.broadcast_to(2*cp.arange(nx+1,dtype=cp.float32),(nz,ny,nx+1)).copy()
    fy=cp.broadcast_to(3*cp.arange(ny+1,dtype=cp.float32)[None,:,None],(nz,ny+1,nx)).copy()
    fz=cp.broadcast_to(cp.array([0.,1.,4.,0.],cp.float32)[:,None,None],(nz+1,ny,nx)).copy()
    c1=cp.ones(nz,cp.float32); c2=cp.zeros(nz,cp.float32); rd=cp.array([-1.,-2.,-3.],cp.float32)
    coord=SimpleNamespace(c1h=c1,c2h=c2,rdnw=rd)
    m=cp.full((ny,nx),1.25,cp.float32)
    correction=[cp.zeros_like(a) for a in (fx,fy,fz)]
    arrays=(fx,correction[0],fy,correction[1],fz,correction[2])
    stock=cp.zeros_like(q)
    moist.launch_pd_renorm_apply(q,mu,*arrays,tend=stock,coord=coord,dx=2.,dy=3.,dt=.1,
                                 msft=m,has_msf=mapped,open_x=bounded,open_y=bounded)
    before=[a.copy() for a in (q,mu,*arrays,stock)]
    kernel,_=transport_kernel(); parts=cp.empty((3,nz,ny,nx),cp.float64)
    args=(q,mu,*arrays,c1,c2,rd,m,np.float32(.5),np.float32(1/3),np.float32(.1),
          parts,np.int32(nz),np.int32(ny),np.int32(nx),np.int32(mapped),np.int32(bounded),np.int32(bounded))
    kernel((1,ny,nz),(128,1,1),args)
    expected=np.empty((3,nz,ny,nx));expected[:2]=-(1.25 if mapped else 1.)
    expected[2]=np.array([1.,6.,-12.])[:,None,None]
    if bounded:
        expected[0,:,:,0]=expected[0,:,:,-1]=0
        expected[1,:,0,:]=expected[1,:,-1,:]=0
    np.testing.assert_allclose(parts.get(),expected,rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(parts.sum(axis=0).get(),stock.get(),rtol=1e-6,atol=1e-6)
    for original,saved in zip((q,mu,*arrays,stock),before):
        np.testing.assert_array_equal(original.get(),saved.get())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('axis',[0,1,2])
def test_limited_single_face_transfer(axis):
    import cupy as cp
    from gpuwm import wrf_exact
    from gpuwm.core import moist
    from gpuwm.verify.scalar_process_budget import transport_kernel
    if wrf_exact.ENABLED or wrf_exact.ADVECTION_ENABLED:
        pytest.skip('observer refuses strict WRF')
    nz,ny,nx=3,4,5
    q=cp.full((nz,ny,nx),.1,cp.float32);mu=cp.ones((ny,nx),cp.float32)
    fx=cp.zeros((nz,ny,nx+1),cp.float32)
    fy=cp.zeros((nz,ny+1,nx),cp.float32)
    fz=cp.zeros((nz+1,ny,nx),cp.float32)
    corrections=[cp.zeros_like(a) for a in (fx,fy,fz)]
    corrections[axis][1,1,1]=100.
    c1=cp.ones(nz,cp.float32);c2=cp.zeros(nz,cp.float32);rd=-c1;m=mu.copy()
    coord=SimpleNamespace(c1h=c1,c2h=c2,rdnw=rd)
    arrays=(fx,corrections[0],fy,corrections[1],fz,corrections[2])
    stock=cp.zeros_like(q)
    moist.launch_pd_renorm_apply(q,mu,*arrays,tend=stock,coord=coord,dx=1.,dy=1.,dt=1.)
    kernel,_=transport_kernel();parts=cp.empty((3,nz,ny,nx),cp.float64)
    args=(q,mu,*arrays,c1,c2,rd,m,*[np.float32(1.)]*3,parts,
          np.int32(nz),np.int32(ny),np.int32(nx),*[np.int32(0)]*3)
    kernel((1,ny,nz),(128,1,1),args)
    expected=np.zeros((3,nz,ny,nx))
    donor,receiver=[((1,1,0),(1,1,1)),((1,0,1),(1,1,1)),((1,1,1),(0,1,1))][axis]
    expected[(axis,)+donor]=-.1;expected[(axis,)+receiver]=.1
    np.testing.assert_allclose(parts.get(),expected,rtol=2e-6,atol=1e-8)
    np.testing.assert_allclose(parts.sum(axis=0).get(),stock.get(),rtol=2e-6,atol=1e-8)
