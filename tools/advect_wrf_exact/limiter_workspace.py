"""Observe native limiter inputs while requiring unchanged total tendencies."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from types import SimpleNamespace

import numpy as np


def endbrace(source,start):
    depth=0
    for i in range(start,len(source)):
        depth+=(source[i]=='{')-(source[i]=='}')
        if depth==0:return i
    raise ValueError('unmatched brace')


def build_native(directory,wrf_source,reference_build):
    source=(wrf_source/'dyn_em/module_advect_em.F').read_text()
    marker='END SUBROUTINE advect_scalar_pd'
    position=source.index(marker)
    stores='''
  h_tendency=0.
  z_tendency=0.
  i_start=its
  i_end=min(ite,ide-1)
  j_start=jts
  j_end=min(jte,jde-1)
  if(config_flags%specified.or.config_flags%nested.or.config_flags%open_xs) i_start=max(its,ids+1)
  if(config_flags%specified.or.config_flags%nested.or.config_flags%open_xe) i_end=min(ite,ide-2)
  if(config_flags%specified.or.config_flags%nested.or.config_flags%open_ys) j_start=max(jts,jds+1)
  if(config_flags%specified.or.config_flags%nested.or.config_flags%open_ye) j_end=min(jte,jde-2)
  do j=j_start,j_end
  do k=kts,ktf
  do i=i_start,i_end
    h_tendency(i,k,j)=ph_low(i,k,j)
    z_tendency(i,k,j)=flux_out(i,k,j)
  enddo
  enddo
  enddo
'''
    observed=source[:position]+stores+source[position:]
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'module_advect_em.F'
    path.write_text(observed)
    original=reference_build
    flags=['-cpp','-ffree-form','-ffree-line-length-none','-O0','-ffp-contract=off','-fcheck=all',
           '-Dwrfmodel','-DEM_CORE=1','-DNMM_CORE=0','-DRWORDSIZE=4','-DIWORDSIZE=4','-DDWORDSIZE=8','-DLWORDSIZE=4']
    command=['gfortran','-I',str(original),'-c',*flags,str(path)]
    subprocess.run(command,cwd=directory,check=True)
    command=['gfortran','-o','run_advect',*(str(original/name) for name in
        ('stub_wrf.o','module_model_constants.o','module_wrf_error.o')),
        'module_advect_em.o',str(original/'run_advect.o')]
    subprocess.run(command,cwd=directory,check=True)
    return directory/'run_advect'


def device_observer_source():
    from gpuwm.core.kernels import module_source
    source=module_source('pd_advection')
    start=source.index('__device__\nreal pd_scale(')
    body=source.index('{',start)
    end=endbrace(source,body)
    clone=source[start:end+1].replace('real pd_scale(','real oracle_pd_scale(',1)
    body=clone.index('{')
    pos=clone.rfind(')',0,body)
    clone=clone[:pos]+', real* oracle_low, real* oracle_out'+clone[pos:]
    clone=clone.replace('    if (fo > ph_low)',
        '    oracle_low[IDX3(k,j,i)]=ph_low; oracle_out[IDX3(k,j,i)]=fo;\n    if (fo > ph_low)')
    start=source.index('void pd_renorm_apply(')
    body=source.index('{',start)
    signature=source[start:body].replace('void pd_renorm_apply(','void oracle_pd_scale_words(')
    pos=signature.rfind(')')
    signature=signature[:pos]+',real* low, real* out,real* cloned'+signature[pos:]
    call='''q0,mu_old,c1h,c2h,rdnw,fxl,fxc,fyl,fyc,fzl,fzc,msft,
        dx_inv,dy_inv,dt,ny,nx,has_msf,open_x,open_y,mub,mup_old'''
    kernel=signature+'''{
    int i=blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y,k=blockIdx.z;
    if(i>=nx||j>=ny||k>=nz)return;
    size_t h=IDX3(k,j,i);
    low[h]=0.0f;out[h]=0.0f;
    tend_out[h]=pd_scale(k,j,i,'''+call+''');
    cloned[h]=oracle_pd_scale(k,j,i,'''+call+''',low,out);
}
'''
    return source+'\n'+clone+'\nextern "C" __global__\n'+kernel


def observe(case):
    import cupy as cp
    import gpuwm.core.moist as moist
    from unittest.mock import patch
    from gpuwm.verify.advect_oracle import _initial
    a={key:cp.asarray(value) for key,value in case.inputs.items() if value.dtype==np.float32}
    coord=SimpleNamespace(**{key:a[key] for key in ('c1h','c2h','rdnw','rdn','fnm','fnp')},
                          mub2d=a['mub'],mup0=a['mu_perturbation'])
    nz,ny,nx=case.shape;m=case.metadata
    shapes=((nz,ny,nx+1),)*2+((nz,ny+1,nx),)*2+((nz+1,ny,nx),)*2
    fluxes=[cp.zeros(shape,cp.float32) for shape in shapes]
    moist.launch_pd_fluxes(a['scalar_pd'],a['q0'],a['ru'],a['rv'],a['rw'],a['muts'],coord,
        m['dx'],m['dy'],m['dt'],*fluxes,msft=a['msftx'],has_msf=True,
        open_x=m['open_x'],open_y=m['open_y'])
    actual,cloned,low,out=(cp.zeros(case.shape,cp.float32) for _ in range(4))
    captured=[]
    with patch.object(moist,'get_kernel',lambda module,name:lambda grid,block,args:captured.append((grid,block,args))):
        moist.launch_pd_renorm_apply(a['q0'],a['mu_old'],*fluxes,tend=actual,coord=coord,
          dx=m['dx'],dy=m['dy'],dt=m['dt'],msft=a['msftx'],has_msf=True,
          open_x=m['open_x'],open_y=m['open_y'])
    grid,block,args=captured[0]
    module=cp.RawModule(code=device_observer_source(),options=('-std=c++17',))
    module.get_function('oracle_pd_scale_words')(grid,block,(*args,low,out,cloned))
    if not bool(cp.array_equal(actual.view(cp.uint32),cloned.view(cp.uint32))):
        raise AssertionError('Observing limiter registers changed scale words')
    return cp.asnumpy(low),cp.asnumpy(out)


def main():
    from gpuwm.verify.advect_oracle import load_advect_cases,write_fortran_input,read_fortran_output,measure_words
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wrf-source',type=Path,required=True)
    p.add_argument('--reference-build',type=Path,required=True)
    a=p.parse_args()
    executable=build_native(a.output/'build',a.wrf_source,a.reference_build)
    results={}
    for case in load_advect_cases():
        if case.name not in ('real_west_boundary','real_steep_terrain'):continue
        inp=a.output/(case.name+'.in');out=a.output/(case.name+'.out')
        write_fortran_input(case,'advect_scalar_pd',inp)
        subprocess.run([str(executable),str(inp),str(out)],check=True)
        native=read_fortran_output(case,out,diagnostic=True)
        assert np.array_equal(native['tendency'].view(np.uint32),case.reference['advect_scalar_pd'].view(np.uint32)), 'Stores altered the native reference'
        low,flow=observe(case)
        nz,ny,nx=case.shape
        expectedlow=native['h_tendency'][:nz,4:4+ny,4:4+nx]
        expectedout=native['z_tendency'][:nz,4:4+ny,4:4+nx]
        results[case.name]={'ph_low':measure_words(low,expectedlow),'flux_out':measure_words(flow,expectedout)}
        np.savez(a.output/(case.name+'.npz'),low=low,flow=flow,expectedlow=expectedlow,expectedout=expectedout)
    (a.output/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
