"""Exercise the captured-column protocol through real Fortran executables."""
from pathlib import Path
import csv
import importlib.util
import json
import shutil
import subprocess

import numpy as np
import pytest

from gpuwm.core.ruc import (RUC_DRIVER_COLUMN_STATE, RUC_DRIVER_COLUMN_FORCING,
                           RUC_DRIVER_PROFILE_STATE, RUC_DRIVER_ARW_FORCING)


TOOL = Path(__file__).resolve().parents[1] / 'tools/ruc_hrrr_fork_oracle'
COLUMNS = RUC_DRIVER_COLUMN_STATE + RUC_DRIVER_COLUMN_FORCING + RUC_DRIVER_ARW_FORCING
PROFILES = RUC_DRIVER_PROFILE_STATE
NAMES = COLUMNS + PROFILES + ('ivgtyp','isltyp','zs','landusef','soilctop')
NCOL, NSOIL, NLCAT, NSCAT = 3,9,21,19


def _generator():
    spec=importlib.util.spec_from_file_location('ruc_column_generator',TOOL/'generate_real_columns.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate


def _fixture(directory):
    arrays={}
    for index,name in enumerate(COLUMNS):
        arrays['input__'+name]=(np.arange(NCOL,dtype=np.float32)+np.float32(10*index+.25))
    for index,name in enumerate(PROFILES):
        arrays['input__'+name]=(np.arange(NSOIL,dtype=np.float32)[:,None]*np.float32(100)
                                +np.arange(NCOL,dtype=np.float32)[None,:]
                                +np.float32(1000*(index+1)+.25))
    arrays['keyword__ivgtyp']=np.array([1,10,12],dtype=np.int32)
    arrays['keyword__isltyp']=np.array([4,6,3],dtype=np.int32)
    arrays['keyword__landusef']=(np.arange(NLCAT*NCOL,dtype=np.float32).reshape(NLCAT,NCOL)+.125)
    arrays['keyword__soilctop']=(np.arange(NSCAT*NCOL,dtype=np.float32).reshape(NSCAT,NCOL)+.375)
    arrays['zs']=np.array([0.,.01,.04,.10,.30,.60,1.,1.60,3.],dtype=np.float32)
    path=directory/'capture.npz'
    np.savez_compressed(path,**arrays)
    path.with_suffix('.json').write_text(json.dumps({'columns':NCOL,'driver_keywords':{
        'dt':20.,'ktau':1,'mminlu':'MODIFIED_IGBP_MODIS_NOAH','rdlai2d':True,
        'mosaic_lu':1,'mosaic_soil':1,'em_core':1,'xice_threshold':.02}}))
    return path,arrays


def _compiler():
    compiler=shutil.which('gfortran')
    if compiler is None:
        pytest.skip('the protocol executable check requires GNU Fortran')
    return compiler


def _declare(name):
    kind='integer' if name in ('ivgtyp','isltyp') else 'real'
    if name in PROFILES or name in ('z3d','p8w','t3d','qv3d','qc3d','rho3d'):
        shape='(NCOL,NSOIL,1)'
    elif name=='landusef':
        shape='(NCOL,NLCAT,1)'
    elif name=='soilctop':
        shape='(NCOL,NSCAT,1)'
    elif name=='zs':
        shape='(NSOIL)'
    else:
        shape='(NCOL,1)'
    return f'  {kind} :: {name}{shape}'


def _dump_arguments():
    lines=["  open(newunit=u,file='lsmruc-arguments.csv',status='replace',action='write')",
           "  write(u,'(A)') 'field,column,level,value'"]
    for name in NAMES:
        if name=='zs':
            lines += ['  do k=1,NSOIL',f"    write(u,'(*(g0,:,\",\"))') '{name}',0,k,{name}(k)",'  end do']
            continue
        count=(NSOIL if name in PROFILES or name in ('z3d','p8w','t3d','qv3d','qc3d','rho3d')
               else NLCAT if name=='landusef' else NSCAT if name=='soilctop' else 1)
        value=(f'{name}(i,k,1)' if count>1 else f'{name}(i,1)')
        lines += ['  do i=1,NCOL',f'    do k=1,{count}',
                  f"      write(u,'(*(g0,:,\",\"))') '{name}',i,k,{value}",'    end do','  end do']
    lines += ['  close(u)']
    return '\n'.join(lines)


def _expected(arrays,name,column,level):
    if name=='zs':
        return arrays['zs'][level-1]
    prefix='keyword__' if name in ('ivgtyp','isltyp','landusef','soilctop') else 'input__'
    value=arrays[prefix+name]
    result=value[level-1,column-1] if value.ndim==2 else value[column-1]
    return np.float32(result)*np.float32(.001) if name=='snowfallac' else result


def _assert_dump(path,arrays):
    rows=list(csv.DictReader(path.open()))
    assert {row['field'] for row in rows}==set(NAMES)
    for row in rows:
        expected=_expected(arrays,row['field'],int(row['column']),int(row['level']))
        actual=np.float32(row['value'])
        assert actual.view(np.uint32)==np.float32(expected).view(np.uint32),row


def test_binary_columns_are_read_at_the_expected_fortran_indices(tmp_path):
    compiler=_compiler()
    snapshot,arrays=_fixture(tmp_path)
    output=tmp_path/'columns';_generator()(snapshot,output)
    source='''program read_captured_columns
  implicit none
  integer,parameter :: NCOL=3,NSOIL=9,NLCAT=21,NSCAT=19
  integer :: i,k,u
'''+ '\n'.join(_declare(name) for name in NAMES)+'''
  open(unit=11,file='inputs.bin',access='stream',form='unformatted',status='old')
'''+ '\n'.join('  read(11) '+name for name in NAMES)+'''
  close(11)
'''+_dump_arguments()+'''
end program read_captured_columns
'''
    (output/'reader.F90').write_text(source)
    subprocess.run([compiler,'-O0','-ffree-line-length-none','reader.F90','-o','reader'],cwd=output,check=True)
    subprocess.run([str(output/'reader')],cwd=output,check=True)
    _assert_dump(output/'lsmruc-arguments.csv',arrays)


def _module_stub():
    init_dimensions='ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,sms,sme,its,ite,jts,jte,sts,ste'
    run_dimensions='ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte'
    signature='''spp_lsm,pattern_spp_lsm,field_sf,dt,ktau,nsl, &
 lakemodel,lakemask,graupelncv,snowncv,rainncv,zs,rainbl,snow,snowh,snowc, &
 frzfrac,frpcpn,rhosnf,precipfr,z3d,p8w,t3d,qv3d,qc3d,rho3d,glw,gsw, &
 emiss,chklowq,chs,flqc,flhc,mavail,canwat,vegfra,alb,znt,z0,snoalb,albbck,lai, &
 mminlu,landusef,nlcat,mosaic_lu,mosaic_soil,soilctop,nscat,qsfc,qsg,qvg,qcg, &
 dew,soilt1,tsnav,tbot,ivgtyp,isltyp,xland,iswater,isice,xice,xice_threshold, &
 cp0,rovcp0,g0,lv,stb,soilmois,sh2o,smavail,smmax,tso,soilt,hfx,qfx,lh, &
 sfcrunoff,udrunoff,acrunoff,sfcexc,sfcevp,grdflx,snowfallac,acsnow,snom, &
 smfr3d,keepfr3dflag,myj,shdmin,shdmax,rdlai2d, &
 '''+run_dimensions
    return '''module module_sf_ruclsm
 implicit none
 integer,parameter :: NCOL=3,NSOIL=9,LANDCATS=21,SOILCATS=19
 contains
 subroutine ruclsminit(sh2o,smfr3d,tso,soilmois,isltyp,ivgtyp,mminlu,xice, &
 mavail,nsl,iswater,isice,znt,restart,allowed,'''+init_dimensions+''')
 integer :: nsl,iswater,isice,'''+init_dimensions+'''
 real :: sh2o(NCOL,NSOIL,1),smfr3d(NCOL,NSOIL,1),tso(NCOL,NSOIL,1),soilmois(NCOL,NSOIL,1)
 integer :: isltyp(NCOL,1),ivgtyp(NCOL,1)
 real :: xice(NCOL,1),mavail(NCOL,1),znt(NCOL,1)
 character(len=*) :: mminlu
 logical :: restart,allowed
 sh2o=-999.0
 smfr3d=-999.0
 tso=-999.0
 soilmois=-999.0
 isltyp=-999
 ivgtyp=-999
 xice=-999.0
 mavail=-999.0
 znt=-999.0
 end subroutine ruclsminit
 subroutine lsmruc('''+signature+''')
 integer :: spp_lsm,ktau,nsl,lakemodel,nlcat,nscat,mosaic_lu,mosaic_soil,iswater,isice
 integer :: '''+run_dimensions+'''
 real :: dt,xice_threshold,cp0,rovcp0,g0,lv,stb
 real :: pattern_spp_lsm(NCOL,NSOIL,1),field_sf(NCOL,NSOIL,1)
 character(len=*) :: mminlu
 logical :: frpcpn,myj,rdlai2d
 integer :: i,k,u
'''+ '\n'.join(_declare(name).replace('NLCAT','LANDCATS').replace('NSCAT','SOILCATS') for name in NAMES)+'''
'''+_dump_arguments().replace('NLCAT','LANDCATS').replace('NSCAT','SOILCATS')+'''
 end subroutine lsmruc
end module module_sf_ruclsm
'''


def test_driver_restores_captured_state_after_table_initialization(tmp_path):
    compiler=_compiler()
    snapshot,arrays=_fixture(tmp_path)
    output=tmp_path/'columns';_generator()(snapshot,output)
    (output/'stub.F90').write_text(_module_stub())
    def execute(source,exe):
        subprocess.run([compiler,'-O0','-ffree-line-length-none','stub.F90',source,'-o',exe],cwd=output,check=True)
        subprocess.run([str(output/exe)],cwd=output,check=True)
    execute('run_real.F90','restore')
    _assert_dump(output/'lsmruc-arguments.csv',arrays)
    # Removing the restore must expose the deliberate initializer damage.
    source=(output/'run_real.F90').read_text()
    start=source.rindex("  open(unit=11,file='inputs.bin'")
    end=source.index('  close(11)',start)+len('  close(11)')
    (output/'without_restore.F90').write_text(source[:start]+source[end:])
    execute('without_restore.F90','no_restore')
    with pytest.raises(AssertionError):
        _assert_dump(output/'lsmruc-arguments.csv',arrays)
