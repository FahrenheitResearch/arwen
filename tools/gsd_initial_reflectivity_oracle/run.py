"""Extract the pinned initial helper and its unchanged calc_refl10cm oracle.

Usage: python tools/gsd_initial_reflectivity_oracle/run.py SOURCE_DIR OUTPUT_DIR
Desktop WSL Ubuntu-24.04/gfortran. No model/diagnostic arithmetic is rewritten.
"""
from pathlib import Path
import hashlib
import json
import re
import shlex
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
texts = {name: (source/name).read_text() for name in
         ('module_diag_gsd.F', 'module_mp_thompson.F', 'module_mp_radar.F', 'module_model_constants.F')}
PINNED = {
    'module_model_constants.F': '0d0ee7d243fe673288efafcdcdc470ee028ad8d083648f12225ce2d168f8e87c',
    'module_diag_gsd.F': 'c5771bcd5bf9cefd642e79832453cb8f2a951471d81351850596cc761c53dd64',
    'module_mp_thompson.F': '4d60011188443eb432294f7693beb64bdbc8f812541a15c7800013060c877283',
    'module_mp_radar.F': '08329c87604b234efab53f7986163a4f6050f58823da9099cb464163c1920f08',
}
for name, digest in PINNED.items():
    actual = hashlib.sha256((source/name).read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f'{name}: expected pinned {digest}, got {actual}')

def routine(text, name, kind='SUBROUTINE'):
    match = re.search(r'(?im)^ *' + (r'REAL ' if kind == 'FUNCTION' else '') + kind + r' '+name+r'\b.*?^ *END '+kind+r' '+name+r'\b[^\n]*', text, re.S)
    if match is None:
        raise ValueError(name)
    return match.group()+'\n'
t = texts['module_mp_thompson.F']
# The complete declarations and constant/gamma statements needed by the
# radar routine are extracted unchanged. Microphysics lookup tables are
# never read by calc_refl10cm and are not built by this isolated harness.
header = t[:t.index('      CONTAINS')]
constants = t[t.index('      Sc3 = Sc**'):t.index('!+---+-----------------------------------------------------------------+\n!..Simplify various rate')]
radar_constants = t[t.index('      xam_r = am_r'):t.index('      call radar_init')+len('      call radar_init')]
module = header+'\nCONTAINS\nsubroutine oracle_initialize\ninteger :: n\n'+constants+radar_constants+'\nend subroutine\n'
for name, kind in [('GAMMLN','FUNCTION'),('WGAMMA','FUNCTION'),('calc_refl10cm','SUBROUTINE')]:
    module += routine(t,name,kind)
module += '\nEND MODULE module_mp_thompson\n'
(out/'module_mp_thompson_extract.F').write_text(module,newline='\n')
helper = routine(texts['module_diag_gsd.F'],'gsd_refl_first_time_step')
(out/'helper.f90').write_text('module helper\ncontains\n'+helper+'\nend module\n',newline='\n')
for name in ('module_mp_radar.F','module_model_constants.F'):
    (out/name).write_bytes((source/name).read_bytes())
stubs = (ROOT/'tools/thompson_fork_oracle/stub_wrf.F90').read_text()
stubs += '''
module module_state_description
integer, parameter :: THOMPSON=8, THOMPSONAERO=28
end module
subroutine wrf_error_fatal(message)
character(*) :: message
print *,message
error stop 1
end subroutine
'''
(out/'stubs.f90').write_text(stubs,newline='\n')
nx,ny,nz=8,3,9
names=('th','pii','p','rho','qv','qc','nr','qr','qs','qg')
program='''program oracle
use helper
use module_model_constants, only: rcp,p1000mb
use module_mp_thompson, only: oracle_initialize
implicit none
integer :: s,mp,n,mode
'''+''.join(f'real :: {name}({nx},{nz},{ny})\n' for name in (*names,'refl'))
program+='call oracle_initialize\nread(*,*) n\ndo s=1,n\nread(*,*) mode,mp\n'
program+=''.join('read(*,*) '+name+'\n' for name in names)
# Native phy_prep expressions, applied only to the state-input vectors.
program+='if(mode == 1) then\nth=th+pii\npii=(p/p1000mb)**rcp\nrho=1./rho*(1.+qv)\nendif\n'
program+=f'call gsd_refl_first_time_step(1,{nx},1,{ny},1,{nz},1,{nz},[1],[{nx}],[1],[{ny}],1,mp,th,pii,p,rho,qv,qc,nr,qr,qs,qg,refl)\n'
program+=f"print '({nx*ny*nz}(Z8.8,1X))',transfer(refl,[0],{nx*ny*nz})\nenddo\nend program\n"
(out/'main.f90').write_text(program,newline='\n')
cell=np.arange(nx*ny*nz).reshape(nz,ny,nx)
k=np.broadcast_to(np.arange(nz,dtype=np.float32)[:,None,None],cell.shape)
c=np.broadcast_to(np.arange(nx*ny,dtype=np.float32).reshape(ny,nx),cell.shape)
# Rain-number and mass threshold cases recur throughout warm, mixed and
# cold columns. Positive supplied numbers must remain authoritative.
qr_values=np.array([0.,np.nextafter(np.float32(1e-12),np.float32(0)),1e-12,1e-10,1e-7,1e-5,1e-3,3e-3],np.float32)
nr_values=np.array([0.,-1.,1.,1e3,1e6,0.,-100.,1e5],np.float32)
data=[]
for phase in range(4):
    th=np.asarray(305.+c*.125+k*.25,np.float32)
    temperature=np.asarray(292.-k*5.+(c%3-1)*phase*5,np.float32)
    pii=np.asarray(temperature/th,np.float32)
    p=np.asarray(100000.-k*7000.-c*10,np.float32)
    rho=np.asarray(.622*p/(287.04*temperature*(.01+.622)),np.float32)
    qv=np.asarray(.01+k*.0001+c*.00001,np.float32)
    qc=np.full(cell.shape,1e-4,np.float32)
    nr=np.broadcast_to(nr_values[(np.arange(nx)+phase)%nx][None,None,:],cell.shape).copy()
    qr=np.broadcast_to(qr_values[np.arange(nx)][None,None,:],cell.shape).copy()
    qs=np.asarray(np.where((c+phase)%3==0,1e-3*(k+1)/9.,0),np.float32)
    qg=np.asarray(np.where((c+phase)%3==1,3e-4*(k+1)/9.,0),np.float32)
    data.append(np.array([th,pii,p,rho,qv,qc,nr,qr,qs,qg],np.float32))
data=np.array(data,np.float32)
state_data=data.copy()
for phase, values in enumerate(state_data):
    values[0]=305.+k*.25
    values[1]=c*.125+phase*.25
    values[3]=(1.+values[4])/data[phase,3]
data=np.stack((data,state_data))
lines=['16']
for mode, cases in enumerate(data):
    for values in cases:
        for mp in (8,28):
            lines.append(f'{mode} {mp}')
            lines.extend(' '.join(format(float(x),'.9g') for x in a.transpose(1,0,2).ravel()) for a in values)
payload='\n'.join(lines)+'\n'
(out/'inputs.txt').write_text(payload)
posix='/mnt/'+out.resolve().drive[0].lower()+out.resolve().as_posix()[2:]
cmd='cd '+shlex.quote(posix)+' && gfortran -cpp -DWRF_CHEM=0 -ffree-form -ffree-line-length-none -O2 -ffp-contract=off -fno-tree-vectorize stubs.f90 module_model_constants.F module_mp_radar.F module_mp_thompson_extract.F helper.f90 main.f90 -o oracle && ./oracle < inputs.txt'
r=subprocess.run(['wsl','-d','Ubuntu-24.04','--','bash','-lc',cmd],capture_output=True,text=True)
if r.returncode:
    print(r.stderr);raise SystemExit(r.returncode)
(out/'words.txt').write_text(r.stdout)
words=np.array([int(x,16) for line in r.stdout.splitlines() if len(line.split())==nx*ny*nz for x in line.split()],np.uint32)
assert words.size==16*nx*ny*nz,(words.size,r.stdout)
expected=words.reshape(2,4,2,ny,nz,nx).transpose(0,1,2,4,3,5)
np.savez_compressed(out/'vectors.npz',inputs=data,expected=expected,names=np.array(names))
receipt={'authority':'NOAA-EMC/HRRR 40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827',
 'source_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in texts},
 'helper_sha256':hashlib.sha256(helper.encode()).hexdigest(),
 'calc_refl10cm_sha256':hashlib.sha256(routine(t,'calc_refl10cm').encode()).hexdigest(),
 'compiler':subprocess.check_output(['wsl','-d','Ubuntu-24.04','--','gfortran','--version'],text=True).splitlines()[0],
 'flags':'-O2 -ffp-contract=off -fno-tree-vectorize','calls':16,'columns_per_call':nx*ny,
 'levels':nz,'compared_float32_words':int(words.size)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
