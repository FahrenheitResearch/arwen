"""WRF 4.6.1 SW composition with the oracle's aer_opt=2 optics.

This grades the composition after wrapper preparation. It does not grade
wrapper preparation or a forecast trajectory.
"""
from pathlib import Path
import hashlib,json,re,subprocess,sys
import numpy as np
tree=Path(__file__).resolve().parents[2]
source_dir,build,output=map(lambda p:Path(p).resolve(),sys.argv[1:])
sys.path.insert(0,str(tree))
from gpuwm.core import rrtmg_sw as sw
from tools.aer_opt2_wrf461_oracle.validate import read_columns
source=source_dir/'module_ra_rrtmg_sw.F'
pins={name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in ('module_ra_rrtmg_sw.F','module_ra_rrtmg_lw.F')}

assert pins == {'module_ra_rrtmg_sw.F':'7f8af1da0ca1d25ce784a917bc68600300a7c569881c57e7de8501cd53496b59',
                'module_ra_rrtmg_lw.F':'97950adb65a67ab8bb460cbd2efade8a225e655ba48fd4cc5227569aa9764088'}, 'WRF SW/LW reference bytes differ'
build.mkdir(parents=True,exist_ok=True)
text = source.read_text()
prefix = text[:re.search(r'^\s*module module_ra_rrtmg_sw\s*$', text, re.I | re.M).start()]
lw = (source_dir / 'module_ra_rrtmg_lw.F').read_text()
random_end = re.search(r'^\s*end module mcica_random_numbers\s*$', lw, re.I | re.M).end()
support = "subroutine wrf_error_fatal(message)\ncharacter(len=*) :: message\nprint *, message\nstop 1\nend subroutine\nsubroutine wrf_message(message)\ncharacter(len=*) :: message\nend subroutine\n"
(build / 'fork.F90').write_text(lw[:random_end] + '\n' + prefix + '\n' + support)
dump = dict(np.load(tree / 'tools/rrtmg_wrf461_oracle/sw_fixtures/sw_tables.npz'))
print('table count:', len(dump), flush=True)
uses, reads = [], []
mapping = {'con':'rrsw_con', 'ref':'rrsw_ref', 'tbl':'rrsw_tbl', 'cld':'rrsw_cld', 'wvn':'rrsw_wvn', 'aer':'rrsw_aer', 'parrrsw':'parrrsw'}
with (build / 'tables.bin').open('wb') as out:
    for i, (key, value) in enumerate(dump.items()):
        if '/' not in key: continue
        group, name = key.split('/')
        mod = 'rrsw_' + group if group.startswith('kg') else mapping[group]
        uses.append(f'use {mod}, only: a{i} => {name}')
        if key in ('tbl/od_lo', 'tbl/tblint', 'parrrsw/rrsw_scon'):
            reads.append(f'read(10) pinned_value; if (a{i} /= pinned_value) stop 9')
        else:
            reads.append(f'read(10) a{i}')
        out.write(np.asarray(value, dtype='<i4' if value.dtype.kind in 'iu' else '<f4').tobytes(order='F'))
cols = ['play','plev','tlay','tlev','h2ovmr','o3vmr','co2vmr','ch4vmr','n2ovmr','o2vmr','reicmcl','relqmcl','resnmcl']
scalars = ['tsfc','asdir','asdif','aldir','aldif','coszen','adjes','scon']
mc = ['cldfmcl','taucmcl','ssacmcl','asmcmcl','fsfcmcl','ciwpmcl','clwpmcl','cswpmcl']
opt = ['tauaer','ssaaer','asmaer']
outputs = ['swuflx','swdflx','swhr','swuflxc','swdflxc','swhrc','sibvisdir','sibvisdif','sibnirdir','sibnirdif','swdkdir','swdkdif','swdkdirc']
decl, allocate, reading = [], [], []
for name in cols + outputs + ['swuflxcln','swdflxcln','sw_zbbcddir','sw_dirdflux','sw_difdflux']:
    layer = 'nlay+1' if name in ('plev','tlev') or name in (outputs+['swuflxcln','swdflxcln','sw_zbbcddir','sw_dirdflux','sw_difdflux']) and name not in ('swhr','swhrc') else 'nlay'
    decl.append(f'real, allocatable :: {name}(:,:)')
    allocate.append(f'allocate({name}(1,{layer}))')
    if name in cols: reading.append(f'read(11) {name}')
for name in scalars:
    if name in ('scon','adjes'):
        decl.append(f'real :: {name}')
    else:
        decl.append(f'real :: {name}(1)')
    reading.append(f'read(11) {name}')
for name in mc + opt:
    shape = '(112,1,nlay)' if name in mc else '(1,nlay,14)'
    decl.append(f'real, allocatable :: {name}(:,:,:)')
    allocate.append(f'allocate({name}{shape})')
    reading.append(f'read(11) {name}')
program = '\n'.join(['program flux_oracle', 'use rrtmg_sw_rad, only: rrtmg_sw', *uses, 'implicit none',
 'integer :: nlay, icld, inflgsw, iceflgsw, liqflgsw, dyofyr', 'real :: pinned_value', *decl,
 'real, allocatable :: ecaer(:,:,:)',
 "open(10,file='tables.bin',access='stream',form='unformatted',status='old')", *reads, 'close(10)',
 "open(11,file='input.bin',access='stream',form='unformatted',status='old')",
 'read(11) nlay,icld,inflgsw,iceflgsw,liqflgsw,dyofyr', *allocate,
 'allocate(ecaer(1,nlay,6)); ecaer=0.', *reading, 'close(11)',
 'call rrtmg_sw(1,nlay,icld,play,plev,tlay,tlev,tsfc,h2ovmr,o3vmr,co2vmr,ch4vmr,n2ovmr,o2vmr, &',
 'asdir,asdif,aldir,aldif,coszen,adjes,dyofyr,scon,inflgsw,iceflgsw,liqflgsw, &',
 'cldfmcl,taucmcl,ssacmcl,asmcmcl,fsfcmcl,ciwpmcl,clwpmcl,cswpmcl,reicmcl,relqmcl,resnmcl, &',
 'tauaer,ssaaer,asmaer,ecaer,swuflx,swdflx,swhr,swuflxc,swdflxc,swhrc,swuflxcln,swdflxcln,2, &',
 'sibvisdir,sibvisdif,sibnirdir,sibnirdif,swdkdir,swdkdif,swdkdirc,0,sw_zbbcddir(1,1),sw_dirdflux(1,1),sw_difdflux(1,1))',
 "open(12,file='output.bin',access='stream',form='unformatted',status='replace')",
 'write(12) ' + ','.join(outputs), 'close(12)', 'end program'])
(build / 'main.f90').write_text(program)
flags=['-DRWORDSIZE=4','-DEM_CORE=1','-DWRF_CHEM=0','-O0','-ffp-contract=off','-cpp','-ffree-form','-ffree-line-length-none','-ffunction-sections','-fdata-sections']
subprocess.run(['gfortran',*flags,'-J',str(build),str(build/'fork.F90'),str(build/'main.f90'),'-Wl,--gc-sections','-o',str(build/'oracle')],cwd=build,check=True)

base=dict(np.load(tree/'tools/rrtmg_wrf461_oracle/sw_fixtures/fixtures_real.npz'))
entry=lambda key:base['c00001/entry/'+key]
layers=np.r_[np.arange(8),int(entry('nlay'))-1]
interfaces=np.r_[np.arange(9),int(entry('nlay'))]
receipt={'sources':pins,'flags':flags,'columns':96,'scope':'composition after WRF wrapper preparation',
 'coefficient_sha256':hashlib.sha256((tree/'tools/rrtmg_wrf461_oracle/sw_fixtures/sw_tables.npz').read_bytes()).hexdigest(),
 'compiler':subprocess.run(['gfortran','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]}
fixture={}
for case in read_columns(tree/'tools/aer_opt2_wrf461_oracle/fixtures/columns.bin'):
    if case['scheme']!='rrtmg':continue
    c=f"a{case['column']:03d}"
    nlay=9
    entries={}
    for name in cols:
        entries[name]=entry(name)[interfaces if name in ('plev','tlev') else layers].copy()
    for name in mc:entries[name]=entry(name)[:,layers].copy()
    for name in scalars:entries[name]=np.asarray(entry(name)).copy()
    for name in ('icld','inflgsw','iceflgsw','liqflgsw','dyofyr'):entries[name]=entry(name)
    entries['nlay']=np.asarray(nlay,np.int32)
    tau=np.zeros((nlay,14),np.float32);ssa=np.ones_like(tau);asy=np.zeros_like(tau)
    tau[:8],ssa[:8],asy[:8]=case['outputs'].transpose(0,2,1)
    with (build/'input.bin').open('wb') as out:
        out.write(np.asarray([nlay,int(entries['icld']),int(entries['inflgsw']),int(entries['iceflgsw']),int(entries['liqflgsw']),int(entries['dyofyr'])],'<i4').tobytes())
        for name in cols+scalars+mc:out.write(np.asarray(entries[name],'<f4').tobytes(order='F'))
        for v in (tau,ssa,asy):out.write(v.tobytes(order='F'))
    subprocess.run([str(build/'oracle')],cwd=build,check=True)
    raw=np.fromfile(build/'output.bin',dtype='<f4')
    offset=0
    for name in outputs:
        count=nlay if name in ('swhr','swhrc') else nlay+1
        fixture[f'{c}/out/{name}']=raw[offset:offset+count].copy();offset+=count
    assert offset==len(raw)
    for name,value in entries.items():fixture[f'{c}/entry/{name}']=value
    for name,v in zip(('tau','ssa','asy'),(tau,ssa,asy)):fixture[f'{c}/in/{name}']=v
np.savez_compressed(output,**fixture,receipt=json.dumps(receipt))
(build/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
