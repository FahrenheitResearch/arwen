"""Cut nonzero-aerosol flux fixtures with the pinned fork's RRTMG SW.

Usage: python run_sw_flux_oracle.py SOURCE_DIRECTORY BUILD_DIRECTORY OUTPUT.npz

The composition and its helpers are compiled verbatim from the operational
fork. The caller supplies the existing post-initialization coefficient deck
and McICA columns, so this checks the flux and heating arithmetic given those
inputs. It does not qualify the fork's initialization or the WRF wrapper.
The aerosol producer has its separate verbatim-Fortran optics fixture.
"""
from pathlib import Path
import hashlib, json, re, subprocess, sys
from urllib.request import urlopen
import numpy as np

tree = Path(__file__).resolve().parents[2]
source_dir, build, output = map(lambda p: Path(p).resolve(), sys.argv[1:])
sys.path.insert(0, str(tree))
from gpuwm.core import rrtmg_sw as sw
source = source_dir / 'module_ra_rrtmg_sw.F'
pins = {'module_ra_rrtmg_sw.F': '04e97b7e3cea6984979fd4b59ea0293d3ed15e2ecd2ba7a7dd81979bd86f3f39',
        'module_ra_rrtmg_lw.F': '688fe4f80cc500a8825529c39dcd3d1ecc245e45e406736f61f252dec04407e6'}
for name, digest in pins.items():
    path = source_dir / name
    if not path.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        url = ('https://raw.githubusercontent.com/NOAA-EMC/HRRR/'
               '40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827/'
               'sorc/hrrr_wrfarw.fd/WRFV3.9/phys/' + name)
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
build.mkdir(parents=True, exist_ok=True)
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
outputs = ['swuflx','swdflx','swhr','swuflxc','swdflxc','swhrc','sibvisdir','sibvisdif','sibnirdir','sibnirdif','swdkdir','swdkdif','swdkdirc','swdkdifc']
decl, allocate, reading = [], [], []
for name in cols + outputs:
    layer = 'nlay+1' if name in ('plev','tlev') or name in outputs and name not in ('swhr','swhrc') else 'nlay'
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
 'tauaer,ssaaer,asmaer,ecaer,swuflx,swdflx,swhr,swuflxc,swdflxc,swhrc,3, &',
 'sibvisdir,sibvisdif,sibnirdir,sibnirdif,swdkdir,swdkdif,swdkdirc,swdkdifc)',
 "open(12,file='output.bin',access='stream',form='unformatted',status='replace')",
 'write(12) ' + ','.join(outputs), 'close(12)', 'end program'])
(build / 'main.f90').write_text(program)
flags=['-O0','-ffp-contract=off','-cpp','-ffree-form','-ffree-line-length-none','-ffunction-sections','-fdata-sections']
subprocess.run(['gfortran',*flags,'-J',str(build),str(build/'fork.F90'),str(build/'main.f90'),'-Wl,--gc-sections','-o',str(build/'oracle')],cwd=build,check=True)
fix = {}
for file in ('fixtures_real.npz','fixtures_synth.npz','fixtures_tall.npz'):
    fix.update(dict(np.load(tree / 'tools/rrtmg_wrf461_oracle/sw_fixtures' / file)))
cases=sorted(k.split('/')[0] for k in fix if k.endswith('/night') and int(fix[k]) == 0)
tables=sw.tables_from_dump(dump)
receipt={'commit':'40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827', 'tag':'v4.1.21',
 'sources':pins, 'flags':flags, 'cases':[],
 'coefficient_sha256':hashlib.sha256((tree / 'tools/rrtmg_wrf461_oracle/sw_fixtures/sw_tables.npz').read_bytes()).hexdigest(),
 'extract_sha256':hashlib.sha256((build / 'fork.F90').read_bytes()).hexdigest(),
 'program_sha256':hashlib.sha256((build / 'main.f90').read_bytes()).hexdigest(),
 'compiler':subprocess.run(['gfortran','--version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]}
fixture={}
for ci,c in enumerate(cases):
    e=lambda key: fix[f'{c}/entry/{key}']
    nlay=int(e('nlay'))
    rng=np.random.default_rng(710+ci)
    tau=np.zeros((nlay,14),np.float32); ssa=np.ones_like(tau); asy=np.zeros_like(tau)
    tau[:-1]=rng.uniform(0.0,0.08,(nlay-1,14)).astype(np.float32)
    ssa[:-1]=rng.uniform(0.55,0.999,(nlay-1,14)).astype(np.float32)
    asy[:-1]=rng.uniform(0.6,0.82,(nlay-1,14)).astype(np.float32)
    with (build/'input.bin').open('wb') as out:
        out.write(np.array([nlay,int(e('icld')),int(e('inflgsw')),int(e('iceflgsw')),int(e('liqflgsw')),int(e('dyofyr'))],'<i4').tobytes())
        for name in cols + scalars + mc:
            out.write(np.asarray(e(name),'<f4').tobytes(order='F'))
        for value in (tau,ssa,asy): out.write(value.tobytes(order='F'))
    subprocess.run([str(build/'oracle')],cwd=build,check=True)
    raw=np.fromfile(build/'output.bin',dtype='<f4')
    ref=sw.rrtmg_sw(tables,nlay,int(e('icld')),e('play'),e('plev'),e('tlay'),e('tlev'),np.float32(e('tsfc')),e('h2ovmr'),e('o3vmr'),e('co2vmr'),e('ch4vmr'),e('n2ovmr'),e('o2vmr'),np.float32(e('asdir')),np.float32(e('asdif')),np.float32(e('aldir')),np.float32(e('aldif')),np.float32(e('coszen')),np.float32(e('adjes')),int(e('dyofyr')),np.float32(e('scon')),int(e('inflgsw')),int(e('iceflgsw')),int(e('liqflgsw')),*[e(name) for name in mc],e('reicmcl'),e('relqmcl'),e('resnmcl'),tau,ssa,asy,aer_opt=3)
    offset=0; case={'id':c,'nlay':nlay,'outputs':{}}
    for name in outputs:
        count=nlay if name in ('swhr','swhrc') else nlay+1
        value=raw[offset:offset+count]; offset+=count
        fixture[f'{c}/out/{name}']=value.copy()
        if name not in ref: continue
        want=np.asarray(ref[name],np.float32)
        case['outputs'][name]={'words':int(count),'mismatches':int(np.count_nonzero(value.view('u4') != want.view('u4'))),'max_abs':float(np.max(np.abs(value-want)))}
    assert offset==raw.size
    for name,val in zip(('tau','ssa','asy'),(tau,ssa,asy)): fixture[f'{c}/in/{name}']=val
    receipt['cases'].append(case)
    assert all(v['mismatches'] == 0 for v in case['outputs'].values()), case
    print(c, sum(v['words'] for v in case['outputs'].values()), 'words, max ULP 0', flush=True)
np.savez(output,**fixture,receipt=json.dumps(receipt))
(build/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
