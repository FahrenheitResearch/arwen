"""Check the unchanged CUDA bodies on CPU; no CUDA API or GPU is used."""
import ctypes as C
import json
from pathlib import Path
import struct
import subprocess
import sys

root=Path(sys.argv[1]).resolve();build=Path(sys.argv[2]).resolve();build.mkdir(exist_ok=True)
shim=root/'tools/chem_wrf471_oracle/cuda-host-vertmx.cuh'
kernels=root/'gpuwm/core/kernels'
P=C.POINTER(C.c_float);F=C.c_float;I=C.c_int

def compile_kernel(name,nz=59):
    source='#include "'+str(shim)+'"\n'
    if name=='chem_prep':source+='#include "'+str(kernels/'glibc_flt32.cuh')+'"\n'
    source+='#include "'+str(kernels/(name+'.cu'))+'"\n'
    cpp=build/(name+'.cpp');cpp.write_text(source)
    lib=build/(name+'.so')
    command=['g++','-std=c++17','-O0','-ffp-contract=off','-fPIC','-shared',f'-DCHEM_NZ={nz}',str(cpp),'-o',str(lib)]
    subprocess.run(command,check=True)
    result=C.CDLL(str(lib));result.host_column.argtypes=[I]
    return result

def unpack(raw,kind='f'):return struct.unpack('<'+kind*(len(raw)//4),raw)
def array(values):return (F*len(values))(*values)
def raw(a):return bytes(a)
def load(directory):return {p.stem:p.read_bytes() for p in directory.glob('*.bin')}
def f32(x):return F(x).value
def expanded(v):return [x for x in v for _ in range(6)]
def expanded_bytes(b):return b''.join(b[i:i+4]*6 for i in range(0,len(b),4))

tables={}
prep=compile_kernel('chem_prep');prep.chem_prep.argtypes=[P]*11+[I]*6+[P]*13
for directory in sorted((root/'tests/data/oracles/chem/core/chem_prep').iterdir()):
    if not (directory/'MANIFEST.txt').exists():continue
    data=load(directory);a={k:unpack(v) for k,v in data.items()};nz=len(a['alt'])-1
    p=array(expanded([f32(x+y) for x,y in zip(a['p'][:-1],a['pb'][:-1])]));thp=array(expanded(a['t'][:-1]));thb=array([300]*nz)
    inputs=[p,thp,thb,array(expanded(a['alt'][:-1])),array(expanded(a['ph'])),array(expanded(a['phb'])),array([a['u'][k*2+i%2] for k in range(nz) for j in range(2) for i in range(4)]),array([a['v'][k+(j%2)*(nz+1)] for k in range(nz) for j in range(3) for i in range(3)]),array(expanded(a['qv'][:-1])),array(a['fnm'][:-1]),array(a['fnp'][:-1])]
    names=('p_phy','t_phy','rho','dryrho','alt','u_phy','v_phy','z_at_w','dz8w','z','rh','p8w','t8w')
    out={key:array([0]*(6*(nz+(key in ('z_at_w','p8w','t8w'))))) for key in names}
    for c in range(6):
        prep.host_column(c);prep.chem_prep(*inputs,0,1,1,nz,2,3,*out.values())
    tables[directory.name]={}
    for key in names:
        if key not in data:continue
        expected=data[key] if key in ('z_at_w','p8w','t8w') else data[key][:-4]
        expected=expanded_bytes(expected)
        actual=raw(out[key]);assert actual==expected,(directory.name,key)
        tables[directory.name][key]={'max_ulp':0,'n_nonzero':0,'n':len(actual)//4}
vert=compile_kernel('chem_vertmx');vert.chem_vertmx.argtypes=[P]*7+[F,F,I,P]+[P]*5+[I]*7+[P]*3+[I]
for directory in sorted((root/'tests/data/oracles/chem/core/vertmx').iterdir()):
    if not (directory/'MANIFEST.txt').exists():continue
    data=load(directory);a={k:unpack(v,'i' if k in ('urban','indices') else 'f') for k,v in data.items()};nz=len(a['alt'])
    tables[directory.name]={}
    for r in range(2):
        chem=array(expanded(a['input'][r*nz:(r+1)*nz]));accum=array([a['accum_in'][r]]*6)
        inputs=[chem]+[array(expanded(a[key])) for key in ('alt','z_at_w','z','dz8w','exch_h')]+[array([a['vd'][r]]*6)]
        idx=a['indices'];anth=a['anth']
        opts=[array([anth[idx[0]-1]]*6),array(list(a['fire'])*6),array([anth[2]]*6),array([anth[3]]*6),array([anth[idx[2]-1]]*6)]
        mixed=array([0]*(6*nz));ek=array([0]*(6*(nz+1)));dd=array([0]*6)
        for c in range(6):
            vert.host_column(c)
            vert.chem_vertmx(*inputs,a['dt'][0],1e-16,int(r==0),accum,*opts,int(idx[0]>=2),int(idx[3]>=2),int(idx[1]>2),int(idx[2]>2),a['urban'][0],nz,6,mixed,ek,dd,1)
        outputs={'output':(chem,data['output'][r*nz*4:(r+1)*nz*4]),'ekmfull':(ek,data['ekmfull']),'ddmassn':(dd,data['ddmassn'][r*4:(r+1)*4]),'accum':(accum,data['accum'][r*4:(r+1)*4])}
        if r==1:outputs['mixed']=(mixed,data['mixed'])
        for key,(actual,expected) in outputs.items():
            expected=expanded_bytes(expected)
            assert raw(actual)==expected,(directory.name,r,key)
            tables[directory.name][('gas','aerosol')[r]+'/'+key]={'max_ulp':0,'n_nonzero':0,'n':len(expected)//4}
(build/'cuda-host-ulp-tables.json').write_text(json.dumps(tables,indent=2))
print(f'{len(tables)} fixture cases, six columns each: all unchanged CUDA bodies match on CPU, max_ulp=0')
