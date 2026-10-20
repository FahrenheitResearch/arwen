#!/usr/bin/env python3
"""Finish the offline numeric lowering: thread-private state and CUDA ABI."""
from pathlib import Path
import argparse
import json
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('generated',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();meta=json.loads((a.generated/'schema.json').read_text())
    src=(a.generated/'lowered.c').read_text()
    src=re.sub(r'^/\* model.f.*?\*/','// Generated from Apache-2.0 Noah-OWP; see NOTICE and schema.json.',src,count=1,flags=re.S)
    src=src.replace('#include "f2c.h"','')
    src=src.replace('struct {','struct Model {',1).replace('} model_;','};',1).replace('#define model_1 model_','')
    src=src.replace('model_1.','m.')
    # Remove declarations of compiler builtins and local prototypes. All user
    # prototypes are generated below with the explicit thread-state argument.
    src=re.sub(r'\bextern\b[^;]*;','',src)
    src=re.sub(r'\bdouble\s+\w+\s*\([^;]*;','',src)
    funcs=[]
    pat=r'((?:/\* Subroutine \*/ )?(?:int|integer|real|doublereal))\s+(r\d+_)\((.*?)\)\s*\{'
    def signature(m):
        args=m[3].strip();args='' if args=='void' else args
        decl=f'__device__ __forceinline__ {m[1]} {m[2]}(Model &m'+(', '+args if args else '')+')'
        funcs.append(decl+';')
        return decl+' {'
    src=re.sub(pat,signature,src,flags=re.S)
    # Inject thread context into calls, excluding newly rewritten declarations.
    src=re.sub(r'\b(r\d+_)\((?!Model &m)',lambda m:m[1]+'(m, ',src)
    src=src.replace('(m, )','(m)')
    # f2c models REAL exponentiation with double temporaries. The native default
    # REAL operations are float32: narrow these power operands before evaluation.
    src=re.sub(r'\bdoublereal (d__\d[^;]*);',r'real \1;',src)
    src=re.sub(r'\bstatic doublereal (c_b\w+)',r'static real \1',src)
    # Initialized local Fortran diagnostic accumulators do not affect physics.
    # Localize them, and readonly DATA tables, so catchments never share state.
    boundary=src.index('__device__ __forceinline__')
    src=src[:boundary].replace('static ','static __device__ ')+src[boundary:].replace('static ','')
    types='''// SPDX-License-Identifier: Apache-2.0

typedef int integer;
typedef int logical;
typedef float real;
typedef double doublereal;
#define TRUE_ 1
#define FALSE_ 0
#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) abs(x)
#define dmin(a,b) ((a) <= (b) ? (a) : (b))
#define dmax(a,b) ((a) >= (b) ? (a) : (b))
#define min(a,b) dmin(a,b)
#define max(a,b) dmax(a,b)
__device__ __forceinline__ float pow_dd(float*a,float*b){return OWP_POW(*a,*b);}
__device__ __forceinline__ float pow_ri(float*a,int*b){
  int n=*b; float x=*a,r=1.f;
  if(n<0){x=1.f/x; n=-n;}
  while(n){if(n&1)r*=x;n>>=1;if(n)x*=x;}return r;
}
__device__ __forceinline__ float r_mod(float*a,float*b){return fmodf(*a,*b);}
__device__ __forceinline__ float r_sign(float*a,float*b){return *b>=0.f?fabsf(*a):-fabsf(*a);}
__device__ __forceinline__ float r_lg10(float*a){return OWP_LOG10(*a);}
#define exp(x) OWP_EXP((float)(x))
#define log(x) OWP_LOG((float)(x))
#define sqrt(x) sqrtf((float)(x))
#define sin(x) OWP_SIN((float)(x))
#define cos(x) OWP_COS((float)(x))
#define tan(x) tanf((float)(x))
#define asin(x) OWP_ASIN((float)(x))
#define atan(x) OWP_ATAN((float)(x))
#define tanh(x) OWP_TANH((float)(x))
'''
    insert=src.index('/* Table of constant values */')
    src=src[:insert]+'\n'+'\n'.join(funcs)+'\n'+src[insert:]
    # SoA lanes coalesce across catchments. Fields remain separately addressable
    # array views for checkpoint/forcing/output; no device pointer table is built.
    offsets={'f32':0,'i32':0,'f64':0};load=[];store=[]
    ctype={'f32':'float','i32':'int','f64':'double'}
    for f in meta['fields']:
        dtype='f64' if f['type'] in ('real*8','double precision') else 'i32' if f['type'] in ('integer','logical') else 'f32'
        count=1
        for lo,hi in f['bounds']:count*=hi-lo+1
        off=offsets[dtype];offsets[dtype]+=count
        f.update(dtype=dtype,offset=off,count=count)
        for j in range(count):
            lhs='m.'+f['symbol']+(f'[{j}]' if f['bounds'] else '')
            addr=f'{dtype}[{off+j}*n+c]'
            load.append(lhs+'='+addr+';');store.append(addr+'='+lhs+';')
    calls={u['name']:u['symbol']+'_' for u in meta['routines']}
    engine=f'''\nextern "C" __global__ void noahowp_step(
       int n, float *f32, int *i32, double *f64, int *status, int stage, int advance){{
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    Model m;
    {' '.join(load)}
    if(m.f0000!={meta['nsoil']}||m.f0001!={meta['nsnow']}||m.f0170!=2||m.f0078<1||m.f0078>{meta['nsoil']}){{status[c]=4;return;}}
    m.status=0;
    if(stage==5)m.f0013=m.f0011+(m.f0014-1)*(double)m.f0005;
    if(stage==0||stage==5){calls['utilitiesmain']}(m,&m.f0014);
    if(stage==1||stage==5){calls['forcingmain']}(m);
    if(stage==2||stage==5){calls['interceptionmain']}(m);
    if(stage==3||stage==5){calls['energymain']}(m);
    if((stage==4||stage==5)&&!(m.status&2)){calls['watermain']}(m);
    if(stage==6){calls['vegefluxmain']}(m);
    if(stage==7){calls['barefluxmain']}(m);
    if(advance&&!(m.status&2)){{m.f0014+=1;m.f0016+=m.f0005;}}
    {' '.join(store)}
    status[c]=m.status;
}}
'''
    meta['slab_fields']=offsets
    byname={f['name']:f for f in meta['fields']}
    dt=byname['domain.dt']['offset']
    etran=byname['water.etran']['offset']
    qseva=byname['water.qseva']['offset']
    ecan=byname['water.ecan']['offset']
    engine+=f'''\nextern "C" __global__ void noahowp_export_bmi(int n, const float*f32, float*out){{
    int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;
    out[c]=f32[{etran}*n+c]*f32[{dt}*n+c];
    out[n+c]=f32[{qseva}*n+c]*1000.f;
    out[2*n+c]=f32[{ecan}*n+c]*f32[{dt}*n+c];
}}\n'''
    forcing_fields = ('uu','vv','sfctmp','sfcprs','q2','lwdn','soldn','prcpnonc')
    forcing_copy = ' '.join(f'f32[{byname["forcing."+name]["offset"]}*n+c]=a{i}[c];' for i,name in enumerate(forcing_fields))
    engine+='\nextern "C" __global__ void noahowp_copy_forcing(int n,float*f32,'+','.join(f'const float*a{i}' for i in range(8))+'){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=n)return;'+forcing_copy+'}\n'
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'physics.cu').write_text('\n'.join(line.rstrip() for line in (types+src+engine).splitlines())+'\n')
    (a.output/'schema.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':main()
