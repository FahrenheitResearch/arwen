from pathlib import Path
import re
r=Path('tools/uwpbl_wrf471_oracle/libm64')
s=(r/'generated.inc').read_text()
s=re.sub(r'extern const struct (\w+)\s*\{(.*?)\}\s*__\w+\s*;',r'struct \1 {\2};',s,flags=re.S)
orders={'exp_data':'invln2N negln2hiN negln2loN poly shift exp2_shift exp2_poly neglog10_2hiN neglog10_2loN exp10_poly tab invlog10_2N'.split(),'log_data':'ln2hi ln2lo poly poly1 tab'.split(),'pow_log_data':'ln2hi ln2lo poly tab'.split()}
for st,order in orders.items():
 pat=r'(__device__ const struct '+st+r' \w+ = \{)(.*?)(\n\};)'
 def cv(m):
  body=m[2];marks=list(re.finditer(r'(?m)^\.(\w+)\s*=',body));d={}
  for i,t in enumerate(marks):d[t[1]]=body[t.end():marks[i+1].start() if i+1<len(marks) else len(body)].strip().rstrip(',')
  return m[1]+'\n'+',\n'.join(d[k] for k in order)+m[3]
 s=re.sub(pat,cv,s,flags=re.S)
s=s.replace('unsigned _BitInt(128)','unsigned __int128')
s=re.sub(r'\.hi\s*=\s*([^,{}]+),\s*\.lo\s*=\s*([^,{}]+)',r'.lo = \2, .hi = \1',s)
s=s.replace('__builtin_inff ()','g64_inf()').replace('!__builtin_constant_p (xtail) || xtail != 0.0','xtail != 0.0')
s=re.sub(r'__attribute__\(\(cold\)\)','',s)
exec((r/'postprocess.py').read_text())
head='''// ======================================================================
// THIRD-PARTY NOTICE. Original CUDA adaptation is distributed under
// Apache-2.0. The MIT grants below cover the transcribed upstream parts.
// Arm optimized-routines exp/log/pow and data are taken
// under MIT from upstream MIT sources, not from glibc source text.
// Copyright (c) 2017-2026, Arm Limited.
// SPDX-License-Identifier: MIT
// CORE-MATH cos/acos retain their MIT notices below.
// uw_cos and uw_acos target correctly rounded binary64 round-to-nearest,
// ties-to-even results. They are NOT glibc implementations.
// PROVENANCE: see tools/uwpbl_wrf471_oracle/libm64/upstream and generate.py.
// Fused sites derive from the oracle host's installed libm disassembly.
// Verification status is recorded in LANE-REPORT.md.
// ======================================================================
#pragma once
#if !defined(__CUDACC_RTC__)
#include <stdint.h>
#else
typedef unsigned long long uint64_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef int int32_t;
#endif
__device__ __forceinline__ double g64_inf(){return __longlong_as_double(0x7ff0000000000000ULL);}
__device__ __forceinline__ int g64_clzll(unsigned long long x){return __clzll(x);}
'''
lic=(r/'upstream'/'LICENSE-Arm.txt').read_text()
lic=lic[lic.index('Permission is hereby'):lic.index('Apache-2.0 WITH LLVM-exception\n------------------------------')].strip()
mit='\n'.join('// '+line for line in lic.splitlines())+'\n'
for name in ['cos','acos']:
 text=(r/'upstream'/(name+'.c')).read_text();notice=text[:text.index('*/')+2]
 s=s.replace('namespace g64_'+name+' {',notice+'\nnamespace g64_'+name+' {')
end='''
__device__ __forceinline__ double glibc_exp(double x){uint64_t u=g64_exp::asuint64(x);if((u&0x7fffffffffffffffULL)>0x7ff0000000000000ULL)return g64_exp::asdouble(u|0x8000000000000ULL);return g64_exp::exp(x);}
__device__ __forceinline__ double glibc_log(double x){return g64_log::log(x);}
__device__ __forceinline__ double glibc_pow(double x,double y){
 uint64_t ix=g64_pow::asuint64(x),iy=g64_pow::asuint64(y);
 bool nx=(ix&0x7fffffffffffffffULL)>0x7ff0000000000000ULL;
 bool ny=(iy&0x7fffffffffffffffULL)>0x7ff0000000000000ULL;
 if(nx){
  if((iy<<1)==0 && (ix&0x8000000000000ULL))return 1.;
  uint64_t q=ix|0x8000000000000ULL;
  if((iy&0x7fffffffffffffffULL)!=0 && (iy&0x7fffffffffffffffULL)<0x7ff0000000000000ULL && (ix>>63) && g64_pow::checkint(iy)==1)q^=0x8000000000000000ULL;
  return g64_pow::asdouble(q);
 }
 if(ny){if(ix==0x3ff0000000000000ULL && (iy&0x8000000000000ULL))return 1.;return g64_pow::asdouble(iy|0x8000000000000ULL);}
 return g64_pow::pow(x,y);
}
__device__ __forceinline__ double uw_cos(double x){return g64_cos::cr_cos(x);}
__device__ __forceinline__ double uw_acos(double x){return g64_acos::cr_acos(x);}
'''
Path('gpuwm/core/kernels/glibc_flt64.cuh').write_text(head+mit+s+end)
