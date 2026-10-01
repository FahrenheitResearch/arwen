from pathlib import Path
import re,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent/'python-deps'))
from clang import cindex as ci
root=Path(__file__).parent
up=root/'upstream'
config=(up/'math_config.h').read_text()
config=config[config.index('#define EXP_TABLE_BITS'):config.index('extern const struct erff_data')]
config=re.sub(r'ALIGN\(16\) HIDDEN','',config)
shim='''
#define HAVE_FAST_FMA 1
#define TOINT_INTRINSICS 0
#define WANT_ROUNDING 1
#define WANT_ERRNO 0
#define USE_GLIBC_ABI 0
#define unlikely(x) (x)
#define eval_as_double(x) (x)
#define opt_barrier_double(x) (x)
#define force_eval_double(x) ((void)(x))
#define check_oflow(x) (x)
#define check_uflow(x) (x)
#define __FP_FAST_FMA 1
static inline uint64_t asuint64(double x) {union {double f; uint64_t u;} v={x};return v.u;}
static inline double asdouble(uint64_t x) {union {uint64_t u; double f;} v={x};return v.f;}
static inline int issignaling_inline(double x){uint64_t u=asuint64(x);return ((u&0x7ff8000000000000ULL)==0x7ff0000000000000ULL)&&(u&0xfffffffffffffULL);}
static inline double __math_oflow(uint32_t s){return s ? -INFINITY:INFINITY;}
static inline double __math_uflow(uint32_t s){return asdouble((uint64_t)(!!s)<<63);}
static inline double __math_divzero(uint32_t s){return s ? -INFINITY:INFINITY;}
static inline double __math_invalid(double x){return (x-x)/(x-x);}
'''
def strip(s):return re.sub(r'^#include[^\n]*','',s,flags=re.M)
def fuse(s,name):
 if name in ('exp','pow'):
  s=s.replace('kd = eval_as_double (z + Shift);','kd = fma (InvLn2N, x, Shift);')
  s=s.replace('r = x + kd * NegLn2hiN + kd * NegLn2loN;','r = fma (kd, NegLn2loN, fma (kd, NegLn2hiN, x));')
  s=s.replace('tmp = tail + r + r2 * (C2 + r * C3) + r2 * r2 * (C4 + r * C5);','tmp = fma (r2*r2, fma(r,C5,C4), fma(r2,fma(r,C3,C2),tail+r));')
  s=s.replace('return eval_as_double (scale + scale * tmp);','return fma (scale,tmp,scale);')
  s=s.replace('y = 0x1p1009 * (scale + scale * tmp);','y = 0x1p1009 * fma (scale,tmp,scale);')
  # overflow branch only, underflow uses shared rounded product.
  if name=='pow':s=s.replace('y = scale + scale * tmp;', 'y = fma(scale,tmp,scale);',1)
 if name=='log':
  s=s.replace('w = kd * Ln2hi + logc;','w = fma(kd,Ln2hi,logc);')
  s=s.replace('lo = w - hi + r + kd * Ln2lo;','lo = fma(kd,Ln2lo,w-hi+r);')
  s=s.replace('y = lo + r2 * A[0] + r * r2 * (A[1] + r * A[2] + r2 * (A[3] + r * A[4])) + hi;', 'y = fma(r*r2,fma(r2,fma(r,A[4],A[3]),fma(r,A[2],A[1])),fma(r2,A[0],lo))+hi;')
  # near-one split and polynomial: installed compiler fuses across assignments.
  s=s.replace('double rhi = r + w - w;', 'double rhi = fma(-r,0x1p27,fma(r,0x1p27,r));')
  s=s.replace('hi = r + w;\n      lo = r - hi + w;','hi = fma(rhi*rhi,B[0],r);\n      lo = fma(rhi*rhi,B[0],r-hi);')
  s=s.replace('lo += B[0] * rlo * (rhi + r);','lo = fma(B[0]*rlo,rhi+r,lo);')
  # postpone outer r3 multiply so it fuses with low correction.
  start='y = r3 * (B[1] + r * B[2] + r2 * B[3]\n\t\t+ r3 * (B[4] + r * B[5] + r2 * B[6]\n\t\t\t+ r3 * (B[7] + r * B[8] + r2 * B[9] + r3 * B[10])));'
  new='y = fma(r3,fma(r3,B[10],fma(r2,B[9],fma(r,B[8],B[7]))),fma(r2,B[6],fma(r,B[5],B[4]))); y = fma(r3,y,fma(r2,B[3],fma(r,B[2],B[1])));'
  assert start in s
  s=s.replace(start,new).replace('y += lo;','y = fma(r3,y,lo);')
 if name=='pow':
  s=s.replace('t1 = kd * Ln2hi + logc;','t1 = fma(kd,Ln2hi,logc);').replace('lo1 = kd * Ln2lo + logctail;','lo1 = fma(kd,Ln2lo,logctail);')
  s=s.replace('p = (ar3\n       * (A[1] + r * A[2] + ar2 * (A[3] + r * A[4] + ar2 * (A[5] + r * A[6]))));','p = fma(ar2,fma(ar2,fma(r,A[6],A[5]),fma(r,A[4],A[3])),fma(r,A[2],A[1]));')
  s=s.replace('lo = lo1 + lo2 + lo3 + lo4 + p;','lo = fma(ar3,p,lo1+lo2+lo3+lo4);')
  s=s.replace('elo = y * lo + fma (y, hi, -ehi);','elo = fma(y,lo,fma(y,hi,-ehi));')
 return s
pieces=[]
for name in ['exp','log','pow','cos','acos']:
 s=strip((up/(name+'.c')).read_text())
 if name in ['exp','log','pow']:
  s=fuse(s,name)
  data=['exp_data'] if name=='exp' else ['log_data'] if name=='log' else ['exp_data','pow_log_data']
  helpers=strip((up/'pow_common.h').read_text()) if name=='pow' else ''
  s=shim+config+'\n'+''.join(strip((up/(d+'.c')).read_text()) for d in data)+'\n'+helpers+s
 else:
  s=s.replace('fegetround()', 'FE_TONEAREST')
 s='#include <stdint.h>\n#include <math.h>\n#include <float.h>\n#include <fenv.h>\nint G64_MARKER;\n'+s
 tmp=root/(name+'.prep.c');tmp.write_text(s)
 p=subprocess.run(['gcc','-E','-P',str(tmp)],capture_output=True,text=True,check=True).stdout
 p=p[p.index('int G64_MARKER;')+len('int G64_MARKER;'):]
 # remove tests, pragmas, extern data declarations (definitions retained).
 p=re.sub(r'^TEST_[^\n]*(?:\n\s*[^;]*\))?\s*','',p,flags=re.M) if name in ['exp','log','pow'] else p
 p=re.sub(r'^#pragma[^\n]*','',p,flags=re.M)
 # Test macros are multiline. Truncate at first TEST_.
 if 'TEST_' in p:p=p[:p.index('TEST_')]
 tmp=root/(name+'.pin.c');prefix='#include <stdint.h>\n#include <math.h>\n#include <float.h>\n'
 tmp.write_text(prefix+p)
 tu=ci.Index.create().parse(str(tmp),args=['-std=gnu11','-Wno-unknown-pragmas','-I/usr/lib/gcc/x86_64-linux-gnu/15/include'])
 errors=[str(d) for d in tu.diagnostics if d.severity>=ci.Diagnostic.Error]
 if errors:raise RuntimeError((name,errors))
 raw=tmp.read_text(); edits=[]
 # recursive replacement on outermost double arithmetic; children handled recursively.
 def render(c):
  a,b=c.extent.start.offset,c.extent.end.offset
  children=list(c.get_children())
  if c.kind in [ci.CursorKind.BINARY_OPERATOR,ci.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR] and c.type.spelling=='double' and len(children)==2:
   l,r=children;op=raw[l.extent.end.offset:r.extent.start.offset].strip()
   if op in ['+','-','*','/','+=','-=','*=','/=']:
    fn={'+':'__dadd_rn','-':'__dsub_rn','*':'__dmul_rn','/':'__ddiv_rn'}[op[0]]
    lhs=render(l);rhs=render(r)
    return (lhs+' = ' if len(op)==2 else '')+fn+'('+lhs+','+rhs+')'
  out=raw[a:b]
  for ch in reversed(children):
   ca,cb=ch.extent.start.offset,ch.extent.end.offset
   if a<=ca<cb<=b:out=out[:ca-a]+render(ch)+out[cb-a:]
  return out
 for c in tu.cursor.get_children():
  if c.location.file and str(c.location.file)==str(tmp):
   a,b=c.extent.start.offset,c.extent.end.offset
   if a>=len(prefix):edits.append((a,b,render(c) if c.kind==ci.CursorKind.FUNCTION_DECL else raw[a:b]))
 for a,b,v in reversed(edits):raw=raw[:a]+v+raw[b:]
 p=raw[len(prefix):]
 # Qualify all functions and global data based on AST declaration offsets via simple forms.
 p=re.sub(r'\bstatic\s+(?:inline\s+)?(?=(?:double|void|int|signed char)\b)','__device__ __forceinline__ ',p)
 p=re.sub(r'(?m)^double\s*\n?(?=\w+\s*\()','__device__ __forceinline__ double ',p)
 p=p.replace('__builtin_fma','__fma_rn')
 p=re.sub(r'\bfma\s*\(','__fma_rn(',p)
 p=p.replace('__builtin_clzll','g64_clzll').replace('__builtin_floor','floor').replace('__builtin_fabs','fabs').replace('__builtin_sqrt','__dsqrt_rn').replace('__builtin_copysign','copysign').replace('__builtin_roundeven','rint')
 p=re.sub(r'__builtin_expect\s*\(([^,\n]+),\s*[01]\)',r'(\1)',p)
 p=p.replace('__attribute__((noinline,cold))','')
 # global const structs and arrays; CUDA globals are inside per-function namespaces.
 p=re.sub(r'(?m)^static const\s+', '__device__ const ',p)
 p=re.sub(r'(?m)^const struct ', '__device__ const struct ',p)
 # remaining static return types (u128/uint64 etc) function qualifiers.
 p=re.sub(r'\bstatic inline\s+', '__device__ __forceinline__ ',p)
 p=re.sub(r'\bstatic\s*\n(?=double)', '__device__ ',p)
 pieces.append('namespace g64_'+name+' {\n'+p+'\n}\n')
(root/'generated.inc').write_text('\n'.join(pieces))
print('generated',sum(map(len,pieces)))
