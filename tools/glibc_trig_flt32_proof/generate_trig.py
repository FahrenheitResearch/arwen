"""Generate pinned CORE-MATH transcriptions from the retained glibc sources.

Only floating arithmetic is rewritten. The sin/cos FMA path and the 128-bit
tan reduction are separately transcribed in trig_prefix.txt.
"""
from pathlib import Path
import re
from pycparser import c_parser, c_ast, c_generator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'evidence/upstream'

def clean(s):
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'^#.*$', '', s, flags=re.M)
    s = re.sub(r'__attribute__\s*\(\(.*?\)\)', '', s)
    s = re.sub(r'^(?:strong_alias|libm_alias\w*|versioned_symbol)\s*\([^\n]*', '', s, flags=re.M)
    s = s.replace('__glibc_unlikely', '').replace('__glibc_likely', '')
    return s

class Pins(c_generator.CGenerator):
    def __init__(self):
        super().__init__()
        self.types = {}
        self.ret = 'int'
        self.tables = []
        self.scope = ''
        self.aliases = {}
        self.unit = ''
    def ty(self, n):
        if isinstance(n, (c_ast.TypeDecl, c_ast.PtrDecl, c_ast.ArrayDecl)):
            return self.ty(n.type)
        if isinstance(n, c_ast.IdentifierType):
            return n.names[0]
        if isinstance(n, c_ast.ID): return self.types.get(n.name, 'int')
        if isinstance(n, c_ast.Constant): return n.type
        if isinstance(n, c_ast.Cast): return self.ty(n.to_type.type)
        if isinstance(n, c_ast.ArrayRef): return self.ty(n.name)
        if isinstance(n, c_ast.StructRef): return 'float'
        if isinstance(n, c_ast.UnaryOp): return self.ty(n.expr)
        if isinstance(n, c_ast.FuncCall):
            return {'poly12':'double','sqrt':'double','fabs':'double','copysign':'double',
                    'math_opt_barrier':'double','roundeven_finite':'double',
                    'rltl':'double','rbig':'double','fmaf':'float','fabsf':'float',
                    'copysignf':'float','as_special':'float','__math_invalidf':'float',
                    'asuint64':'uint64_t','asuint':'uint32_t'}.get(n.name.name, 'int')
        if isinstance(n, c_ast.BinaryOp):
            if n.op not in ('+', '-', '*', '/'): return 'int'
            ts = [self.ty(n.left), self.ty(n.right)]
            return 'double' if 'double' in ts else 'float' if 'float' in ts else 'int'
        return 'int'
    def cv(self, s, t, n):
        return 'gt_d2f(' + s + ')' if t == 'float' and self.ty(n) == 'double' else s
    def visit_ID(self,n): return self.aliases.get(n.name,n.name)
    def visit_BinaryOp(self,n):
        t = self.ty(n)
        if t in ('float','double') and n.op in ('+','-','*','/'):
            op = {'+':'add','-':'sub','*':'mul','/':'div'}[n.op]
            return '__' + ('d' if t=='double' else 'f') + op + '_rn(' + self.visit(n.left) + ', ' + self.visit(n.right) + ')'
        return super().visit_BinaryOp(n)
    def visit_Assignment(self,n):
        t=self.ty(n.lvalue)
        if t in ('float','double'):
            if n.op != '=':
                op=c_ast.BinaryOp(n.op[0], n.lvalue, n.rvalue)
                return self.visit(n.lvalue)+' = '+self.cv(self.visit(op),t,op)
            return self.visit(n.lvalue)+' = '+self.cv(self.visit(n.rvalue),t,n.rvalue)
        return super().visit_Assignment(n)
    def visit_Decl(self,n,no_type=False):
        self.types[n.name]=self.ty(n.type)
        if 'static' in n.storage and isinstance(n.type,c_ast.ArrayDecl):
            orig=n.name
            new='gt_'+self.scope+'_'+orig
            n.name=new; n.storage=[]
            typ=n.type
            while not isinstance(typ,c_ast.TypeDecl): typ=typ.type
            typ.declname=new
            self.tables.append('__device__ '+super().visit_Decl(n)+';\n')
            self.aliases[orig]=new
            return '/* table '+new+' */'
        if n.init is not None and self.ty(n.type)=='float' and self.ty(n.init)=='double':
            old=n.init; n.init=None
            out=super().visit_Decl(n,no_type)+' = '+self.cv(self.visit(old),'float',old)
            n.init=old
            return out
        return super().visit_Decl(n,no_type)
    def visit_FuncDef(self,n):
        self.scope=n.decl.name.lstrip('_')
        self.ret=self.ty(n.decl.type.type)
        old=n.decl.name
        names={'__asinf':'glibc_asinf','__acosf':'glibc_acosf','__atanf':'glibc_atanf','__tanf':'glibc_tanf'}
        new=names.get(old,'gt_'+self.unit+'_'+self.scope)
        n.decl.name=new; n.decl.type.type.declname=new
        self.aliases[old]=new
        return '__device__ '+super().visit_FuncDef(n)
    def visit_Return(self,n):
        return 'return '+self.cv(self.visit(n.expr),self.ret,n.expr)+';' if n.expr else 'return;'
    def visit_FuncCall(self,n):
        names={'fmaf':'__fmaf_rn','sqrt':'__dsqrt_rn','fabs':'gt_abs','fabsf':'gt_absf',
               'copysign':'gt_sign','copysignf':'gt_signf','asuint':'__float_as_uint',
               'asuint64':'gt_as_u64','__math_invalidf':'gt_invalid','math_opt_barrier':'gt_identity',
               'roundeven_finite':'gt_roundeven','rbig':'gt_rbig'}
        name=n.name.name
        return names.get(name,self.aliases.get(name,name))+'('+self.visit(n.args)+')'

def generate():
    chunks=[]
    for file in ('e_asinf.c','e_acosf.c','s_atanf.c','s_tanf.c'):
        s=clean((SRC/file).read_text())
        if file=='s_atanf.c':
            s=s.replace('PI_OVER2_H','0x1.9p0').replace('PI_OVER2_L','0x1.0fdaa22168c23p-7')
        if file=='s_tanf.c':
            start=s.index('static double')
            end=s.index('float\n__tanf',start)
            s=s[:start]+s[end:]
            s=s.replace('array_length (st)','8').replace('UINT64_C(0)','0ULL')
        s=s.replace('C0','gt_asincos_c0').replace('C1','gt_asincos_c1')
        ast=c_parser.CParser().parse('typedef unsigned int uint32_t; typedef unsigned long long uint64_t; typedef long long int64_t; typedef int bool;\n'+s)
        gen=Pins()
        gen.unit=file[2:-2]
        gen.types.update({'gt_asincos_c0':'double','gt_asincos_c1':'double'})
        bodies=[]
        for n in ast.ext[4:]: bodies.append(gen.visit(n))
        chunks.append('// glibc-2.43 sysdeps/ieee754/flt-32/'+file+'\n'+''.join(gen.tables)+'\n'+'\n'.join(bodies))
    out=(ROOT/'tools/trig_prefix.txt').read_text()+'\n'+'\n'.join(chunks)+'\n#endif\n'
    # Preserve exponent-zero inputs in CUDA FTZ builds; retain the explicit
    # fused operations for all normal inputs in the same source paths.
    out=out.replace('return __fmaf_rn(x, 0x1p-25, x);','return gt_tiny_fmaf(x, 0x1p-25f, x);')
    out=out.replace('return __fmaf_rn(-x, gt_absf(x), x);','return gt_tiny_fmaf(-x, gt_absf(x), x);')
    out=out.replace('return __fmaf_rn(x, gt_absf(x), x);','return gt_tiny_fmaf(x, gt_absf(x), x);')
    out=out.replace('return __fadd_rn(x, x);', 'return gt_invalid(x); // Preserve the input NaN sign and payload.')
    out=out.replace('-0x1.5555555555555p-2f','-0x1.555556p-2f')
    for old,new in [('uint32_t','gt_u32'),('uint64_t','gt_u64'),('int64_t','gt_i64'),('int32_t','int')]:
        out=re.sub(r'\b'+old+r'\b',new,out)
    (ROOT/'gpuwm/core/kernels/glibc_trig_flt32.cuh').write_text(out)

if __name__=='__main__': generate()
