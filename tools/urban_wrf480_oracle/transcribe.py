"""Emit explicitly rounded C++ from the pinned scheme's numerical routines.

This is a source-editing tool, executed on the lane CPU box. It emits only
the selected physics statements. Table I/O and unsupported morphology remain
host responsibilities. The emitted listing carries source line numbers.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, re
from pathlib import Path
import numpy as np
from gpuwm.verify.urban_ucm_ref import _syntax, _split, _Expression
from gpuwm.core.urban_ucm import UCM_TABLE_COLUMNS

NAMES='urban mos multi_layer force_restore sfcdif_urb direvap transp smflx srt sstep wdfcnd rosr12 shflx hrt hstep tbnd tdfcnd shadow_tree stomatal leleaf root_uptake force_restore_tree vf_tree_analytical cdf2d_vf'.split()
PIN='1be801c3e11aad59be8e0e23612bbe71ba4fa29cefc930520fb4c6e2e1169f6a'
TABLE=list(UCM_TABLE_COLUMNS)+['albvg','albt','fvg','rtree','rtree_m','rw_m','htree','dtree','lai_tree','lai_veg','z0vg','z0hvg','capvg','epsvg','epst']


def statements(text):
    held='';start=0
    for number,line in enumerate(text.splitlines(),1):
        quote=None;kept=[]
        for char in line:
            if char in "'\"":quote=None if quote==char else char if quote is None else quote
            if char=='!' and quote is None:break
            kept.append(char)
        line=''.join(kept).strip().lower()
        line=re.sub(r'\blambda\b','f_lambda',line)
        if not line or line.startswith('#'):continue
        if not held:start=number
        held += ' '+line.strip('&').strip()
        if line.endswith('&'):continue
        quote=None;parts=[];begin=0
        for i,char in enumerate(held):
            if char in "'\"":quote=None if quote==char else char if quote is None else quote
            if char==';' and quote is None:parts.append(held[begin:i]);begin=i+1
        parts.append(held[begin:])
        for item in parts:
            if item.strip():yield start,item.strip()
        held=''


class Routine:
    def __init__(self, lines):
        self.lines=lines
        header=lines[0][1]
        m=re.search(r'(subroutine|function)\s+(\w+)\s*\((.*?)\)(?:\s*result\((\w+)\))?',header)
        self.name=m[2];self.args=_split(m[3]);self.result=m[4] if m[1]=='function' else None
        self.types={};self.arrays={};self.intent={};self.constants={}
        self.expr=_Expression(self.constants);self.functions=set()
        self.constants.update(piconst=np.float32(np.pi),maxdirs=3,maxhgts=50)
        self.types.update(piconst='float',maxdirs='int',maxhgts='int')
        self.body=[]
        declarations=True
        for line_no,line in lines[1:-1]:
            if line=='implicit none':continue
            d=re.match(r'^(real|integer|logical|character)\b(.*)',line)
            if d:
                kind,rest=d.groups()
                attrs,items=rest.split('::',1) if '::' in rest else ('',rest.strip())
                dim=re.search(r'dimension\s*\(\s*(.*?)\s*\)',attrs)
                intent=re.search(r'intent\s*\(\s*(.*?)\s*\)',attrs)
                for item in _split(items):
                    nv=item.split('=',1);name=nv[0].strip()
                    ar=re.fullmatch(r'(\w+)\((.*)\)',name)
                    if ar:name=ar[1]
                    self.types[name]='double' if kind=='real' and re.search(r'(kind\s*=\s*8|\(\s*8\s*\))',attrs) else {'real':'float','integer':'int','logical':'bool','character':'char'}[kind]
                    self.intent[name]=intent[1] if intent else 'unspecified'
                    if 'parameter' in attrs:
                        self.constants[name]=(int if kind=='integer' else np.float32)(self.expr.constant(_syntax(nv[1])));continue
                    shape=ar[2] if ar else dim[1] if dim else None
                    if shape:self.arrays[name]=shape
                continue
            if line.startswith('parameter'):
                for item in _split(line[line.index('(')+1:-1]):
                    name,value=item.split('=',1);name=name.strip()
                    self.constants[name]=(int if self.types.get(name)=='int' else np.float32)(self.expr.constant(_syntax(value)))
                continue
            self.body.append((line_no,line))
        for _,line in self.body:
            if self.name=='sfcdif_urb':
                m=re.match(r'(ps[lp]\w+)\s*\((\w+)\)\s*=',line)
                if m:self.functions.add(m[1])
            m=re.match(r'do\s+(\w+)\s*=',line)
            if m:self.types[m[1]]='int'
        self.modified={m[1] for _,line in self.body if (m:=re.match(r'(\w+)\s*(?:\([^)]*\))?\s*=',line))}
        for name in self.modified:
            if name not in self.types and name not in self.args:
                self.types[name]='int' if name[0] in 'ijklmn' else 'float'

    def type(self,n):
        if isinstance(n,ast.Constant):return 'bool' if isinstance(n.value,bool) else 'int' if isinstance(n.value,int) else 'float'
        if isinstance(n,ast.Name):return self.types.get(n.id,'float')
        if isinstance(n,ast.UnaryOp):return self.type(n.operand)
        if isinstance(n,ast.BinOp):
            kinds=(self.type(n.left),self.type(n.right));return 'double' if 'double' in kinds else 'float' if 'float' in kinds else 'int'
        if isinstance(n,ast.Call):
            if n.func.id in self.arrays:return self.types.get(n.func.id,'float')
            return 'int' if n.func.id in ('int','nint','mod') else 'double' if any(self.type(a)=='double' for a in n.args) else 'float'
        return 'bool'

    def literal(self,x):
        if isinstance(x,(bool,np.bool_)):return 'true' if x else 'false'
        if isinstance(x,(int,np.integer)):return str(int(x))
        if isinstance(x,np.float64):return f'__longlong_as_double(0x{int(x.view(np.uint64)):016x}ull)'
        return f'__uint_as_float(0x{int(np.float32(x).view(np.uint32)):08x}u)'

    def node(self,n,element=None):
        if isinstance(n,ast.Name) and n.id in self.constants:return self.literal(self.constants[n.id])
        try:
            if self.type(n)=='double':raise KeyError('mixed precision expression')
            return self.literal(self.expr.constant(n))
        except (KeyError,ValueError,TypeError):pass
        if isinstance(n,ast.Name):
            return n.id+'['+element+']' if element is not None and n.id in self.arrays else n.id
        if isinstance(n,ast.BinOp):
            a,b=self.node(n.left,element),self.node(n.right,element)
            if isinstance(n.op,ast.Pow):
                fn='u480_powi64' if self.type(n.left)=='double' and self.type(n.right)=='int' else 'u_powi' if self.type(n.right)=='int' else 'gfk_pow'
                return f'{fn}({a},{b})'
            op={ast.Add:('FADD','+'),ast.Sub:('u480_sub','-'),ast.Mult:('FMUL','*'),ast.Div:('FDIV','/')}[type(n.op)]
            if self.type(n)=='double':return f"{ {'FADD':'__dadd_rn','u480_sub':'__dsub_rn','FMUL':'__dmul_rn','FDIV':'__ddiv_rn'}[op[0]] }({a},{b})"
            return f'{op[0]}({a},{b})' if self.type(n)=='float' else f'({a}{op[1]}{b})'
        if isinstance(n,ast.UnaryOp):
            return f'(!{self.node(n.operand,element)})' if isinstance(n.op,ast.Not) else f'(-{self.node(n.operand,element)})' if isinstance(n.op,ast.USub) else self.node(n.operand,element)
        if isinstance(n,ast.BoolOp):return '('+(' && ' if isinstance(n.op,ast.And) else ' || ').join(self.node(a,element) for a in n.values)+')'
        if isinstance(n,ast.Compare):
            terms=[];left=n.left
            for op,right in zip(n.ops,n.comparators):
                symbol={ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>=',ast.Eq:'==',ast.NotEq:'!='}[type(op)]
                terms.append(f'({self.node(left,element)}{symbol}{self.node(right,element)})');left=right
            return '('+' && '.join(terms)+')'
        if isinstance(n,ast.Call):
            name=n.func.id;args=[self.node(a,element) for a in n.args]
            if name in self.arrays:
                if len(args)!=1:raise ValueError('multidimensional expression '+name)
                return f'{name}[({args[0]})-1]'
            if name in self.functions:return f'{name}({",".join(args)})'
            intrinsic={'exp':'gfk_exp','log':'gfk_log','alog':'gfk_log','sqrt':'FSQRT','sin':'glibc_sinf','cos':'glibc_cosf','tan':'glibc_tanf','asin':'glibc_asinf','acos':'glibc_acosf','atan':'glibc_atanf','atan2':'glibc_atan2f','log10':'u_log10','abs':'fabsf','real':'u480_real','int':'u480_int','nint':'u480_nint','sign':'u480_sign','mod':'u480_mod','present':'u480_present'}
            if name in ('min','max'):
                value=args[0]
                func=('u480_mind' if name=='min' else 'u480_maxd') if self.type(n)=='double' else ('u_min' if name=='min' else 'u_max')
                for b in args[1:]:value=f'{func}({value},{b})'
                return value
            if name=='present':return 'optional_present'
            if name in intrinsic:
                fn=intrinsic[name]
                if self.type(n)=='double':fn={'abs':'fabs','sqrt':'__dsqrt_rn','sign':'copysign'}.get(name,fn)
                return f'{fn}({",".join(args)})'
            if name in NAMES:return f'u480_{name}({",".join(args)},env,status)'
            raise ValueError('unknown function '+name)
        raise ValueError(ast.dump(n))

    def expression(self,s,element=None):
        def double(match):
            name=f'fp64_{len(self.constants)}';self.constants[name]=np.float64(match[0].replace('d','e'));self.types[name]='double';return name
        s=re.sub(r'(?<![\w.])(?:\d+\.?\d*|\.\d+)d[+-]?\d+',double,s)
        return self.node(_syntax(s),element)

    def signature(self):
        args=[]
        for name in self.args:
            kind=self.types.get(name,'int' if name[0] in 'ijklmn' else 'float')
            if name in self.arrays:args.append(f'{"const " if self.intent.get(name)=="in" else ""}{kind}* {name}')
            else:
                readonly=self.intent.get(name)=='in' or (self.intent.get(name)=='unspecified' and name not in self.modified)
                args.append(f'{"const " if readonly else ""}{kind}& {name}')
        if self.name=='sfcdif_urb':args.append('bool optional_present')
        args+=['const U480Env& env','int& status']
        return f'__device__ {"float" if self.result else "void"} u480_{self.name}('+','.join(args)+')'

    def emit(self, trace=False):
        lines=[self.signature()+' {']
        for name,kind in self.types.items():
            if name in self.args or kind=='char' or name in self.functions:continue
            if name in self.constants:lines.append(f'  const {kind} {name}={self.literal(self.constants[name])};');continue
            if name in self.arrays:
                shape=self.arrays[name]
                size='4' if any(x in shape for x in ('nsoil','nroot','num_','km','nlayer')) else self.expression(shape.split(':')[-1])
                lines.append(f'  {kind} {name}[{size}]={{}};')
            else:lines.append(f'  {kind} {name}={{}};')
        if self.name=='root_uptake' and 'k' not in self.types:lines.append('  int k=0;')
        if self.name=='urban':
            lines += ['  const float* dzgr=env.glob+12;','  const float* porimp=env.glob+16;',
                      '  const float* dengimp=env.glob+19;','  const float* ahdiuprf=env.glob+22;',
                      '  const float* alhseason=env.glob+46;','  const float* alhdiuprf=env.glob+50;',
                      '  const float fgr=env.glob[98];','  const int ahoption=env.sw[5],alhoption=env.sw[6],imp_scheme=env.sw[7],iri_scheme=env.sw[8],groption=env.sw[9];',
                      '  const int treeoption=env.sw[11],ttscheme=env.sw[12];',
                      '  const bool distributed_aerodynamics_option=false;',
                      '  float optional_zt=0.0f,optional_veg=0.0f;']
            lines += ['  if(mh_urb>0.0f) {status=2;return;}',
                      f'  if(utype<1 || utype>env.sw[10]) {{status=3;return;}}']
            self.arrays.update({n:'4' for n in ('dzgr','porimp','dengimp','ahdiuprf','alhseason','alhdiuprf')})
            self.types.update({n:'int' for n in ('treeoption','ttscheme','ahoption','alhoption','imp_scheme','iri_scheme','groption')})
        omitted=0
        for num,text in self.body:
            if self.name=='urban' and num==739:
                text='if(ahoption==1.or.iri_scheme==1) then'
            # The existing runtime refuses this unported gridded morphology.
            # Remove its statements instead of inventing numerical stubs.
            if self.name=='urban' and re.match(r'if\s*\(\s*(mh_urb|distributed_aerodynamics_option)',text) and text.endswith('then'):
                lines.append('  {');omitted=1;continue
            if omitted:
                if omitted==1 and text=='else':omitted=0;continue
                if text.startswith('if') and text.endswith('then'):omitted+=1
                if text in ('end if','endif'):
                    omitted-=1
                    if omitted==0:lines.append('  }')
                continue
            lines.append(f'  // source line {num}')
            if self.name=='urban' and num==1008 and re.match(r'tvgp\s*=\s*tvg$',text):
                for correction in ['sroot=0.0','etr=0.0','etg=0.0','tge=(1.0-fvg)*tg+fvg*tvg',
                                   'hvg=0.0','elevg=0.0','ht=0.0','elet=0.0','sleaf=0.0','svg1=0.0','svg2=0.0']:
                    self.statement(correction,lines)
            if self.name=='sstep' and re.match(r'wplus\s*=\s*0\.',text):
                self.statement('smc(nsoil)=smcp(nsoil)',lines)
            self.statement(text,lines)
            if trace and self.name=='root_uptake' and num==5363:
                lines.append('  if(u480_trace) printf("ROOTARGS %d %.17g %.17g %.17g %.17g %.17g\\n",k,(double)smc[k-1],(double)smcmax,(double)smcdry,(double)FSUB(smc[k-1],smcdry),(double)FSUB(smcmax,smcdry));')
            target=re.fullmatch(r'(\w+)\s*=(.*)',text)
            if trace and target and target[1] not in self.arrays and self.types.get(target[1]) in ('float','double','int'):
                lines.append(f'  if(u480_trace) printf("TRACE {num} {target[1]} %.17g\\n",(double){target[1]});')
            array_target=re.fullmatch(r'(\w+)\s*(?:\((.*?)\))?\s*=(.*)',text)
            if trace and array_target and array_target[1] in self.arrays:
                name,index,_=array_target.groups()
                if not index or index==':':
                    size='4' if any(x in self.arrays[name] for x in ('nsoil','nroot','num_','km','nlayer')) else self.expression(self.arrays[name].split(':')[-1])
                    lines.append(f'  if(u480_trace) for(int ti=0;ti<{size};++ti) printf("TRACE {num} {name}[%d] %.17g\\n",ti+1,(double){name}[ti]);')
                else:
                    ix=self.expression(index)
                    lines.append(f'  if(u480_trace) printf("TRACE {num} {name}[%d] %.17g\\n",(int)({ix}),(double){name}[({ix})-1]);')
        if self.result:lines.append(f'  return {self.result};')
        lines.append('}')
        return '\n'.join(lines)

    def statement(self,text,out):
        if text in ('end if','endif','enddo','end do'):out.append('  }');return
        if text=='else':out.append('  } else {');return
        if text.startswith(('elseif','else if')):
            condition=text[text.index('(')+1:text.rindex(')')];out.append('  } else if ('+self.expression(condition)+') {');return
        if text.startswith('if'):
            start=text.index('(');depth=1;end=start+1
            while depth:depth+=(text[end]=='(')-(text[end]==')');end+=1
            condition,tail=text[start+1:end-1],text[end:].strip()
            out.append('  if ('+self.expression(condition)+') {')
            if tail=='then':return
            self.statement(tail,out);out.append('  }');return
        loop=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',text)
        if loop:
            name,bounds=loop.groups();values=_split(bounds);lo,hi=values[:2];step=values[2] if len(values)>2 else '1'
            comparison='>=' if step.strip().startswith('-') else '<='
            out.append(f'  for ({name}={self.expression(lo)};{name}{comparison}{self.expression(hi)};{name}+={self.expression(step)}) {{');return
        if text=='return':out.append('  return'+(' '+self.result if self.result else '')+';');return
        if text=='exit':out.append('  break;');return
        if text.startswith(('write','print','write_message')):return
        if 'fatal_error(' in text or 'wrf_error_fatal(' in text:
            out.append('  status=1; return'+(' 0.0f' if self.result else '')+';');return
        if text.startswith('call '):
            m=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',text);name,args=m.groups()
            if name=='read_param':
                for i,n in enumerate(TABLE):
                    alias='akanda_urban' if n=='akanda_urban' else n
                    out.append(f'  {alias}=env.tab[(utype-1)*{len(TABLE)}+{i}];')
                out += ['  boundr=env.sw[0];boundb=env.sw[1];boundg=env.sw[2];ch_scheme=env.sw[3];ts_scheme=env.sw[4];']
                return
            values=[self.expression(a) for a in _split(args)]
            if name=='sfcdif_urb':
                if len(values)==10:values+=['optional_zt','optional_veg','false']
                else:values+=['true']
            out.append(f'  u480_{name}('+','.join(values+['env','status'])+');');return
        sf=re.fullmatch(r'(ps[lp]\w+)\s*\((\w+)\)\s*=(.*)',text)
        if sf:
            name,arg,rhs=sf.groups();out.append(f'  auto {name}=[&](float {arg}) {{return {self.expression(rhs)};}};');return
        m=re.fullmatch(r'(\w+)\s*(?:\((.*?)\))?\s*=(.*)',text)
        if not m:raise ValueError(self.name+': '+text)
        name,index,rhs=m.groups()
        if self.types.get(name)=='char':return
        if name in self.arrays and (not index or index==':'):
            size='4' if any(x in self.arrays[name] for x in ('nsoil','num_','km','nlayer')) else self.expression(self.arrays[name].split(':')[-1])
            out.append(f'  for(int vi=0;vi<{size};++vi) {name}[vi]={self.expression(rhs,"vi")};')
        else:
            left=name if not index else f'{name}[({self.expression(index)})-1]'
            out.append(f'  {left}={self.expression(rhs)};')


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--trace',action='store_true');a=p.parse_args()
    raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN:raise ValueError('source pin mismatch')
    selected=[];stack=[]
    for row in statements(raw.decode()):
        text=row[1]
        start=re.search(r'(?:subroutine|function)\s+(\w+)\s*\(',text)
        if start and start[1] in NAMES:stack.append([row]);continue
        if stack:
            if text=='contains':continue
            stack[-1].append(row)
            if re.match(r'end\s*(subroutine|function)\b',text):
                selected.append(Routine(stack.pop()))
    prototypes='\n'.join(r.signature()+';' for r in selected)
    listing='\n\n'.join(r.emit(a.trace) for r in selected)
    support='''__device__ inline double u480_mind(double a,double b) {return a<b?a:b;}
__device__ inline double u480_maxd(double a,double b) {return a>b?a:b;}
__device__ inline double u480_powi64(double x,int n) {unsigned int m=n<0?0u-(unsigned int)n:(unsigned int)n;double y=m&1?x:1.0;while(m>>=1) {x=__dmul_rn(x,x);if(m&1)y=__dmul_rn(y,x);}return n<0?__ddiv_rn(1.0,y):y;}
'''
    a.output.write_text('// Generated from the pinned WRF v4.8 numerical statements.\nstruct U480Env {const float* tab; const float* glob; const int* sw;};\n'+('#ifdef __CUDACC__\n#define u480_trace (blockIdx.x*blockDim.x+threadIdx.x==6)\n#else\n#include <cstdio>\nstatic bool u480_trace=false;\n#endif\n' if a.trace else '')+support+prototypes+'\n\n'+listing+'\n')
    urban=next(r for r in selected if r.name=='urban')
    schema=[{'name':n,'type':urban.types.get(n,'float'),'layers':4 if n in urban.arrays else 0} for n in urban.args]
    a.output.with_suffix('.json').write_text(json.dumps({'arguments':schema,'table':TABLE},indent=2))
    bridge=['extern "C" void urban480_host(const unsigned long long* args,const float* table,const float* globals,const int* switches,int* codes,int n) {',
            'U480Env env{table,globals,switches};','for(int row=0;row<n;++row) {int status=0;']
    if a.trace:bridge.append('u480_trace=row==6;')
    for i,v in enumerate(schema):
        kind=v['type'];name=v['name'];storage='int' if kind=='bool' else kind
        if v['layers']:
            bridge.append(f'{kind} {name}[4]; for(int k=0;k<4;++k) {name}[k]=((const {storage}*)args[{i}])[row+k*n];')
        else:bridge.append(f'{kind} {name}=((const {storage}*)args[{i}])[row];')
    bridge.append('u480_urban('+','.join(urban.args+['env','status'])+');')
    for i,v in enumerate(schema):
        kind=v['type'];name=v['name'];storage='int' if kind=='bool' else kind
        if v['layers']:bridge.append(f'for(int k=0;k<4;++k) (({storage}*)args[{i}])[row+k*n]={name}[k];')
        else:bridge.append(f'(({storage}*)args[{i}])[row]={name};')
    bridge+=['codes[row]=status;','}','}']
    a.output.with_suffix('.bridge.cpp').write_text('\n'.join(bridge))
    print(json.dumps({'routines':[r.name for r in selected],'output_bytes':a.output.stat().st_size}))


if __name__=='__main__':main()
