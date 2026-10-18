"""Generate column C++ from the pinned WRF band routines.

This development tool is deliberately closed to unrecognized statements.
It emits a source line on each executable statement for manual review.
"""
from pathlib import Path
import math,re,sys,itertools

ROOT=Path(sys.argv[1]); OUTPUT=Path(sys.argv[2])

def split(text,sep=','):
    out=[]; depth=0; start=0
    for i,ch in enumerate(text):
        if ch=='(':depth+=1
        if ch==')':depth-=1
        if ch==sep and depth==0:out.append(text[start:i].strip());start=i+1
    out.append(text[start:].strip())
    return out

def statements(source):
    acc=''; start=0
    for n,line in enumerate(source.splitlines(),1):
        if not 3113<=n<=12290:continue
        line=line.split('!')[0].strip().lower()
        if not line:continue
        if not acc:start=n
        acc+=' '+line.lstrip('&').rstrip('&')
        if line.endswith('&'):continue
        for part in split(acc,';'):
            if part:yield start,part
        acc=''

class Symbol:
    def __init__(self,name,kind,dims=(),intent='',parameter=None):
        self.name=name;self.kind=kind;self.original=list(dims);self.intent=intent
        self.column=bool(dims and dims[0]=='chunk')
        self.dims=list(dims[1:] if self.column else dims);self.parameter=parameter
    def bounds(self):
        return [tuple(d.split(':')) if ':' in d else ('1',d) for d in self.dims]
    def ctype(self):return {'real':'float','fp':'GReal','integer':'int','logical':'bool'}[self.kind]
    def typename(self):return f'GArray<{self.ctype()},{len(self.dims)}>' if self.dims else self.ctype()

class Routine:
    def __init__(self,line,signature):
        m=re.match(r'subroutine\s+(\w+)\s*\((.*)\)',signature)
        if not m:raise ValueError(signature)
        self.name=m[1];self.args=split(m[2]);self.symbols={};self.body=[];self.data=[];self.loops=[]
    def declaration(self,line):
        m=re.match(r'(real(?:\s*\([^)]*\))?|integer|logical)\s*(.*)',line)
        if not m:return False
        typ,rest=m.groups();kind='fp' if 'fp_kind' in typ else ('real' if typ.startswith('real') else typ)
        attributes,names=rest.split('::',1) if '::' in rest else ('',rest)
        dm=re.search(r'dimension\s*\((.*?)\)',attributes)
        dims=split(dm[1]) if dm else []
        im=re.search(r'intent\s*\((.*?)\)',attributes)
        intent=im[1] if im else ''
        for item in split(names):
            parts=split(item,'=');value=parts[1] if len(parts)>1 else None
            m=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?',parts[0].strip())
            if not m:raise ValueError('declaration '+item)
            name,shape=m.groups()
            self.symbols[name]=Symbol(name,kind,split(shape) if shape else dims,intent,value)
        return True

TOKEN=re.compile(r'\s*(\.true\.|\.false\.|\.[a-z]+\.|(?:\d+\.\d+|\d+\.(?=[ed][+-]?\d|[^a-z]|$)|\.\d+|\d+)(?:[ed][+-]?\d+)?(?:_fp_kind)?|[a-z_]\w*|\*\*|<=|>=|/=|==|[()+*/,:<>=-])')
class Expr:
    def __init__(self,text):
        text=re.sub(r',\s*kind\s*=\s*fp_kind',',fp_kind',text.strip());self.tokens=[];at=0
        while at<len(text):
            m=TOKEN.match(text,at)
            if not m:raise ValueError('expression '+text[at:])
            self.tokens.append(m[1]);at=m.end()
        self.at=0;self.tree=self.parse(0)
        if self.at!=len(self.tokens):raise ValueError('unparsed '+str(self.tokens[self.at:]))
    def take(self):v=self.tokens[self.at];self.at+=1;return v
    def peek(self):return self.tokens[self.at] if self.at<len(self.tokens) else ''
    def parse(self,minp):
        v=self.take()
        if v in ('+','-','.not.'):node=('unary',v,self.parse(7 if v!='.not.' else 4))
        elif v=='(':
            node=self.parse(0)
            if self.take()!=')':raise ValueError('parenthesis')
        elif self.peek()=='(':
            self.take();args=[]
            if self.peek()!=')':
                while True:
                    if self.peek() in (':',',',')'):arg=None
                    else:arg=self.parse(0)
                    if self.peek()==':':
                        self.take();hi=None if self.peek() in (',',')') else self.parse(0)
                        arg=('slice',arg,hi)
                    args.append(arg)
                    if self.peek()!=',':break
                    self.take()
            if self.take()!=')':raise ValueError('call parenthesis')
            node=('call',v,args)
        else:node=('atom',v)
        precedence={'.eqv.':1,'.neqv.':1,'.or.':2,'.and.':3,'.eq.':4,'.ne.':4,'.lt.':4,'.le.':4,'.gt.':4,'.ge.':4,'==':4,'/=':4,'<':4,'>':4,'<=':4,'>=':4,'+':5,'-':5,'*':6,'/':6,'**':8}
        while self.peek() in precedence and precedence[self.peek()]>=minp:
            op=self.take();p=precedence[op];right=self.parse(p if op=='**' else p+1)
            node=('binary',op,node,right)
        return node

def _inexact_float_literal(text):
    """A float32 literal ``...f`` whose reciprocal is not exact."""
    if not re.match(r'^(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?f$',text):return False
    value=float(text[:-1])
    return value!=0.0 and math.frexp(value)[0]!=0.5

def literal(v):
    if v=='.true.':return 'true'
    if v=='.false.':return 'false'
    if not re.match(r'^\d|^\.\d',v):return v
    suffix='_fp_kind' in v; v=v.replace('_fp_kind','')
    if 'd' in v:return v.replace('d','e')
    if '.' in v or 'e' in v:
        return f'GReal({v})' if suffix else v+'f'
    return f'GReal({v})' if suffix else v

class Emitter:
    def __init__(self,routine):self.r=routine;self.indices=[];self.serial=0;self.labels=set()
    def emit(self,node):
        kind=node[0]
        if kind=='atom':
            name=node[1];s=self.r.symbols.get(name)
            if s and s.dims:return name+'('+','.join(self.indices[:len(s.dims)])+')' if self.indices else name
            return literal(name)
        if kind=='unary':return '('+{'-':'-','+':'+','.not.':'!'}[node[1]]+self.emit(node[2])+')'
        if kind=='binary':
            _,op,left,right=node;a=self.emit(left);b=self.emit(right)
            if op=='**':
                if right[0]=='atom':
                    if right[1] in ('2.0','2.','2'):return f'({a}*{a})'
                    if right[1] in ('1.0','1.','1'):return a
                return f'g_pow({a},{b})'
            if op=='/' and right[0]=='atom' and _inexact_float_literal(b):
                # A146: NVVM turns a float x / C into x * RN(1/C) on Blackwell
                # under -ftz=true, which the goddard unit compiles with.  g_div
                # is __fdiv_rn for a float operand and WRF's double division
                # for an fp_kind one (overloads in the header below); a bare
                # __fdiv_rn narrowed the fp_kind operands to float and made
                # g_pow(float, GReal) ambiguous (2.9.0 round 2, test host D GPU leg).
                return f'g_div({a},{b})'
            op={'.eqv.':'==','.neqv.':'!=','.and.':'&&','.or.':'||','.eq.':'==','.ne.':'!=','.lt.':'<','.le.':'<=','.gt.':'>','.ge.':'>=','/=':'!='}.get(op,op)
            return f'({a}{op}{b})'
        if kind=='call':
            _,name,args=node
            s=self.r.symbols.get(name)
            if s:
                args=list(args)
                if s.column:args=args[1:]
                if not s.dims:return name
                ii=iter(self.indices);values=[]
                for arg in args:
                    values.append(next(ii) if arg[0]=='slice' else self.emit(arg))
                return name+'('+','.join(values)+')'
            if name=='sum':
                self.serial+=1;v=f'g_sum_{self.serial}';idx=f'g_si_{self.serial}'
                expr=args[0];symbol=self.r.symbols[expr[1]]
                aa=expr[2][1:] if symbol.column else expr[2]
                slices=[(i,a) for i,a in enumerate(aa) if a[0]=='slice']
                if len(slices)!=1:raise ValueError('sum rank')
                i,a=slices[0];lo,hi=symbol.bounds()[i]
                lo=self.emit(a[1]) if a[1] else self.emit(Expr(lo).tree)
                hi=self.emit(a[2]) if a[2] else self.emit(Expr(hi).tree)
                saved=self.indices;self.indices=[idx];value=self.emit(expr);self.indices=saved
                return f'([&](){{ {symbol.ctype()} {v}=0; for(int {idx}={lo};{idx}<={hi};++{idx}) {v}+={value}; return {v}; }}())'
            if name=='real':return f"{'GReal' if len(args)>1 else 'float'}({self.emit(args[0])})"
            if name=='int':return f'int({self.emit(args[0])})'
            name={'exp':'g_exp','log':'g_log','log10':'g_log10','sqrt':'g_sqrt','max':'g_max','min':'g_min','abs':'g_abs'}.get(name,name)
            return name+'('+','.join(self.emit(a) for a in args)+')'
        raise ValueError('unknown node '+str(node))
    def expr(self,text):return self.emit(Expr(text).tree)
    def assignment(self,left,right):
        lhs=Expr(left).tree;rhs=Expr(right).tree
        symbol=self.r.symbols[lhs[1]]
        args=lhs[2][1:] if lhs[0]=='call' and symbol.column else (lhs[2] if lhs[0]=='call' else [('slice',None,None)]*len(symbol.dims))
        prefix=[];indices=[]
        for dim,arg in enumerate(args):
            if arg[0]!='slice':continue
            idx=f'g_i{len(indices)}';indices.append(idx)
            lo,hi=symbol.bounds()[dim]
            lo=self.emit(arg[1]) if arg[1] else self.expr(lo)
            hi=self.emit(arg[2]) if arg[2] else self.expr(hi)
            prefix.append(f'for(int {idx}={lo};{idx}<={hi};++{idx}){{')
        self.indices=indices
        value=self.emit(lhs)+'='+self.emit(rhs)+';'
        self.indices=[]
        return ''.join(prefix)+value+'}'*len(prefix)
    def statement(self,text):
        text=re.sub(r'^\w+:\s*','',text)
        m=re.match(r'(\d+)\s+(.*)',text)
        label=''
        if m:
            if m[1] in self.labels and m[2]=='continue':return '}'
            label=f'g_label_{m[1]}: ';text=m[2]
        if re.match(r'if\s*\(',text):
            openat=text.index('(');depth=1;j=openat+1
            while depth:
                if text[j]=='(':depth+=1
                if text[j]==')':depth-=1
                j+=1
            cond=self.expr(text[openat+1:j-1]);tail=text[j:].strip()
            return label+f'if({cond})'+('{' if tail=='then' else self.statement(tail))
        if text.startswith('elseif') or text.startswith('else if'):
            return '}else '+self.statement(text[text.index('if'):])
        if re.match(r'^(endif|end if)( |$)',text):return '}'
        if text=='else':return '}else{'
        if text.startswith('enddo') or text.startswith('end do'):return '}'
        if text.startswith('do '):
            loop=text[3:]
            m=re.match(r'(\d+)\s+(.*)',loop)
            if m:self.labels.add(m[1]);loop=m[2]
            name,span=loop.split('=',1);lo,hi,*step=split(span)
            step=step[0] if step else '1'
            cmp='>=' if step.startswith('-') else '<='
            return label+f'for({name.strip()}={self.expr(lo)};{name.strip()}{cmp}{self.expr(hi)};{name.strip()}+={self.expr(step)}){{'
        if text.startswith('go to '):return label+'goto g_label_'+text[6:].strip()+';'
        if text=='continue':return label+';'
        if text.startswith('call '):return label+'g_'+text[5:].split('(')[0]+'('+','.join(self.expr(a) for a in split(text[text.index('(')+1:-1]))+');'
        if '=' in text:
            left,right=text.split('=',1);return label+self.assignment(left.strip(),right.strip())
        raise ValueError('statement '+text)

routines=[];routine=None
for line,text in statements((ROOT/'phys/module_ra_goddard.F').read_text()):
    try:
        if text.startswith('subroutine '):routine=Routine(line,text);routines.append(routine)
        elif text.startswith('end subroutine'):routine=None
        elif routine is None:continue
        elif routine.declaration(text):pass
        elif text.startswith('parameter '):
            for assignment in split(text[text.index('(')+1:-1]):
                name,value=assignment.split('=',1)
                routine.symbols[name.strip()].parameter=value.strip()
        elif text.startswith('data '):routine.data.append((line,text[5:]))
        elif text=='implicit none':pass
        else:routine.body.append((line,text))
    except Exception as error:raise ValueError(f'WRF:{line}: {text}: {error}') from error

PRELUDE=r'''
#ifndef GDD_MAX_LEVELS
#define GDD_MAX_LEVELS 32
#endif
// WRF band internals use fp_kind=SELECTED_REAL_KIND(15,307).
using GReal=double;
constexpr int chunk=1,nband_sw=11,nband_lw=10;
constexpr GReal co2=GReal(401.30e-6f),n2o=GReal(.32e-6f),ch4=GReal(1.79e-6f);
constexpr GReal cfc11=GReal(268.e-12f),cfc12=GReal(503.e-12f),cfc22=GReal(105.e-12f);
constexpr GReal fcld_min=GReal(.01f),taux_min=GReal(.0001f);
constexpr GReal opt_min=GReal(1.e-6f),ssa_min=GReal(0.f),ssa_max=GReal(.999999f),asy_min=GReal(-1.f),asy_max=GReal(1.f),const_tiny=GReal(0x1p-126f);
constexpr bool overcast=true,high=false,trace=true,fast_overcast=true;
template<class T,int R> struct GArray {
 T *data; int lower[R],extent[R];
#ifdef GDD_CHECK_BOUNDS
 const char* label=nullptr;
#endif
 template<int S> __device__ operator GArray<T,S>() const {GArray<T,S> out; out.data=data; for(int i=0;i<S;++i){out.lower[i]=lower[i];out.extent[i]=extent[i];}return out;}
 template<class... I> __device__ T &operator()(I... i) const {
  static_assert(sizeof...(I)==R,"Fortran array rank mismatch");
  int index[R]={int(i)...};int off=0,step=1;
#ifdef GDD_CHECK_BOUNDS
  for(int d=0;d<R;++d)if(index[d]<lower[d]||index[d]>=lower[d]+extent[d])
    printf("GDD bounds %s dim=%d index=%d lower=%d extent=%d\n",label?label:"argument",d,index[d],lower[d],extent[d]);
#endif
  for(int d=0;d<R;++d){off+=(index[d]-lower[d])*step;step*=extent[d];}
  return data[off];
 }
};
template<class A,class B> __device__ auto g_max(A a,B b)->decltype(a+b){return a>b?a:b;}
template<class A,class B> __device__ auto g_min(A a,B b)->decltype(a+b){return a<b?a:b;}
template<class A> __device__ A g_abs(A a){return a<0?-a:a;}
__device__ float g_exp(float x){return gfk_exp(x);}
__device__ double g_exp(double x){return plm_exp(x);}
__device__ float g_log(float x){return gfk_log(x);}
__device__ double g_log(double x){return plm_log(x);}
__device__ float g_log10(float x){return gfk_log(x)*0.4342944819032518f;}
__device__ double g_log10(double x){return plm_log(x)*0.43429448190325182765;}
__device__ float g_sqrt(float x){return __fsqrt_rn(x);}
__device__ double g_sqrt(double x){return __dsqrt_rn(x);}
__device__ float g_pow(float x,float y){return gfk_pow(x,y);}
__device__ double g_pow(double x,double y){return plm_pow(x,y);}
__device__ double g_pow(double x,float y){return plm_pow(x,double(y));}
// A146 literal divisions: IEEE quotient for a float operand (NVVM would turn a
// float x / C into x * RN(1/C) under -ftz=true); an fp_kind (double) operand
// keeps WRF's double division, which NVVM leaves IEEE.
__device__ float g_div(float x,float y){return __fdiv_rn(x,y);}
__device__ double g_div(double x,float y){return x/double(y);}
__device__ double g_div(double x,double y){return x/y;}
'''
output=[PRELUDE];definitions=[]

def const_int(text,constants):
    return int(eval(text,{'__builtins__':{}},constants))

def data_values(expr,constants):
    if expr.startswith('('):
        parts=split(expr[1:-1]);m=re.fullmatch(r'(\w+)\s*=\s*(.*)',parts[-2]) if len(parts)>2 else None
        if m:
            name,lo=m.groups();hi=parts[-1]
            for value in range(const_int(lo,constants),const_int(hi,constants)+1):
                yield from data_values(','.join(parts[:-2]),dict(constants,**{name:value}))
            return
    for value in split(expr):
        if value.startswith('('):yield from data_values(value,constants)
        else:
            m=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?',value)
            if not m:raise ValueError('data target '+value)
            yield m[1], [const_int(i,constants) for i in split(m[2])] if m[2] else None

for r in routines:
    e=Emitter(r);constants={'nband_sw':11,'nband_lw':10};initial={}
    for s in r.symbols.values():
        if s.parameter is not None and s.kind=='integer':constants[s.name]=const_int(s.parameter,constants)
    for line,text in r.data:
        target,values,_=text.split('/',2)
        expanded=[]
        for item in split(values):
            m=re.fullmatch(r'(\d+)\*(.*)',item)
            expanded.extend([m[2]]*int(m[1]) if m else [item])
        locations=list(data_values(target,constants))
        if len(locations)==1 and locations[0][1] is None:
            name=locations[0][0];bounds=r.symbols[name].bounds()
            if bounds:
                lo=[const_int(b[0],constants) for b in bounds];hi=[const_int(b[1],constants) for b in bounds]
                locations=[(name,list(reversed(i))) for i in itertools.product(*[range(a,b+1) for a,b in zip(reversed(lo),reversed(hi))])]
        if len(locations)!=len(expanded):raise ValueError(f'data length WRF:{line} {len(locations)} != {len(expanded)}')
        for (name,indices),value in zip(locations,expanded):
            s=r.symbols[name];bounds=s.bounds()
            size=1;off=0
            if indices is not None:
                for idx,(lo,hi) in zip(indices,bounds):
                    low=const_int(lo,constants);extent=const_int(hi,constants)-low+1
                    off+=(idx-low)*size;size*=extent
            else:size=1
            if name not in initial:initial[name]=['0']*size
            initial[name][off]=e.expr(value)
    writes=set()
    for line,text in r.body:
        if '=' in text and not text.startswith(('do ','if','elseif','else if')):
            writes.add(re.sub(r'[( =].*','',text.split('=',1)[0].strip()))
    args=[]
    for name in r.args:
        s=r.symbols[name]
        if name=='irestrict':args.append('int irestrict')
        elif s.dims:args.append(s.typename()+' '+name)
        else:args.append(s.ctype()+(' &' if name in writes else ' ')+name)
    signature='__device__ void g_'+r.name+'('+','.join(args)+')'
    output.append(signature+';')
    body=[signature+'{']
    for s in r.symbols.values():
        if s.name in r.args:continue
        if s.name=='ic':body.append('int ic=1;');continue
        if s.parameter is not None:
            if s.dims:raise ValueError('array parameter')
            body.append('const '+s.ctype()+' '+s.name+'='+e.expr(s.parameter)+';');continue
        if s.name in initial:
            data=initial[s.name]
            sym=f'gdd_{r.name}_{s.name}'
            if s.dims:
                output.append('__device__ '+s.ctype()+' '+sym+'[]={'+','.join(data)+'};')
                bounds=s.bounds();lower=[e.expr(a) for a,b in bounds];extent=[e.expr(f'({b})-({a})+1') for a,b in bounds]
                body.append(s.typename()+' '+s.name+'={'+sym+', {'+','.join(lower)+'}, {'+','.join(extent)+'}};')
            else:body.append(s.ctype()+' '+s.name+'='+data[0]+';')
        elif s.dims:
            bounds=s.bounds();lower=[e.expr(a) for a,b in bounds];extent=[e.expr(f'({b})-({a})+1') for a,b in bounds]
            bound=[re.sub(r'\bnp\b','GDD_MAX_LEVELS',v) for v in extent]
            body.append(s.ctype()+' '+s.name+'_storage['+'*'.join('('+b+')' for b in bound)+'];')
            body.append(s.typename()+' '+s.name+'={'+s.name+'_storage, {'+','.join(lower)+'}, {'+','.join(extent)+'}};')
        else:body.append(s.ctype()+' '+s.name+';')
    for symbol in r.symbols.values():
        if symbol.dims:
            body.append('#ifdef GDD_CHECK_BOUNDS\n'+symbol.name+'.label="'+r.name+':'+symbol.name+'";\n#endif')
    for line,text in r.body:
        try:
            if r.name=='swrad' and text.startswith('call sw_uvpar '):
                body.append('// Defined visible cloud depth replaces the unassigned WRF scratch array.\nfor(k=1;k<=np;++k) taux(k)=taucl(k,8);')
            body.append(f'// WRF module_ra_goddard.F:{line}\n'+e.statement(text))
        except Exception as error:raise ValueError(f'WRF:{line}: {text}: {error}') from error
    body.append('}')
    definitions.extend(body)
output.extend(definitions)
OUTPUT.write_text('\n'.join(output)+'\n')
print('generated',len(routines),'routines',OUTPUT.stat().st_size,'bytes')
