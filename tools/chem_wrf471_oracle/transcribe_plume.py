"""Transcribe the pinned plume statements to explicit-rounding Python and CUDA.

This is a development tool, not a runtime Fortran interpreter. Generated files
are reviewed and graded against the original Fortran. It refuses unknown syntax
instead of silently omitting a physics statement. Input files are read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np


def split(s, delimiter=','):
    out=[]; start=0; depth=0; quote=None
    for i,c in enumerate(s):
        if quote:
            if c==quote: quote=None
        elif c in "'\"": quote=c
        elif c=='(': depth+=1
        elif c==')': depth-=1
        elif c==delimiter and not depth: out.append(s[start:i].strip()); start=i+1
    out.append(s[start:].strip())
    return out


def statements(path):
    pending=''; first=0
    for n,line in enumerate(path.read_text().splitlines(),1):
        line=line.split('!')[0].strip().lower()
        if not line: continue
        if not pending: first=n
        pending+=line.lstrip('&')
        if pending.endswith('&'): pending=pending[:-1]+' '; continue
        for s in split(pending,';'):
            yield first,s
        pending=''
    assert not pending


@dataclass
class Var:
    dtype: str
    shape: tuple = ()
    parameter: str | None = None
    initial: str | None = None


def declaration(line, variables):
    m=re.match(r'^(real|integer|logical|character)(?:\([^)]*\))?\s*(.*)',line)
    if not m: return False
    dtype={'real':'r','integer':'i','logical':'b','character':'c'}[m[1]]
    rest=m[2]
    if '::' in rest: attr,names=rest.split('::',1)
    else: attr=''; names=rest
    shape=re.search(r'dimension\(([^)]*)\)',attr)
    for item in split(names):
        name,sep,value=item.partition('=')
        name=name.strip()
        a=re.fullmatch(r'(\w+)\s*(?:\((.*)\))?',name)
        if not a: raise ValueError(('declaration',line,item))
        dims=tuple(split(a[2] if a[2] else shape[1])) if a[2] or shape else ()
        variables[a[1]]=Var(dtype,dims,value.strip() if 'parameter' in attr else None,
                           value.strip() if sep and 'parameter' not in attr else None)
    return True


class Routine:
    def __init__(self,arm,name,args,lines,module):
        self.arm=arm; self.name=name; self.args=args; self.lines=lines
        self.vars={}; self.module=module
        self.body=[]; self.parameters={}; self.outputs=[]
        for n,line in lines:
            if declaration(line,self.vars): continue
            m=re.match(r'parameter\s*\((.*)\)$',line)
            if m:
                for item in split(m[1]):
                    v,e=item.split('=',1); self.vars[v.strip()].parameter=e.strip()
                continue
            if line.startswith(('implicit ','use ','type(')): continue
            self.body.append((n,line))
        self.args=[a for a in args if a not in ('coms','errmsg','errflg')]
        if arm=='wrf' and name=='get_fire_properties': self.args.append('heat_flux')
        for a in self.args:
            assert a in self.vars,(name,a)
        for a in self.args:
            if not self.vars[a].shape and any(re.match(r'^'+a+r'\s*=(?!=)',l) for _,l in self.body):
                self.outputs.append(a)
        if name=='esat_pr': self.outputs=['esat_pr']


TOKEN=re.compile(r"\s*(\*\*|\.\w+\.|(?:\d+(?:\.\d*)?|\.\d+)(?:[ed][+-]?\d+)?(?:_\w+)?|[a-z_]\w*|[<>/=]=|[()+\-*/,:<>]|'[^']*')")
PRECEDENCE={'.or.':1,'.and.':2,'==':3,'/=':3,'>':3,'<':3,'>=':3,'<=':3,
            '.eq.':3,'.ne.':3,'.gt.':3,'.lt.':3,'.ge.':3,'.le.':3,
            '+':4,'-':4,'*':5,'/':5,'**':7}


class Parser:
    def __init__(self,s):
        self.tokens=[]; p=0
        while p<len(s):
            m=TOKEN.match(s,p)
            if not m: raise ValueError(('expression',s,s[p:]))
            self.tokens.append(m[1]); p=m.end()
        self.i=0
    def peek(self): return self.tokens[self.i] if self.i<len(self.tokens) else ''
    def take(self): t=self.peek(); self.i+=1; return t
    def expr(self,level=0):
        t=self.take()
        if t in ('-','+','.not.'):
            left=('unary',t,self.expr(6 if t!='.not.' else 2))
        elif t=='(':
            left=self.expr(); assert self.take()==')'
        elif re.match(r'^(?:\d|\.\d)',t) or t in ('.true.','.false.') or t.startswith("'"):
            left=('literal',t)
        elif re.match(r'^\w',t):
            if self.peek()=='(':
                self.take(); args=[]
                while self.peek()!=')':
                    if self.peek()==':': lo=None
                    else: lo=self.expr()
                    if self.peek()==':':
                        self.take(); hi=None if self.peek() in (',',')') else self.expr()
                        lo=('slice',lo,hi)
                    args.append(lo)
                    if self.peek()!=',': break
                    self.take()
                assert self.take()==')'; left=('call',t,args)
            else: left=('name',t)
        else: raise ValueError(('token',t))
        while self.peek() in PRECEDENCE and PRECEDENCE[self.peek()]>=level:
            op=self.take(); p=PRECEDENCE[op]
            left=('binary',op,left,self.expr(p if op=='**' else p+1))
        return left


def parse(s):
    p=Parser(s.strip()); e=p.expr(); assert not p.peek(),(s,p.peek()); return e


class Generator:
    def __init__(self,root,folded=None):
        self.root=root; self.routines={}; self.folded=folded or {}; self.constexpr={}
        self.module={}
        for _,l in statements(root/'chem/module_zero_plumegen_coms.F'):
            declaration(l,self.module)
        self.module={k:v for k,v in self.module.items() if v.parameter is None}
        # Solver diagnostics are private column state, never cross-column carry.
        for name in ('steps','steps1','steps2','top1','top2'):
            self.module[name]=Var('i' if name.startswith('steps') else 'r')
        self.module['initialized']=Var('b')
        self.slot={}; self.nslots=0
        for name,v in self.module.items():
            self.slot[name]=self.nslots
            self.nslots+=3 if len(v.shape)>1 else 1
        self.constants={'nkp':'200','ntime':'200','g':'9.81','r_d':'287.',
                        'cp':'1004.5','rcp':'287./1004.5','p1000mb':'100000.',
                        'rgas':'287.','cpor':'1./(287./1004.5)','p00':'100000.'}
        for arm,path in [('wrf','chem/module_chem_plumerise_scalar.F'),
                         ('gsl','gsl-ccpp-physics/physics/smoke_dust/module_smoke_plumerise.F90')]:
            lines=list(statements(root/path)); i=0
            while i<len(lines):
                n,line=lines[i]
                m=re.match(r'(subroutine|function)\s+(\w+)\s*(?:\((.*)\))?$',line)
                if not m: i+=1; continue
                name=m[2]; args=split(m[3]) if m[3] else []; body=[]; i+=1
                while not lines[i][1].startswith(('end subroutine','end function')):
                    body.append(lines[i]); i+=1
                if name not in ('plumerise','printout'):
                    # Make explicit the only nonstructured control statements.
                    if arm=='wrf' and name=='get_env_condition':
                        body=[(n,('if(zt(k)<znz) exit' if 'go to 13' in l else l))
                              for n,l in body if l not in ("stop ' envir stop 12'",'13 continue')]
                    if arm=='wrf' and name=='evaporate':
                        body=[(n,('if(qh(l)>1.e-10) then' if 'goto 33' in l else
                                  'endif' if l.startswith('33 ') else l)) for n,l in body]
                    if arm=='wrf' and name=='esat_pr':
                        body=[(n,('if(temc<=-40.0) then' if 'goto 230' in l else
                                  'endif' if l.startswith('230 ') else l)) for n,l in body]
                        orig=next(l for n,l in lines if l.startswith('230 esatm'))
                        at=next(j for j,(_,l) in enumerate(body) if l=='endif')
                        body.insert(at+1,(2459,orig[4:]))
                    # Diagnostic I/O has no computed value; izprint is always zero.
                    if name=='makeplume':
                        filtered=[]; skip=0
                        for n,l in body:
                            if re.match(r'if\s*\(izprint',l) or l.startswith('if(ilastprint'):
                                skip=1; continue
                            if skip:
                                if l in ('endif','end if'): skip=0
                                continue
                            if l=='101 continue': continue
                            filtered.append((n,l))
                        body=filtered
                    self.routines[arm+'_'+name]=Routine(arm,name,args,body,self.module)

    def var(self,r,name):
        if name.startswith('state_'): return self.module[name[6:]]
        if name in r.vars: return r.vars[name]
        if name in self.module: return self.module[name]
        if name in self.constants: return Var('i' if name in ('nkp','ntime') else 'r',parameter=self.constants[name])
        raise ValueError(('unknown variable',r.name,name))

    def state(self,r,name):
        return name.startswith('state_') or (name in self.module and name not in r.vars)

    def constant(self,r,e):
        if e[0]=='literal': return True
        if e[0]=='name': return self.var(r,e[1]).parameter is not None
        if e[0]=='unary': return self.constant(r,e[2])
        if e[0]=='binary': return self.constant(r,e[2]) and self.constant(r,e[3])
        if e[0]=='call': return e[1] in ('float','real','int','sqrt','exp','abs','min','max') and all(self.constant(r,a) for a in e[2])
        return False

    def const_source(self,r,e):
        if e[0]=='name': return '('+self.const_source(r,parse(self.var(r,e[1]).parameter))+')'
        if e[0]=='literal': return e[1]
        if e[0]=='unary': return '('+e[1]+self.const_source(r,e[2])+')'
        if e[0]=='binary': return '('+self.const_source(r,e[2])+e[1]+self.const_source(r,e[3])+')'
        if e[0]=='call': return e[1]+'('+','.join(self.const_source(r,a) for a in e[2])+')'
        raise ValueError(e)

    def emit(self,r,e,lang,fold=True):
        tag=e[0]
        if fold and self.constant(r,e):
            source=self.const_source(r,e)
            if source in ('.true.','.false.'):
                return ('True' if source=='.true.' else 'False') if lang=='py' else ('true' if source=='.true.' else 'false'),'b'
            if source.startswith("'"): return source,'c'
            typ=self.emit(r,e,lang,False)[1]
            if typ in ('b','c'): return self.emit(r,e,lang,False)
            if typ=='i':
                value=int(eval(source.replace('/','//')))
                return str(value),'i'
            self.constexpr[source]=len(self.constexpr)
            key=hashlib.sha256(source.encode()).hexdigest()[:16]
            word=self.folded.get(key)
            if word is None: word=0
            return (f'word(0x{word:08x})' if lang=='py' else f'__uint_as_float(0x{word:08x}u)'),'r'
        if tag=='literal':
            t=e[1]
            if t.startswith("'"): return (t if lang=='py' else "'"+t[1]+"'"),'c'
            if t in ('.true.','.false.'): return ('True' if t=='.true.' else 'False') if lang=='py' else ('true' if t=='.true.' else 'false'),'b'
            typ='r' if any(c in t for c in '.ed') else 'i'
            return t.replace('d','e'),typ
        if tag=='name':
            name=e[1]; v=self.var(r,name)
            if v.parameter is not None: return self.emit(r,parse(v.parameter),lang)
            if self.state(r,name):
                slot=self.slot[name.removeprefix('state_')]
                s=f's[{slot}, 0]' if lang=='py' else f's[{slot}][0]'
                if v.shape: s=(f's[{slot}:{slot+3}]' if lang=='py' else f'(s+{slot})') if len(v.shape)>1 else f's[{slot}]'
                elif v.dtype=='i': s=f'int({s})'
                elif v.dtype=='b': s=f'bool({s})'
                return s,'a' if v.shape else v.dtype
            return name,'a' if v.shape else v.dtype
        if tag=='slice':
            lo=self.emit(r,e[1],lang)[0] if e[1] else '1'
            hi=self.emit(r,e[2],lang)[0] if e[2] else '200'
            return f'{lo}:{hi}','slice'
        if tag=='call':
            name=e[1]; args=[self.emit(r,a,lang) for a in e[2]]
            if name in r.vars or name in self.module or name.startswith('state_'):
                base=self.emit(r,('name',name),lang)[0]; typ=self.var(r,name).dtype
                if lang=='py':
                    index=[]
                    for a,t in args:
                        if t=='slice':
                            lo,hi=a.split(':'); index.append(f'{lo}:({hi})+1')
                        else: index.append(a)
                    if len(index)>1: index.reverse()
                    return base+'['+', '.join(index)+']',('a' if any(t=='slice' for _,t in args) else typ)
                if len(args)==1: return base+'['+args[0][0]+']',typ
                return base+'['+args[1][0]+']['+args[0][0]+']',typ
            aa=', '.join(a for a,_ in args)
            if name in ('real','float'): return (f'f32({args[0][0]})' if lang=='py' else f'float({args[0][0]})'),'r'
            if name=='int': return f'int({args[0][0]})','i'
            if name in ('min','max','abs','sqrt','exp'):
                typ='r' if any(t in ('r','a') for _,t in args) else 'i'
                fn={'min':'fmin','max':'fmax','abs':'fabs','sqrt':'fsqrt','exp':'fexp'}[name]
                if typ=='i': fn={'min':'min','max':'max','abs':'abs'}[name]
                return f'{fn}({aa})',typ
            if name=='sum':
                if lang=='cu':
                    arr=e[2][0]; base=self.emit(r,('name',arr[1]),lang)[0]
                    sl=arr[2][0]
                    if len(arr[2])>1: base+='['+self.emit(r,arr[2][1],lang)[0]+']'
                    return f'fsum({base}, '+self.emit(r,sl[1],lang)[0]+', '+self.emit(r,sl[2],lang)[0]+')','r'
                return f'fsum({aa})','r'
            if name=='esat_pr': return f'{r.arm}_esat_pr(s, {aa})','r'
            raise ValueError(('call expression',name))
        if tag=='unary':
            a,t=self.emit(r,e[2],lang)
            op={'-':'-','+':'+','.not.':'not ' if lang=='py' else '!'}[e[1]]
            return f'({op}{a})',t
        if tag=='binary':
            op=e[1]; a,ta=self.emit(r,e[2],lang); b,tb=self.emit(r,e[3],lang)
            if op in ('+','-','*','/','**'):
                typ='r' if 'r' in (ta,tb) or 'a' in (ta,tb) else 'i'
                if typ=='i':
                    return (f'({a} // {b})' if op=='/' and lang=='py' else f'({a} {op} {b})'),'i'
                fn={'+':'fadd','-':'fsub','*':'fmul','/':'fdiv','**':'fpow'}[op]
                if op=='**' and tb=='i': fn='fpowi'
                return f'{fn}({a}, {b})','r'
            op={'.eq.':'==','.ne.':'!=','/=':'!=','.gt.':'>','.lt.':'<','.ge.':'>=','.le.':'<=',
                '.or.':'or' if lang=='py' else '||','.and.':'and' if lang=='py' else '&&'}.get(op,op)
            return f'({a} {op} {b})','b'
        raise ValueError(('node',e))

    def target(self,r,e,lang):
        if e[0]=='name' and self.state(r,e[1]):
            slot=self.slot[e[1].removeprefix('state_')]
            if self.var(r,e[1]).shape: return self.emit(r,e,lang,False)[0]
            return f's[{slot}, 0]' if lang=='py' else f's[{slot}][0]'
        return self.emit(r,e,lang,False)[0]

    def element(self,r,e):
        if e[0]=='name' and self.var(r,e[1]).shape:
            return ('call',e[1],[('name','_a')])
        if e[0]=='call':
            return ('call',e[1],[('name','_a') if a[0]=='slice' else self.element(r,a) for a in e[2]])
        if e[0]=='binary': return ('binary',e[1],self.element(r,e[2]),self.element(r,e[3]))
        if e[0]=='unary': return ('unary',e[1],self.element(r,e[2]))
        return e

    def function(self,r,lang):
        name=r.arm+'_'+r.name; lines=[]; depth=1; stack=[]; lastline=None
        def add(l): lines.append(('    '*depth)+l)
        ret=', '.join(r.outputs)
        if lang=='py': lines.append(f'def {name}(s'+('' if not r.args else ', '+', '.join(r.args))+'):')
        else:
            args=['float s[][202]']
            for a in r.args:
                v=r.vars[a]; typ={'i':'int','r':'float','c':'char','b':'bool'}[v.dtype]
                if len(v.shape)>1:
                    args.append(typ+' (*'+a+')[202]'); continue
                if v.shape: typ+='*'
                elif a in r.outputs: typ+='&'
                args.append(typ+' '+a)
            lines.append('__device__ '+('float' if r.name=='esat_pr' else 'void')+' '+name+'('+', '.join(args)+') {')
        for a,v in r.vars.items():
            if a in r.args or v.parameter is not None: continue
            if lang=='py':
                if v.shape: add(f'{a} = np.zeros('+('(5, 202)' if len(v.shape)>1 else '202')+', dtype='+('np.int32' if v.dtype=='i' else 'np.float32')+')')
                else: add(f'{a} = '+(self.emit(r,parse(v.initial),lang)[0] if v.initial else ('0' if v.dtype=='i' else "''" if v.dtype=='c' else 'f32(0.)')))
            else:
                typ={'i':'int','r':'float','c':'char','b':'bool'}[v.dtype]
                if v.shape:
                    # A routine's own 202-word array is a row of the column's
                    # global workspace (freitas_columns), not a local: local
                    # memory is reserved for every resident thread of the card.
                    if len(v.shape)>1:
                        raise ValueError((r.name,a,'a 2-D local array needs its own workspace rows'))
                    row=self.ws_row; self.ws_row+=1
                    if typ=='float':
                        add(f'float* {a} = s[{row}];  // workspace row {row}: was a 202-word local array')
                        add(f'for (int _z=0; _z<202; ++_z) {a}[_z] = 0.0f;')
                    else:
                        add(f'{typ}* {a} = reinterpret_cast<{typ}*>(s[{row}]);  // workspace row {row}: was a 202-word local array')
                        add(f'for (int _z=0; _z<202; ++_z) {a}[_z] = 0;')
                    continue
                add(typ+' '+a+' = '+(self.emit(r,parse(v.initial),lang)[0] if v.initial else '0')+';')
        if r.name=='makeplume': add(f's[{self.slot["steps"]}, 0] = 0' if lang=='py' else f's[{self.slot["steps"]}][0] = 0;')
        for n,original in r.body:
            l=original.replace('coms%','state_')
            if l.startswith(('data ','stop ','print','write','open','close','format')):
                if l.startswith('data heat_flux'):
                    if r.arm=='wrf' and r.name=='get_fire_properties': continue
                    values=split(l.split('/',2)[1])
                    for k,v in enumerate(values):
                        target=f'heat_flux[{k//2+1}, {k%2+1}]' if lang=='py' else f'heat_flux[{k//2+1}][{k%2+1}]'
                        add(target+' = '+self.emit(r,parse(v),lang)[0]+(';' if lang=='cu' else ''))
                    continue
                if l.startswith(('print','write','open','close')): continue
                if l.startswith('stop '):
                    add("raise ValueError('plume source guard: invalid duration or interpolation bounds')" if lang=='py' else 'return;'); continue
                raise ValueError((r.name,n,l))
            if l in ('101 continue','return'):
                if l=='return':
                    if r.name=='makeplume':
                        for prefix,value in [('top','ztopmax'),('steps',f's[{self.slot["steps"]}, 0]' if lang=='py' else f's[{self.slot["steps"]}][0]')]:
                            add(('if imm == 1:' if lang=='py' else 'if (imm==1) {')); depth+=1
                            add((f's[{self.slot[prefix+"1"]}, 0]' if lang=='py' else f's[{self.slot[prefix+"1"]}][0]')+' = '+value+(';' if lang=='cu' else '')); depth-=1
                            add('else:' if lang=='py' else '} else {'); depth+=1
                            add((f's[{self.slot[prefix+"2"]}, 0]' if lang=='py' else f's[{self.slot[prefix+"2"]}][0]')+' = '+value+(';' if lang=='cu' else '')); depth-=1
                            if lang=='cu': add('}')
                    add(('return '+ret if ret else 'return') if lang=='py' else ('return esat_pr;' if r.name=='esat_pr' else 'return;'))
                continue
            if l in ('endif','end if') or l.startswith(('enddo','end do')):
                block=stack.pop()
                if block.startswith('for:'):
                    _,var,step=block.split(':')
                    if lang=='py': add(f'{var} += {step}')
                depth-=1
                if lang=='cu': add('}')
                continue
            if l.startswith(('elseif','else if','else')):
                depth-=1
                if l=='else': add('else:' if lang=='py' else '} else {')
                else:
                    condition=l[l.index('(')+1:l.rindex(')')]
                    c=self.emit(r,parse(condition),lang)[0]
                    add('elif '+c+':' if lang=='py' else '} else if ('+c+') {')
                depth+=1; continue
            m=re.match(r'(?:\w+:\s*)?do\s+(\w+)\s*=\s*(.*)',l)
            if m:
                var=m[1]; start,end,*steps=split(m[2]); step=steps[0] if steps else '1'
                a=self.emit(r,parse(start),lang)[0]; b=self.emit(r,parse(end),lang)[0]; st=self.emit(r,parse(step),lang)[0]
                v=self.target(r,parse(var),lang)
                if lang=='py':
                    add(v+' = '+a); add('while '+self.emit(r,parse(var),lang)[0]+(' >= ' if step.startswith('-') else ' <= ')+b+':')
                else: add(f'for ({v}={a}; '+self.emit(r,parse(var),lang)[0]+('>=' if step.startswith('-') else '<=')+b+f'; {v}+={st}) {{')
                stack.append('for:'+v+':'+st); depth+=1; continue
            if l.startswith('do while') or l=='do':
                c='True' if lang=='py' else 'true'
                if l!='do': c=self.emit(r,parse(l[l.index('(')+1:l.rindex(')')]),lang)[0]
                add('while '+c+':' if lang=='py' else 'while ('+c+') {'); stack.append('while'); depth+=1; continue
            if l.startswith('if'):
                p=l.index('('); d=1; q=p+1
                while d:
                    if l[q]=='(': d+=1
                    if l[q]==')': d-=1
                    q+=1
                c=self.emit(r,parse(l[p+1:q-1]),lang)[0]; tail=l[q:].strip()
                add('if '+c+':' if lang=='py' else 'if ('+c+') {'); depth+=1
                if tail=='then': stack.append('if'); continue
                inline=True; l=tail
            else: inline=False
            if l.startswith('call '):
                m=re.fullmatch(r'call\s+(\w+)\s*(?:\((.*)\))?',l); fn=m[1]
                if fn=='printout': code='pass' if lang=='py' else ';'
                else:
                    called=self.routines[r.arm+'_'+fn]
                    actual=split(m[2]) if m[2] else []
                    original_args=([a for a in split(next(l for _,l in statements(self.root/('chem/module_chem_plumerise_scalar.F' if r.arm=='wrf' else 'gsl-ccpp-physics/physics/smoke_dust/module_smoke_plumerise.F90')) if re.match(r'(subroutine|function)\s+'+fn+r'\b',l)).split('(',1)[1][:-1])] if '(' in next(l for _,l in statements(self.root/('chem/module_chem_plumerise_scalar.F' if r.arm=='wrf' else 'gsl-ccpp-physics/physics/smoke_dust/module_smoke_plumerise.F90')) if re.match(r'(subroutine|function)\s+'+fn+r'\b',l)) else [])
                    actual=[a for a,p in zip(actual,original_args) if p not in ('coms','errmsg','errflg')]
                    aa=[]; oo=[]
                    for p,a in zip(called.args,actual):
                        e=parse(a)
                        if called.vars[p].shape and e[0]=='call': e=('name',e[1])
                        aa.append(self.emit(r,e,lang)[0])
                        if p in called.outputs: oo.append(self.target(r,e,lang))
                    code=((', '.join(oo)+' = ') if oo and lang=='py' else '')+r.arm+'_'+fn+'(s'+('' if not aa else ', '+', '.join(aa))+')'+(';' if lang=='cu' else '')
                add(code)
            elif l.startswith('stop '):
                add("raise ValueError('plume source duration exceeds heating storage')" if lang=='py' else 'return;')
            elif l in ('exit','cycle','return'):
                if l=='cycle' and lang=='py':
                    block=next(b for b in reversed(stack) if b.startswith(('for:','while')))
                    if block.startswith('for:'):
                        _,v,st=block.split(':'); add(v+' += '+st)
                returning=('return esat_pr' if r.name=='esat_pr' else 'return') if lang=='cu' else ('return '+ret if ret else 'return')
                add({'exit':'break','cycle':'continue','return':returning}[l]+(';' if lang=='cu' else ''))
            elif '=' in l:
                left,right=l.split('=',1)
                if left.strip() in ('errflg','errmsg'): add('pass' if lang=='py' else ';')
                else:
                    e=parse(left); rr,t=self.emit(r,parse(right),lang); v=self.var(r,e[1])
                    ll=self.target(r,e,lang)
                    array=(e[0]=='name' and v.shape) or (e[0]=='call' and any(a[0]=='slice' for a in e[2]))
                    if array:
                        lo='1'; hi='200'
                        if e[0]=='call':
                            sl=e[2][0]
                            lo=self.emit(r,sl[1],lang)[0] if sl[1] else '1'
                            hi=self.emit(r,sl[2],lang)[0] if sl[2] else '200'
                        r.vars['_a']=Var('i')
                        target=self.target(r,self.element(r,e),lang)
                        value=self.emit(r,self.element(r,parse(right)),lang)[0]
                        if lang=='py':
                            add(f'for _a in range({lo}, ({hi})+1):'); depth+=1
                            add(target+' = '+('f32('+value+')' if v.dtype=='r' else value)); depth-=1
                        else: add(f'for (int _a={lo}; _a<={hi}; ++_a) {target} = {value};')
                    elif lang=='py':
                        if e[0]=='name' and v.shape: ll+='[1:201]'
                        if e[0]=='name' and v.shape and t=='a': rr+='[1:201]'
                        if v.dtype=='r' and t!='a': rr='f32('+rr+')'
                        if v.dtype=='i' and t!='a': rr='int('+rr+')'
                        add(ll+' = '+rr)
                    else: add(ll+' = '+rr+';')
                    if r.name=='makeplume' and left.strip() in ('time','state_time') and '+' in right:
                        target=f's[{self.slot["steps"]}, 0]' if lang=='py' else f's[{self.slot["steps"]}][0]'
                        add(target+' += 1'+(';' if lang=='cu' else ''))
            else: raise ValueError(('statement',r.name,n,l))
            if inline:
                depth-=1
                if lang=='cu': add('}')
        assert not stack,(r.name,stack)
        if lang=='py':
            if ret: add('return '+ret)
        else: lines.append('}')
        return '\n'.join(lines)+'\n'

    def write(self,out):
        py=[]; cu=[]
        # Workspace rows after the 114 state rows, the heat table (114..118)
        # and the plume tops (119) that freitas_columns lays out.
        self.ws_row=120
        # Normalize module member spelling after declarations have resolved scope.
        for r in self.routines.values():
            r.body=[(n,l.replace('coms%','state_')) for n,l in r.body]
            for lang,dest in [('py',py),('cu',cu)]:
                source=self.function(r,lang)
                if lang=='cu':
                    for helper in ('fadd','fsub','fmul','fdiv','fabs','fmin','fmax','fexp','fsqrt','fpow','fpowi','fsum'):
                        source=re.sub(r'\b'+helper+r'\(', 'pl_'+helper+'(',source)
                dest.append(source)
        out.mkdir(parents=True,exist_ok=True)
        (out/'plume_generated.py').write_text('\n'.join(py))
        (out/'plume_generated.cuh').write_text('\n'.join(cu))
        (out/'plume_state.json').write_text(json.dumps({k:{'slot':self.slot[k],'type':v.dtype,'shape':v.shape} for k,v in self.module.items()},indent=2))
        (out/'fold_expressions.json').write_text(json.dumps(self.constexpr,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('source_root'); p.add_argument('output'); p.add_argument('--folded')
    a=p.parse_args()
    g=Generator(Path(a.source_root),json.loads(Path(a.folded).read_text()) if a.folded else None)
    g.write(Path(a.output))
