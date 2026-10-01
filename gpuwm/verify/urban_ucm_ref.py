# WRF THIRD-PARTY NOTICE
# WRF was developed at the National Center for Atmospheric Research (NCAR) which is
# operated by the University Corporation for Atmospheric Research (UCAR). NCAR and UCAR
# make no proprietary claims, either statutory or otherwise, to this version and release
# of WRF and consider WRF to be in the public domain for use by any person or entity for
# any purpose without any fee or charge. UCAR requests that any WRF user include this
# notice on any partial or full copies of WRF. WRF is provided on an "AS IS" basis and any
# warranties, either express or implied, including but not limited to implied warranties
# of non-infringement, originality, merchantability and fitness for a particular purpose,
# are disclaimed. In no event shall UCAR be liable for any damages, whatsoever, whether
# direct, indirect, consequential or special, that arise out of or in connection with the
# access, use or performance of WRF, including infringement actions.

"""CPU, float32 reference for WRF v4.7.1's single-layer urban canopy model.

The embedded statement listing is from phys/module_sf_urban.F (WRF v4.7.1).
It is evaluated by a deliberately small, private Fortran REAL(4) evaluator:
no compiler, native library, GPU, fixture lookup, or WRF checkout is needed.
Each arithmetic node rounds separately; integer and real powers are distinct.
Only the requested default morphology and ten-argument SFCDIF arms run.
The listing omits SHADOW, NUDAPT, distributed drag, optional SFCDIF arguments,
and the Lech functions. It contains no fixture values.

Table and switch names are the oracle CSV names. Switch arrays may be supplied
as arrays (``dzr``) or numbered CSV entries (``dzr1`` through ``dzr4``).
All returned scalars and renewed state values are numpy.float32.
"""
from __future__ import annotations

import ast
import math
import re
from functools import lru_cache
from typing import Mapping, MutableMapping

import numpy as np
from gpuwm.core.noahmp_libm import expf, logf, powf, sqrtf

F = np.float32
WRF_SOURCE_SHA256 = '623868c74c4b9d579e9c3811e9c334d731394c2afbfea7d693221626fbf0b0ea'
STATE_FIELDS = 'tr tb tg tc qc uc trl tbl tgl xxxr xxxb xxxg xxxc cmr chr cmc chc cmgr chgr cmcr tgr tgrl smr drelr drelb drelg flxhumr flxhumb flxhumg'.split()
LAYER_FIELDS = {'trl', 'tbl', 'tgl', 'tgrl', 'smr'}
OUTPUT_FIELDS = 'ts qs sh lh lh_kinematic sw alb lw g rn psim psih gz1oz0 u10 v10 th2 q2 ust znt'.split()
FORCING_FIELDS = 'ta qa ua u1 v1 ssg llg rain rhoo za declin cosz omg xlat delt znt chs chs2'.split()
_ALIASES = {x: x + '_urb' for x in 'cmr chr cmc chc cmgr chgr'.split()}


def _powi(x, n):
    """libgcc __powisf2, including its multiplication order."""
    x = F(x)
    n = int(n)
    negative = n < 0
    n = abs(n)
    y = x if n & 1 else F(1)
    while (n := n >> 1):
        x = F(x * x)
        if n & 1:
            y = F(y * x)
    return F(F(1) / y) if negative else y


def _op(kind, a, b):
    real = isinstance(a, (np.floating, float)) or isinstance(b, (np.floating, float))
    if kind == 'pow':
        return _powi(a, b) if isinstance(b, (int, np.integer)) else F(powf(F(a), F(b)))
    if real:
        a, b = F(a), F(b)
    if kind == 'add': return a + b
    if kind == 'sub': return a - b
    if kind == 'mul': return a * b
    if kind == 'div': return a / b if real else math.trunc(a / b)
    raise ValueError(kind)


def _intrinsic(name, *args):
    if name in ('log', 'alog'): return F(logf(F(args[0])))
    if name == 'exp': return F(expf(F(args[0])))
    if name == 'sqrt': return F(sqrtf(F(args[0])))
    if name == 'atan': return F(math.atan(float(args[0])))
    if name == 'log10': return F(math.log10(float(args[0])))
    if name == 'abs': return abs(args[0])
    if name == 'int': return math.trunc(float(args[0]))
    if name == 'real': return F(args[0])
    if name == 'mod': return args[0] - math.trunc(float(args[0] / args[1])) * args[1]
    if name in ('min', 'max'):
        v = args[0]
        for b in args[1:]:
            v = v if (v < b if name == 'min' else v > b) else b
        return v
    if name in ('sin', 'cos', 'tan', 'asin', 'acos'):
        return F(getattr(math, name)(float(args[0])))
    raise NotImplementedError('intrinsic ' + name)


_INTRINSICS = set('log alog exp sqrt atan log10 abs int real mod min max sin cos tan asin acos'.split())


def _split(text):
    result, start, depth = [], 0, 0
    for i, c in enumerate(text):
        depth += (c == '(') - (c == ')')
        if c == ',' and depth == 0:
            result.append(text[start:i].strip()); start = i + 1
    result.append(text[start:].strip())
    return result


def _syntax(text):
    text = text.strip().replace("/=", "!=")
    for old, new in {'.true.': 'True', '.false.': 'False', '.and.': ' and ', '.or.': ' or ', '.not.': ' not ', '.eq.': '==', '.ne.': '!=', '.lt.': '<', '.le.': '<=', '.gt.': '>', '.ge.': '>='}.items():
        text = text.replace(old, new)
    # Fortran accepts 10.or. and 1.e-3, Python requires a token boundary.
    text = re.sub(r'(\d)\.(?=\s*(?:and|or)\b)', r'\1 ', text)
    text = re.sub(r'(\d)[dD]([+-]?\d+)', r'\1e\2', text)
    text = re.sub(r'(?<![\w.])0+(\d+)(?![\w.])', r'\1', text)
    return ast.parse(text.strip(), mode='eval').body


class _Expression:
    def __init__(self, constants):
        self.constants = constants
        self.pool = []

    def literal(self, value):
        self.pool.append(value)
        return f'c[{len(self.pool)-1}]'

    def constant(self, node):
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, (bool, int)) else F(node.value)
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        if isinstance(node, ast.UnaryOp):
            a = self.constant(node.operand)
            if isinstance(node.op, ast.USub): return -a
            if isinstance(node.op, ast.UAdd): return a
            return not a
        if isinstance(node, ast.BinOp):
            a, b = self.constant(node.left), self.constant(node.right)
            kind = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div', ast.Pow:'pow'}[type(node.op)]
            if kind == 'pow' and not isinstance(b, (int, np.integer)):
                return F(math.pow(float(a), float(b)))
            return _op(kind, a, b)
        if isinstance(node, ast.Call) and node.func.id in _INTRINSICS:
            args = [self.constant(a) for a in node.args]
            name = node.func.id
            if name in ('exp', 'log', 'alog', 'atan', 'log10', 'sqrt'):
                fn = 'log' if name == 'alog' else name
                return F(getattr(math, fn)(float(args[0])))
            return _intrinsic(name, *args)
        raise KeyError('not constant')

    def node(self, n):
        try: return self.literal(self.constant(n))
        except KeyError: pass
        if isinstance(n, ast.Name): return f'e[{n.id!r}]'
        if isinstance(n, ast.BinOp):
            kind = {ast.Add:'add', ast.Sub:'sub', ast.Mult:'mul', ast.Div:'div', ast.Pow:'pow'}[type(n.op)]
            return f'_op({kind!r},{self.node(n.left)},{self.node(n.right)})'
        if isinstance(n, ast.UnaryOp):
            op = {ast.USub:'-', ast.UAdd:'+', ast.Not:'not '}[type(n.op)]
            return '(' + op + self.node(n.operand) + ')'
        if isinstance(n, ast.BoolOp):
            return '(' + (' and ' if isinstance(n.op, ast.And) else ' or ').join(self.node(a) for a in n.values) + ')'
        if isinstance(n, ast.Compare):
            ops={ast.Lt:'<',ast.LtE:'<=',ast.Gt:'>',ast.GtE:'>=',ast.Eq:'==',ast.NotEq:'!='}
            return '('+self.node(n.left)+''.join(ops[type(op)]+self.node(b) for op,b in zip(n.ops,n.comparators))+')'
        if isinstance(n, ast.Call):
            name = n.func.id
            args = ','.join(self.node(a) for a in n.args)
            if name == 'present': return f'({n.args[0].id!r} in e)'
            if name in _INTRINSICS: return f'_intrinsic({name!r},{args})'
            return f'_get(e,{name!r},({args},))'
        raise NotImplementedError(ast.dump(n))

    def expr(self, text): return self.node(_syntax(text))


def _get(e, name, indices):
    v = e[name]
    return v(*indices) if callable(v) else v[tuple(int(i)-1 for i in indices)]


def _set(e, name, indices, value, integer=False):
    if indices:
        e[name][tuple(int(i)-1 for i in indices)] = value
    elif isinstance(e.get(name), np.ndarray):
        e[name][...] = value
    else:
        e[name] = int(value) if integer else (value if isinstance(value, bool) else F(value))


def _range(a, b, step=1): return range(int(a), int(b) + (1 if step > 0 else -1), int(step))


def _call(name, e, arguments):
    routine = _routines()[name]
    local = {**e['_globals']}
    for formal, (actual, indices, value) in zip(routine.arguments, arguments):
        local[formal] = value
    local['_globals'] = e['_globals']
    routine.run(local)
    for formal, (actual, indices, value) in zip(routine.arguments, arguments):
        if actual is not None and formal in routine.modified:
            _set(e, actual, indices, local[formal], actual in e['_integers'])


class _Routine:
    """Compile the embedded, restricted statement listing into CPU operations."""
    def __init__(self, source):
        lines = source.strip().splitlines()
        header = re.fullmatch(r'subroutine (\w+)\s*\((.*)\)', lines[0])
        self.name, args = header.groups()
        self.arguments = _split(args)
        self.modified = set()
        integers, constants, declarations = set(), {}, []
        expr = _Expression(constants)
        body, indent = ['def run(e):'], 1
        def emit(s): body.append('    '*indent+s)
        for line in lines[1:-1]:
            if line == 'implicit none': continue
            declaration = re.match(r'^(real|integer|logical|character)(.*)', line)
            if declaration:
                kind, rest = declaration.groups()
                if '::' in rest: attrs, items=rest.split('::',1)
                else: attrs, items='',rest.strip()
                if 'optional' in attrs: continue
                dim=re.search(r'dimension\((.*?)\)',attrs)
                for item in _split(items):
                    nameval=item.split('=',1)
                    v=nameval[0].strip()
                    ar=re.fullmatch(r'(\w+)\((.*)\)',v)
                    name=ar.group(1) if ar else v
                    if kind=='integer': integers.add(name)
                    if 'parameter' in attrs:
                        constants[name]=(int if kind == 'integer' else F)(expr.constant(_syntax(nameval[1])));continue
                    shape=ar.group(2) if ar else (dim.group(1) if dim else None)
                    if shape:
                        sizes=[]
                        for d in _split(shape):
                            if ':' in d:
                                lo,hi=d.split(':'); sizes.append(f'int({expr.expr(hi)})-int({expr.expr(lo)})+1')
                            else: sizes.append(f'int({expr.expr(d)})')
                        declarations.append(f"e.setdefault({name!r},np.zeros(({','.join(sizes)},),dtype=np.float32))")
                    elif kind!='character':
                        declarations.append(f"e.setdefault({name!r},{'0' if kind=='integer' else ('False' if kind=='logical' else 'F(0)')})")
                continue
            if line.startswith('parameter'):
                for item in _split(line[line.index('(')+1:-1]):
                    name,val=item.split('=',1);name=name.strip()
                    constants[name]=(int if name in integers else F)(expr.constant(_syntax(val)))
                continue
            if not body[1:]:
                emit(f"e['_integers']={integers!r}")
                for d in declarations: emit(d)
            if re.match(r'end\s*if$',line) or re.match(r'end\s*do$',line): indent-=1;continue
            if line == 'else': indent-=1;emit('else:');indent+=1;continue
            if line.startswith('else if') or line.startswith('elseif'):
                indent-=1
                condition=line[line.index('(')+1:line.rindex(')')]
                emit('elif '+expr.expr(condition)+':');indent+=1;continue
            if line.startswith('if'):
                start=line.index('(');depth=1;i=start+1
                while depth:
                    depth+=(line[i]=='(')-(line[i]==')');i+=1
                condition,tail=line[start+1:i-1],line[i:].strip()
                emit('if '+expr.expr(condition)+':')
                if tail=='then': indent+=1;continue
                indent+=1;self.statement(tail, expr, emit, integers);indent-=1;continue
            do=re.fullmatch(r'do\s+(\w+)\s*=\s*(.*)',line)
            if do:
                name, bounds=do.groups()
                emit(f"for _i_{name} in _range({','.join(expr.expr(x) for x in _split(bounds))}):")
                indent+=1;emit(f"e[{name!r}]=_i_{name}");continue
            # SFCDIF's statement functions are declarations with executable bodies.
            sf=re.fullmatch(r'(ps[lp]\w+)\s*\((\w+)\)\s*=(.*)',line)
            if sf:
                name,arg,rhs=sf.groups()
                sub=expr.expr(rhs).replace(f'e[{arg!r}]', '_arg')
                emit(f"e[{name!r}]=lambda _arg: {sub}");continue
            self.statement(line, expr, emit, integers)
        self.code='\n'.join(body)
        namespace=dict(globals(), c=expr.pool)
        try: exec(self.code,namespace)
        except Exception:
            raise RuntimeError(self.name+'\n'+self.code)
        self.run=namespace['run']

    def statement(self, line, expr, emit, integers):
        if line=='return': emit('return');return
        if line=='exit': emit('break');return
        if line.startswith('call read_param'): emit('pass');return
        if line.startswith('write') or line.startswith('print') or line.startswith('write_message'): emit('pass');return
        if line.startswith('fatal_error'):
            emit("raise ValueError('ZDC+Z0C+2. >= ZA')");return
        if line.startswith('call '):
            m=re.fullmatch(r'call\s+(\w+)\s*\((.*)\)',line)
            name, args=m.groups(); actuals=[]
            for arg in _split(args):
                n=_syntax(arg)
                if isinstance(n,ast.Name): actuals.append(f'({n.id!r},(),{expr.node(n)})')
                elif isinstance(n,ast.Call) and n.func.id not in _INTRINSICS:
                    idx=','.join(expr.node(a) for a in n.args)
                    actuals.append(f'({n.func.id!r},({idx},),{expr.node(n)})')
                else: actuals.append(f'(None,(),{expr.node(n)})')
            emit(f'_call({name!r},e,[{",".join(actuals)}])')
            self.modified.update(a.func.id if isinstance(a,ast.Call) else a.id for a in map(_syntax,_split(args)) if isinstance(a,(ast.Name,ast.Call)))
            return
        m=re.fullmatch(r'(\w+)\s*(?:\((.*?)\))?\s*=(.*)',line)
        if not m: raise NotImplementedError(self.name+': '+line)
        name,idx,rhs=m.groups()
        indices=','.join(expr.expr(x) for x in _split(idx)) if idx else ''
        emit(f'_set(e,{name!r},({indices}{"," if idx else ""}),{expr.expr(rhs)},{name in integers})')
        self.modified.add(name)


@lru_cache(maxsize=1)
def _routines():
    return {r.name:r for r in (_Routine(s) for s in _SOURCE.split('\n\n'))}


def urban_step(table_row: Mapping[str, np.float32], switches: Mapping[str, object],
               forcing: Mapping[str, np.float32], state: MutableMapping[str, object],
               *, jmonth: int) -> dict[str, np.float32]:
    """Advance one four-layer urban column, renewing ``state`` in place."""
    if switches.get('distributed_aerodynamics_option', False):
        raise NotImplementedError('distributed_aerodynamics_option (slucm_distributed_drag)')
    if F(forcing.get('mh_urb', state.get('mh_urb', table_row.get('mh_urb', switches.get('mh_urb', F(0)))))) > F(0):
        raise NotImplementedError('NUDAPT (mh_urb > 0)')
    e = {k:F(v) for k,v in table_row.items()}
    g = dict(switches)
    for name, n in [('dzr',4),('dzb',4),('dzg',4),('dzgr',4),('porimp',3),('dengimp',3),('ahdiuprf',24),('alhseason',4),('alhdiuprf',48)]:
        g[name]=np.asarray(g[name] if name in g else [g[f'{name}{i}'] for i in range(1,n+1)],dtype=np.float32)
    g['distributed_aerodynamics_option']=False
    e.update(g)
    e.update({k:F(forcing[k]) for k in FORCING_FIELDS})
    for k in STATE_FIELDS:
        e[_ALIASES.get(k,k)] = np.array(state[k],dtype=np.float32,copy=True) if k in LAYER_FIELDS else F(state[k])
    e.update(num_roof_layers=4,num_wall_layers=4,num_road_layers=4,utype=int(table_row['utype']),jmonth=int(jmonth),lsolar=False,mh_urb=F(0),etr=np.zeros(4,dtype=np.float32),_globals=g)
    # Defined irrigation time even when anthropogenic heat is disabled.
    if int(g['iri_scheme']) == 1 or int(g['ahoption']) == 1:
        t=math.trunc(float(F(F(F(F(e['omg']/F(3.14159))*F(180))/F(15))+F(12))+F(0.5)))
        t=t-math.trunc(t/24)*24
        if t<0:t+=24
        e['tloc']=24 if t==0 else t
    with np.errstate(all='ignore'):
        _routines()['urban'].run(e)
    for k in STATE_FIELDS:
        val=e[_ALIASES.get(k,k)]
        if k in LAYER_FIELDS:
            if isinstance(state[k],np.ndarray): state[k][...]=val
            else: state[k]=val.copy()
        else: state[k]=F(val)
    return {k:F(e[k]) for k in OUTPUT_FIELDS}


_SOURCE = r'''
subroutine urban(lsolar, num_roof_layers,num_wall_layers,num_road_layers, dzr,dzb,dzg, utype,ta,qa,ua,u1,v1,ssg,ssgd,ssgq,llg,rain,rhoo, za,declin,cosz,omg,xlat,delt,znt, chs, chs2, tr, tb, tg, tc, qc, uc, trl,tbl,tgl, xxxr, xxxb, xxxg, xxxc, ts,qs,sh,lh,lh_kinematic, sw,alb,lw,g,rn,psim,psih, gz1oz0, cmr_urb,chr_urb,cmc_urb,chc_urb, u10,v10,th2,q2,ust,mh_urb,stdh_urb,lf_urb, lp_urb,hgt_urb,frc_urb,lb_urb,zo_check, cmcr,tgr,tgrl,smr,cmgr_urb,chgr_urb,jmonth, drelr,drelb,drelg,flxhumr,flxhumb,flxhumg, lf_urb_s, z0_urb, vegfrac_in)
implicit none
real, parameter    :: cp=0.24
real, parameter    :: el=583.
real, parameter    :: sig=8.17e-11
real, parameter    :: sig_si=5.67e-8
real, parameter    :: ak=0.4
real, parameter    :: pi=3.14159
real, parameter    :: tetena=7.5
real, parameter    :: tetenb=237.3
real, parameter    :: sratio=0.75
real, parameter    :: cpp=1004.5
real, parameter    :: ell=2.442e+06
real, parameter    :: xka=2.4e-5
logical, intent(in) :: lsolar
integer, intent(in) :: num_roof_layers
integer, intent(in) :: num_wall_layers
integer, intent(in) :: num_road_layers
real, intent(in), dimension(1:num_roof_layers) :: dzr
real, intent(in), dimension(1:num_wall_layers) :: dzb
real, intent(in), dimension(1:num_road_layers) :: dzg
integer, intent(in) :: utype
integer, intent(in) :: jmonth
real, intent(in)    :: ta
real, intent(in)    :: qa
real, intent(in)    :: ua
real, intent(in)    :: u1
real, intent(in)    :: v1
real, intent(in)    :: ssg
real, intent(in)    :: llg
real, intent(in)    :: rain
real, intent(in)    :: rhoo
real, intent(in)    :: za
real, intent(in)    :: declin
real, intent(in)    :: cosz
real, intent(in)    :: omg
real, intent(in)    :: xlat
real, intent(in)    :: delt
real, intent(in)    :: chs,chs2
real, intent(inout) :: ssgd
real, intent(inout) :: ssgq
real, intent(inout) :: cmr_urb
real, intent(inout) :: chr_urb
real, intent(inout) :: cmc_urb
real, intent(inout) :: chc_urb
real, intent(inout) :: znt
real, intent(inout) :: mh_urb
real, intent(inout) :: stdh_urb
real, intent(inout) :: hgt_urb
real, intent(inout) :: lp_urb
real, intent(inout) :: frc_urb
real, intent(inout) :: lb_urb
real, intent(inout), dimension(4) :: lf_urb
real, intent(inout) :: zo_check
real, intent(in) :: lf_urb_s
real, intent(in) :: z0_urb
real, intent(in) :: vegfrac_in
real, intent(out) :: ts
real, intent(out) :: qs
real, intent(out) :: sh
real, intent(out) :: lh
real, intent(out) :: lh_kinematic
real, intent(out) :: sw
real, intent(out) :: alb
real, intent(out) :: lw
real, intent(out) :: g
real, intent(out) :: rn
real, intent(out) :: psim
real, intent(out) :: psih
real, intent(out) :: gz1oz0
real, intent(out) :: u10
real, intent(out) :: v10
real, intent(out) :: th2
real, intent(out) :: q2
real, intent(out) :: ust
real, intent(inout):: tr, tb, tg, tc, qc, uc
real, intent(inout):: xxxr, xxxb, xxxg, xxxc
real, dimension(1:num_roof_layers), intent(inout) :: trl
real, dimension(1:num_wall_layers), intent(inout) :: tbl
real, dimension(1:num_road_layers), intent(inout) :: tgl
real, intent(inout):: flxhumr,flxhumb,flxhumg,drelr,drelb,drelg
real, intent(inout):: tgr,cmcr,chgr_urb,cmgr_urb
real, dimension(1:num_roof_layers), intent(inout) :: smr
real, dimension(1:num_roof_layers), intent(inout) :: tgrl
real :: zr, z0c, z0hc, zdc, svf, r, rw, hgt, ah, alh
real :: sigma_zed
real :: capr, capb, capg, aksr, aksb, aksg, albr, albb, albg
real :: epsr, epsb, epsg, z0r,  z0b,  z0g,  z0hb, z0hg
real :: trlend,tblend,tglend
real :: t1vr, t1vc,th2v
real :: rlmo_urb
real :: akanda_urban
real :: th2x
integer :: boundr, boundb, boundg
integer :: ch_scheme, ts_scheme
logical :: shadow
integer                        :: numdir
real,    dimension ( maxdirs ) :: street_direction
real,    dimension ( maxdirs ) :: street_width
real,    dimension ( maxdirs ) :: building_width
integer                        :: numhgt
real,    dimension ( maxhgts ) :: height_bin
real,    dimension ( maxhgts ) :: hpercent_bin
real :: betr, betb, betg
real :: sx, sd, sq, rx
real :: ur, zc, xlb, bb
real :: z, ribb, ribg, ribc, bhr, bhb, bhg, bhc
real :: tsc, lnet, snet, flxuv, thg, flxth, flxhum, flxg
real :: w, vfgs, vfgw, vfwg, vfws, vfww
real :: houi1, houi2, houi3, houi4, houi5, houi6, houi7, houi8
real :: slx, slx1, slx2, slx3, slx4, slx5, slx6, slx7, slx8
real :: flxthr, flxthb, flxthg
real :: sr, sb, sg, rr, rb, rg
real :: sr1, sr2, sb1, sb2, sg1, sg2, rr1, rr2, rb1, rb2, rg1, rg2
real :: hr, hb, hg, eler, eleb, eleg, g0r, g0b, g0g
real :: alphac, alphar, alphab, alphag
real :: chc, chr, chb, chg, cdc, cdr, cdb, cdg, cdgr
real :: c1r, c1b, c1g, te, tc1, tc2, qc1, qc2, qs0r, qs0b, qs0g,rho,es
real :: desdt
real :: f
real :: dqs0rdtr
real :: drrdtr, dhrdtr, delerdtr, dg0rdtr
real :: dtr, dfdt
real :: fx, fy, gf, gx, gy
real :: dtcdtb, dtcdtg
real :: dqcdtb, dqcdtg
real :: drbdtb1,  drbdtg1,  drbdtb2,  drbdtg2
real :: drgdtb1,  drgdtg1,  drgdtb2,  drgdtg2
real :: drbdtb,   drbdtg,   drgdtb,   drgdtg
real :: dhbdtb,   dhbdtg,   dhgdtb,   dhgdtg
real :: delebdtb, delebdtg, delegdtg, delegdtb
real :: dg0bdtb,  dg0bdtg,  dg0gdtg,  dg0gdtb
real :: dqs0bdtb, dqs0gdtg
real :: dtb, dtg, dtc
real :: theataz
real :: theatas
real :: fai
real :: cnt,snt
real :: ps
real :: tav
real :: xxx, x, z0, z0h, cd, ch
real :: xxx2, psim2, psih2, xxx10, psim10, psih10
real :: psix, psit, psix2, psit2, psix10, psit10
real :: trp, tbp, tgp, tcp, qcp, tst, qst
real :: tsp, chs_local, chs2_local
real :: wdr,hgt2,bw,dhgt
real, parameter :: vonk = 0.4
real :: lambda_f,alpha_macd,beta_macd,lambda_fr
real :: lambda_p, vegfrac
integer :: iteration, k, nudapt
integer :: tloc, tloc2, kalh
real :: flxhumrp, flxhumbp, flxhumgp
real :: drelrp, drelbp, drelgp
real :: tgrp, cmcrp
real, dimension(1:num_roof_layers) :: zsoilr, etr, smrp
integer :: kz
real :: runoff1, runoff2, runoff3
real :: sgr, sgr1, t1vgr, chgr, alphagr
real :: flxthgr, flxhumgr, hgr, elegr, g0gr
real :: qs0gr, epgr, edir, ettr, fv, dtgr, drip
real :: dqs0grdtgr, ecr,rain1, raindr, dew, etar, betgr
real :: df1, rgr, rgrr, rch, yy, zz1, ssoilr
real :: drrdtgr, dhrdtgr, delerdtgr, dg0rdtgr, dfdvt
real,parameter  :: shdfac   = 0.80
real,parameter  :: albv     = 0.20
real,parameter  :: epsv     = 0.93
real,parameter  :: lai      = 1.50
real,parameter  :: cmcmax   = 0.5e-3
real,parameter  :: smcref   = 0.329
real,parameter  :: smcdry   = 0.066
real,parameter  :: smcwlt   = 0.084
real,parameter  :: smcmax   = 0.439
real,parameter  :: rsmax    = 5000
real,parameter  :: rsmin    = 100
real,parameter  :: rgl      = 100
real,parameter  :: cfactr   = 0.5
real,parameter  :: dwsat    = 0.143e-4
real,parameter  :: dksat    = 3.38e-6
real,parameter  :: bexp     = 5.25
real,parameter  :: fxexp    = 2.0
real,parameter  :: zbot     = -2.0
real,parameter  :: quartz   = 0.40
real,parameter  :: csoil    = 2.0e+6
real,parameter  :: hs       = 36
integer,parameter ::  nroot = 2
integer,parameter ::  ngr   = 4
integer,parameter ::  impr  = 1
integer,parameter ::  impb  = 2
integer,parameter ::  impg  = 3
shadow = .false.
if(ahoption==1) then
tloc=mod(int(omg/pi*180./15.+12.+0.5 ),24)
if(tloc.lt.0) tloc=tloc+24
if(tloc==0) tloc=24
endif
if(alhoption==1) then
tloc2=mod(int((omg/pi*180./15.+12.)*2.+0.5 ),48)
if(tloc2.lt.0) tloc2=tloc2+48
if(tloc2==0) tloc2=48
endif
call read_param(utype,zr,sigma_zed,z0c,z0hc,zdc,svf,r,rw,hgt, ah,capr,capb,capg,aksr,aksb,aksg,albr,albb, albg,epsr,epsb,epsg,z0r,z0b,z0g,z0hb,z0hg, betr,betb,betg,trlend,tblend,tglend, numdir, street_direction, street_width, building_width, numhgt, height_bin, hpercent_bin, boundr,boundb,boundg,ch_scheme,ts_scheme, akanda_urban,alh)
if(ahoption==1) ah=ah*ahdiuprf(tloc)
kalh=0
if(alhoption==1) then
if(jmonth==3 .or. jmonth==4 .or. jmonth==5) kalh=1
if(jmonth==6 .or. jmonth==7 .or. jmonth==8) kalh=2
if(jmonth==9 .or. jmonth==10.or. jmonth==11)kalh=3
if(jmonth==12.or. jmonth==1 .or. jmonth==2) kalh=4
endif
if(alhoption==1) alh = alh*alhdiuprf(tloc2)*alhseason(kalh)
if( zdc+z0c+2. >= za) then
fatal_error("zdc + z0c + 2m is larger than the 1st wrf level - stop in subroutine urban - change zdc and z0c" )
end if
if(.not.lsolar) then
ssgd = sratio*ssg
ssgq = ssg - ssgd
endif
ssgd = sratio*ssg
ssgq = ssg - ssgd
w=2.*1.*hgt
vfgs=svf
vfgw=1.-svf
vfwg=(1.-svf)*(1.-r)/w
vfws=vfwg
vfww=1.-2.*vfwg
sx=(ssgd+ssgq)/697.7/60.
sd=ssgd/697.7/60.
sq=ssgq/697.7/60.
rx=llg/697.7/60.
rho=rhoo*0.001
trp=tr
tbp=tb
tgp=tg
tcp=tc
qcp=qc
tsp = (tr * r + tb * w + tg * rw) / (r + rw + w)
flxhumrp = flxhumr
flxhumbp = flxhumb
flxhumgp = flxhumg
drelrp = drelr
drelbp = drelb
drelgp = drelg
tgrp   = tgr
cmcrp  = cmcr
smrp   = smr
if(iri_scheme==1) then
if (tloc==21 .or. tloc==22) then
if(jmonth==5 .or. jmonth==6 .or. jmonth ==7 .or. jmonth==8 .or. jmonth==9 ) then
do kz = 1,2
smrp(kz)= smcref
end do
endif
endif
endif
tav=ta*(1.+0.61*qa)
ps=rhoo*287.*tav/100.
if ( zr + 2. < za ) then
ur=ua*log((zr-zdc)/z0c)/log((za-zdc)/z0c)
zc=0.7*zr
xlb=0.4*(zr-zdc)
bb = 0.4 * zr / ( xlb * alog( ( zr - zdc ) / z0c ) )
uc=ur*exp(-bb*(1.-zc/zr))
else
zc=za/2.
uc=ua/2.
end if
if (ssg > 0.0) then
sr1=sx*(1.-albr)
sgr1=sx*(1.-albv)
sg1=sx*vfgs*(1.-albg)
sb1=sx*vfws*(1.-albb)
sg2=sb1*albb/(1.-albb)*vfgw*(1.-albg)
sb2=sg1*albg/(1.-albg)*vfwg*(1.-albb) + sb1*albb*vfww
sr=sr1
sgr=sgr1
sg=sg1+sg2
sb=sb1+sb2
if (groption ==1) then
snet=r*fgr*sgr+r*(1.-fgr)*sr+w*sb+rw*sg
else
snet=r*sr+w*sb+rw*sg
endif
else
sr=0.
sg=0.
sgr=0.
sb=0.
snet=0.
end if
t1vr = trp* (1.0+ 0.61 * qa)
th2v = (ta + ( 0.0098 * za)) * (1.0+ 0.61 * qa)
rlmo_urb=0.0
call sfcdif_urb (za,z0r,t1vr,th2v,ua,akanda_urban,cmr_urb,chr_urb,rlmo_urb,cdr)
alphar =  rho*cp*chr_urb
chr=alphar/rho/cp/ua
rain1 = rain * 0.001 /3600
if (imp_scheme==1) then
if (rain > 1.) betr=0.7
endif
if (imp_scheme==2) then
if (flxhumrp <= 0.) flxhumrp = 0.
drelr = drelrp+(rain1-flxhumrp*rhoo/1000.)*delt/porimp(impr)
if (rain > 0. .and. drelr < drelrp) drelr = drelrp
if (drelr <= 0.) then
drelr  = 0.0
betr	  = 0.0
elseif (drelr <= dengimp(impr)) then
betr = drelr/dengimp(impr)*porimp(impr)
else
drelr = dengimp(impr)
betr  = porimp(impr)
endif
if ( betr < 1.e-5 ) betr = 0.0
endif
if (ts_scheme == 1) then
do iteration=1,20
es=6.11*exp( (2.5*10.**6./461.51)*(trp-273.15)/(273.15*trp) )
desdt=(2.5*10.**6./461.51)*es/(trp**2.)
qs0r=0.622*es/(ps-0.378*es)
dqs0rdtr = desdt*0.622*ps/((ps-0.378*es)**2.)
rr=epsr*(rx-sig*(trp**4.)/60.)
hr=rho*cp*chr*ua*(trp-ta)*100.
eler=rho*el*chr*ua*betr*(qs0r-qa)*100.
g0r=aksr*(trp-trl(1))/(dzr(1)/2.)
f = sr + rr - hr - eler - g0r
drrdtr = (-4.*epsr*sig*trp**3.)/60.
dhrdtr = rho*cp*chr*ua*100.
delerdtr = rho*el*chr*ua*betr*dqs0rdtr*100.
dg0rdtr =  2.*aksr/dzr(1)
dfdt = drrdtr - dhrdtr - delerdtr - dg0rdtr
dtr = f/dfdt
tr = trp - dtr
trp = tr
if( abs(f) < 0.000001 .and. abs(dtr) < 0.000001 ) exit
end do
call multi_layer(num_roof_layers,boundr,g0r,capr,aksr,trl,dzr,delt,trlend)
else
es=6.11*exp( (2.5*10.**6./461.51)*(trp-273.15)/(273.15*trp) )
qs0r=0.622*es/(ps-0.378*es)
rr=epsr*(rx-sig*(trp**4.)/60.)
hr=rho*cp*chr*ua*(trp-ta)*100.
eler=rho*el*chr*ua*betr*(qs0r-qa)*100.
g0r=sr+rr-hr-eler
call force_restore(capr,aksr,delt,sr,rr,hr,eler,trlend,trp,tr)
trp=tr
end if
flxthr=hr/rho/cp/100.
flxhumr=eler/rho/el/100.
if (groption == 1) then
t1vgr = tgrp* (1.0+ 0.61 * qa)
rlmo_urb=0.0
call sfcdif_urb (za,z0r,t1vgr,th2v,ua,akanda_urban,cmgr_urb,chgr_urb,rlmo_urb,cdgr)
alphagr =  rho*cp*chgr_urb
chgr=alphagr/rho/cp/ua
runoff1 = 0.0
runoff2 = 0.0
runoff3 = 0.0
kz = 1
zsoilr (kz) = - dzgr (kz)
do kz = 2,ngr
zsoilr (kz) = - dzgr(kz) + zsoilr (kz -1)
end do
do iteration=1,100
kz=1
es=6.11*exp( (2.5*10.**6./461.51)*(tgrp-273.15)/(273.15*tgrp) )
desdt=(2.5*10.**6./461.51)*es/(tgrp**2.)
qs0gr=0.622*es/(ps-0.378*es)
dqs0grdtgr = desdt*0.622*ps/((ps-0.378*es)**2.)
epgr=rhoo*chgr*ua*(qs0gr-qa)
if (epgr > 0.0) then
call direvap (edir,epgr,smrp(kz),shdfac,smcmax,smcdry,fxexp)
call transp  (ettr,etr,ecr,shdfac,epgr,cmcrp,cfactr,cmcmax,lai,rsmin,rsmax,rgl,sx, tgrp,ta,qa,smrp,smcwlt,smcref,cpp,ps,chgr,epsv,delt,nroot,ngr,dzgr, zsoilr,hs)
call smflx   (smrp,smr,ngr,cmcrp,cmcr,delt,rain,zsoilr,smcmax,bexp,smcwlt,dksat, dwsat,shdfac,cmcmax,runoff1,runoff2,runoff3,edir,ecr,etr,drip)
else
dew  = - epgr
raindr = rain  + dew * 3600.
edir=0.0
ecr =0.0
ettr=0.0
call smflx (smrp,smr,ngr,cmcrp,cmcr,delt,raindr,zsoilr,smcmax,bexp,smcwlt,dksat, dwsat,shdfac,cmcmax,runoff1,runoff2,runoff3,edir,ecr,etr,drip)
end if
edir = edir  * 1000.0
ettr = ettr  * 1000.0
ecr  = ecr   * 1000.0
etar = edir + ettr + ecr
if (etar < 1.e-20) etar = 0.0
if ( epgr <= 0.0 ) then
betgr = 0.0
else
betgr = etar / epgr
end if
elegr= etar* rho * el /rhoo * 100
call tdfcnd (df1,smr(kz), quartz, smcmax )
df1 = df1 * exp(-2.0 * shdfac)
rgr = epsv*(rx-sig*(ta**4.)/60.)
rgrr= (sgr+rgr) * 697.7 * 60.
rch = rhoo*cpp*chgr
rr1 = epsv*(ta**4) * 6.48e-8 / (ps* chgr) + 1.0
if (rain >  0.0) then
rr2 = rr1 + rain / 3600 * 4.218e+3 / rch
else
rr2 = rr1
end if
yy  = ta + (rgrr / rch - betgr * epgr * ell/ rch) / rr2
zz1 = df1 / (-0.5 * zsoilr (kz) * rch * rr2 ) + 1.0
hgr=rho*cp*chgr*ua*(tgrp-ta)*100.
runoff3 = runoff3/ delt
runoff2 = runoff2+ runoff3
g0gr    = df1*(tgrp-tgrl(1))/(dzgr(1)/2.)/697.7/60
fv = sgr + rgr - hgr - elegr - g0gr
drrdtgr   = (-4.*epsv*sig*tgrp**3.)/60.
dhrdtgr   = rho*cp*chgr*ua*100.
delerdtgr = rho*el*chgr*ua*betgr*dqs0grdtgr*100.
dg0rdtgr  = 2.*df1/ dzgr(kz) * ( 1.0 / 4.1868 ) * 1.e-4
dfdvt = drrdtgr - dhrdtgr - delerdtgr - dg0rdtgr
dtgr = fv/dfdvt/ 6
tgr  = tgrp - dtgr
tgrp = tgr
if( abs(fv) < 0.0001 .and. abs(dtgr) < 0.001 ) then
exit
endif
end do
call shflx (ssoilr,tgrl,smr,smcmax,ngr,tgrp,delt,yy,zz1,zsoilr, trlend,zbot,smcwlt,df1,quartz,csoil,capr)
flxthgr=hgr/rho/cp/100.
flxhumgr=elegr/rho/el/100.
else
flxthgr=0.
flxhumgr=0.
endif
t1vc = tcp* (1.0+ 0.61 * qa)
rlmo_urb=0.0
call sfcdif_urb(za,z0c,t1vc,th2v,ua,akanda_urban,cmc_urb,chc_urb,rlmo_urb,cdc)
alphac =  rho*cp*chc_urb
if (ch_scheme == 1) then
z=zdc
bhb=log(z0b/z0hb)/0.4
bhg=log(z0g/z0hg)/0.4
ribb=(9.8*2./(tcp+tbp))*(tcp-tbp)*(z+z0b)/(uc*uc)
ribg=(9.8*2./(tcp+tgp))*(tcp-tgp)*(z+z0g)/(uc*uc)
call mos(xxxb,alphab,cdb,bhb,ribb,z,z0b,uc,tcp,tbp,rho)
call mos(xxxg,alphag,cdg,bhg,ribg,z,z0g,uc,tcp,tgp,rho)
else
alphab=rho*cp*(6.15+4.18*uc)/1200.
if(uc > 5.) alphab=rho*cp*(7.51*uc**0.78)/1200.
alphag=rho*cp*(6.15+4.18*uc)/1200.
if(uc > 5.) alphag=rho*cp*(7.51*uc**0.78)/1200.
end if
chc=alphac/rho/cp/ua
chb=alphab/rho/cp/uc
chg=alphag/rho/cp/uc
if (imp_scheme==1) then
betb=0.0
if(rain > 1.) betg=0.7
endif
if (imp_scheme==2) then
if (flxhumbp <= 0.) flxhumbp = 0.
if (flxhumgp <= 0.) flxhumgp = 0.
drelb = drelbp+(rain1-flxhumbp*rhoo/1000.)*delt/porimp(impb)
if (rain > 0. .and. drelb < drelbp) drelb = drelbp
drelg = drelgp+(rain1-flxhumgp*rhoo/1000.)*delt/porimp(impg)
if (rain > 0. .and. drelg < drelgp) drelg = drelgp
if (drelb <= 0.) then
drelb   = 0.0
betb    = 0.0
elseif (drelb <= dengimp(impb)) then
betb  = drelb/dengimp(impb)*porimp(impb)
else
drelb = dengimp(impb)
betb = porimp(impb)
endif
if (drelg <= 0.) then
drelg   = 0.0
betg    = 0.0
elseif (drelg <= dengimp(impg)) then
betg  = drelg/dengimp(impg)*porimp(impg)
else
drelg = dengimp(impg)
betg  = porimp(impg)
endif
if ( betg < 1.e-5 ) betg = 0.0
if ( betb < 1.e-5 ) betb = 0.0
endif
if (ts_scheme == 1) then
do iteration=1,20
es=6.11*exp( (2.5*10.**6./461.51)*(tbp-273.15)/(273.15*tbp) )
desdt=(2.5*10.**6./461.51)*es/(tbp**2.)
qs0b=0.622*es/(ps-0.378*es)
dqs0bdtb=desdt*0.622*ps/((ps-0.378*es)**2.)
es=6.11*exp( (2.5*10.**6./461.51)*(tgp-273.15)/(273.15*tgp) )
desdt=(2.5*10.**6./461.51)*es/(tgp**2.)
qs0g=0.622*es/(ps-0.378*es)
dqs0gdtg=desdt*0.622*ps/((ps-0.378*es)**2.)
rg1=epsg*( rx*vfgs +epsb*vfgw*sig*tbp**4./60. -sig*tgp**4./60. )
rb1=epsb*( rx*vfws +epsg*vfwg*sig*tgp**4./60. +epsb*vfww*sig*tbp**4./60. -sig*tbp**4./60. )
rg2=epsg*( (1.-epsb)*(1.-svf)*vfws*rx +(1.-epsb)*(1.-svf)*vfwg*epsg*sig*tgp**4./60. +epsb*(1.-epsb)*(1.-svf)*(1.-2.*vfws)*sig*tbp**4./60. )
rb2=epsb*( (1.-epsg)*vfwg*vfgs*rx +(1.-epsg)*epsb*vfgw*vfwg*sig*(tbp**4.)/60. +(1.-epsb)*vfws*(1.-2.*vfws)*rx +(1.-epsb)*vfwg*(1.-2.*vfws)*sig*epsg*tgp**4./60. +epsb*(1.-epsb)*(1.-2.*vfws)*(1.-2.*vfws)*sig*tbp**4./60. )
rg=rg1+rg2
rb=rb1+rb2
drbdtb1=epsb*(4.*epsb*sig*tb**3.*vfww-4.*sig*tb**3.)/60.
drbdtg1=epsb*(4.*epsg*sig*tg**3.*vfwg)/60.
drbdtb2=epsb*(4.*(1.-epsg)*epsb*sig*tb**3.*vfgw*vfwg +4.*epsb*(1.-epsb)*sig*tb**3.*vfww*vfww)/60.
drbdtg2=epsb*(4.*(1.-epsb)*epsg*sig*tg**3.*vfwg*vfww)/60.
drgdtb1=epsg*(4.*epsb*sig*tb**3.*vfgw)/60.
drgdtg1=epsg*(-4.*sig*tg**3.)/60.
drgdtb2=epsg*(4.*epsb*(1.-epsb)*sig*tb**3.*vfww*vfgw)/60.
drgdtg2=epsg*(4.*(1.-epsb)*epsg*sig*tg**3.*vfwg*vfgw)/60.
drbdtb=drbdtb1+drbdtb2
drbdtg=drbdtg1+drbdtg2
drgdtb=drgdtb1+drgdtb2
drgdtg=drgdtg1+drgdtg2
hb=rho*cp*chb*uc*(tbp-tcp)*100.
hg=rho*cp*chg*uc*(tgp-tcp)*100.
dtcdtb=w*alphab/(rw*alphac+rw*alphag+w*alphab)
dtcdtg=rw*alphag/(rw*alphac+rw*alphag+w*alphab)
dhbdtb=rho*cp*chb*uc*(1.-dtcdtb)*100.
dhbdtg=rho*cp*chb*uc*(0.-dtcdtg)*100.
dhgdtg=rho*cp*chg*uc*(1.-dtcdtg)*100.
dhgdtb=rho*cp*chg*uc*(0.-dtcdtb)*100.
eleb=rho*el*chb*uc*betb*(qs0b-qcp)*100.
eleg=rho*el*chg*uc*betg*(qs0g-qcp)*100.
dqcdtb=w*alphab*betb*dqs0bdtb/(rw*alphac+rw*alphag*betg+w*alphab*betb)
dqcdtg=rw*alphag*betg*dqs0gdtg/(rw*alphac+rw*alphag*betg+w*alphab*betb)
delebdtb=rho*el*chb*uc*betb*(dqs0bdtb-dqcdtb)*100.
delebdtg=rho*el*chb*uc*betb*(0.-dqcdtg)*100.
delegdtg=rho*el*chg*uc*betg*(dqs0gdtg-dqcdtg)*100.
delegdtb=rho*el*chg*uc*betg*(0.-dqcdtb)*100.
g0b=aksb*(tbp-tbl(1))/(dzb(1)/2.)
g0g=aksg*(tgp-tgl(1))/(dzg(1)/2.)
dg0bdtb=2.*aksb/dzb(1)
dg0bdtg=0.
dg0gdtg=2.*aksg/dzg(1)
dg0gdtb=0.
f = sb + rb - hb - eleb - g0b
fx = drbdtb - dhbdtb - delebdtb - dg0bdtb
fy = drbdtg - dhbdtg - delebdtg - dg0bdtg
gf = sg + rg - hg - eleg - g0g
gx = drgdtb - dhgdtb - delegdtb - dg0gdtb
gy = drgdtg - dhgdtg - delegdtg - dg0gdtg
dtb =  (gf*fy-f*gy)/(fx*gy-gx*fy)
dtg = -(gf+gx*dtb)/gy
tb = tbp + dtb
tg = tgp + dtg
tbp = tb
tgp = tg
tc1=rw*alphac+rw*alphag+w*alphab
tc2=rw*alphac*ta+rw*alphag*tgp+w*alphab*tbp
tc=tc2/tc1
qc1=rw*alphac+rw*alphag*betg+w*alphab*betb
qc2=rw*alphac*qa+rw*alphag*betg*qs0g+w*alphab*betb*qs0b
qc=qc2/qc1
dtc=tcp - tc
tcp=tc
qcp=qc
if( abs(f) < 0.000001 .and. abs(dtb) < 0.000001 .and. abs(gf) < 0.000001 .and. abs(dtg) < 0.000001 .and. abs(dtc) < 0.000001) exit
end do
call multi_layer(num_wall_layers,boundb,g0b,capb,aksb,tbl,dzb,delt,tblend)
call multi_layer(num_road_layers,boundg,g0g,capg,aksg,tgl,dzg,delt,tglend)
else
es=6.11*exp((2.5*10.**6./461.51)*(tbp-273.15)/(273.15*tbp) )
qs0b=0.622*es/(ps-0.378*es)
es=6.11*exp((2.5*10.**6./461.51)*(tgp-273.15)/(273.15*tgp) )
qs0g=0.622*es/(ps-0.378*es)
rg1=epsg*( rx*vfgs +epsb*vfgw*sig*tbp**4./60. -sig*tgp**4./60. )
rb1=epsb*( rx*vfws +epsg*vfwg*sig*tgp**4./60. +epsb*vfww*sig*tbp**4./60. -sig*tbp**4./60. )
rg2=epsg*( (1.-epsb)*(1.-svf)*vfws*rx +(1.-epsb)*(1.-svf)*vfwg*epsg*sig*tgp**4./60. +epsb*(1.-epsb)*(1.-svf)*(1.-2.*vfws)*sig*tbp**4./60. )
rb2=epsb*( (1.-epsg)*vfwg*vfgs*rx +(1.-epsg)*epsb*vfgw*vfwg*sig*(tbp**4.)/60. +(1.-epsb)*vfws*(1.-2.*vfws)*rx +(1.-epsb)*vfwg*(1.-2.*vfws)*sig*epsg*tgp**4./60. +epsb*(1.-epsb)*(1.-2.*vfws)*(1.-2.*vfws)*sig*tbp**4./60. )
rg=rg1+rg2
rb=rb1+rb2
hb=rho*cp*chb*uc*(tbp-tcp)*100.
eleb=rho*el*chb*uc*betb*(qs0b-qcp)*100.
g0b=sb+rb-hb-eleb
hg=rho*cp*chg*uc*(tgp-tcp)*100.
eleg=rho*el*chg*uc*betg*(qs0g-qcp)*100.
g0g=sg+rg-hg-eleg
call force_restore(capb,aksb,delt,sb,rb,hb,eleb,tblend,tbp,tb)
call force_restore(capg,aksg,delt,sg,rg,hg,eleg,tglend,tgp,tg)
tbp=tb
tgp=tg
tc1=rw*alphac+rw*alphag+w*alphab
tc2=rw*alphac*ta+rw*alphag*tgp+w*alphab*tbp
tc=tc2/tc1
qc1=rw*alphac+rw*alphag*betg+w*alphab*betb
qc2=rw*alphac*qa+rw*alphag*betg*qs0g+w*alphab*betb*qs0b
qc=qc2/qc1
tcp=tc
qcp=qc
end if
flxthb=hb/rho/cp/100.
flxhumb=eleb/rho/el/100.
flxthg=hg/rho/cp/100.
flxhumg=eleg/rho/el/100.
if(groption==1) then
if(ahoption==1) then
flxth  = ((1.-fgr)*r*flxthr + fgr*r*flxthgr + w*flxthb + rw*flxthg)+ ah/rhoo/cpp
else
flxth  = ((1.-fgr)*r*flxthr + fgr*r*flxthgr + w*flxthb + rw*flxthg)
endif
if(alhoption==1) then
flxhum  = ((1.-fgr)*r*flxhumr + fgr*r*flxhumgr + w*flxhumb + rw*flxhumg)+ alh/rhoo/ell
else
flxhum  = ((1.-fgr)*r*flxhumr + fgr*r*flxhumgr + w*flxhumb + rw*flxhumg)
endif
flxuv  = ((1.-fgr)*r*cdr + fgr*r*cdgr + rw*cdc )*ua*ua
flxg =   ((1.-fgr)*r*g0r + fgr*r*g0gr+ w*g0b + rw*g0g)
lnet =   (1.-fgr) * r * rr + fgr *r* rgr + w * rb +  rw * rg
else
if(ahoption==1) then
flxth  = ( r*flxthr  + w*flxthb  + rw*flxthg ) + ah/rhoo/cpp
else
flxth  = ( r*flxthr  + w*flxthb  + rw*flxthg )
endif
if(alhoption==1) then
flxhum = ( r*flxhumr + w*flxhumb + rw*flxhumg )+ alh/rhoo/ell
else
flxhum = ( r*flxhumr + w*flxhumb + rw*flxhumg )
endif
flxuv  = ( r*cdr + rw*cdc )*ua*ua
flxg =   ( r*g0r + w*g0b + rw*g0g )
lnet =     r*rr + w*rb + rw*rg
endif
sh    = flxth  * rhoo * cpp
lh    = flxhum * rhoo * ell
lh_kinematic = flxhum * rhoo
lw    = llg - (lnet*697.7*60.)
sw    = ssg - (snet*697.7*60.)
alb   = 0.
if( abs(ssg) > 0.0001) alb = sw/ssg
g = -flxg*697.7*60.
rn = (snet+lnet)*697.7*60.
ust = sqrt(flxuv)
tst = -flxth/ust
qst = -flxhum/ust
z0 = z0c
z0h = z0hc
z = za - zdc
znt = z0
xxx = 0.4*9.81*z*tst/ta/ust/ust
if ( xxx >= 1. ) xxx = 1.
if ( xxx <= -5. ) xxx = -5.
if ( xxx > 0 ) then
psim = -5. * xxx
psih = -5. * xxx
else
x = (1.-16.*xxx)**0.25
psim = 2.*alog((1.+x)/2.) + alog((1.+x*x)/2.) - 2.*atan(x) + pi/2.
psih = 2.*alog((1.+x*x)/2.)
end if
gz1oz0 = alog(z/z0)
cd = 0.4**2./(alog(z/z0)-psim)**2.
chs_local = 0.4 * ust / (alog(z / z0h) - psih)
ts = ta + flxth/chs
qs = qa + flxhum/chs
xxx2 = (2./z)*xxx
if ( xxx2 >= 1. ) xxx2 = 1.
if ( xxx2 <= -5. ) xxx2 = -5.
if ( xxx2 > 0 ) then
psim2 = -5. * xxx2
psih2 = -5. * xxx2
else
x = (1.-16.*xxx2)**0.25
psim2 = 2.*alog((1.+x)/2.) + alog((1.+x*x)/2.) - 2.*atan(x) + 2.*atan(1.)
psih2 = 2.*alog((1.+x*x)/2.)
end if
xxx10 = (10./z)*xxx
if ( xxx10 >= 1. ) xxx10 = 1.
if ( xxx10 <= -5. ) xxx10 = -5.
if ( xxx10 > 0 ) then
psim10 = -5. * xxx10
psih10 = -5. * xxx10
else
x = (1.-16.*xxx10)**0.25
psim10 = 2.*alog((1.+x)/2.) + alog((1.+x*x)/2.) - 2.*atan(x) + 2.*atan(1.)
psih10 = 2.*alog((1.+x*x)/2.)
end if
psix = alog(z/z0) - psim
psit = alog(z/z0h) - psih
psix2 = alog(2./z0) - psim2
psit2 = alog(2./z0h) - psih2
psix10 = alog(10./z0) - psim10
psit10 = alog(10./z0h) - psih10
u10 = u1 * (psix10/psix)
v10 = v1 * (psix10/psix)
th2 = ts + (ta-ts) *(chs/chs2)
q2 = qs + (qa-qs)*(psit2/psit)
end subroutine urban

subroutine mos(xxx,alpha,cd,b1,rib,z,z0,ua,ta,tsf,rho)
implicit none
real, parameter     :: cp=0.24
real, intent(in)    :: b1, z, z0, ua, ta, tsf, rho
real, intent(out)   :: alpha, cd
real, intent(inout) :: xxx, rib
real                :: xxx0, x, x0, faih, dpsim, dpsih
real                :: f, df, xxxp, us, ts, al, xkb, dd, psim, psih
integer             :: newt
integer, parameter  :: newt_end=10
if(rib <= -15.) rib=-15.
if(rib < 0.) then
do newt=1,newt_end
if(xxx >= 0.) xxx=-1.e-3
xxx0=xxx*z0/(z+z0)
x=(1.-16.*xxx)**0.25
x0=(1.-16.*xxx0)**0.25
psim=alog((z+z0)/z0) -alog((x+1.)**2.*(x**2.+1.)) +2.*atan(x) +alog((x0+1.)**2.*(x0**2.+1.)) -2.*atan(x0)
faih=1./sqrt(1.-16.*xxx)
psih=alog((z+z0)/z0)+0.4*b1 -2.*alog(sqrt(1.-16.*xxx)+1.) +2.*alog(sqrt(1.-16.*xxx0)+1.)
dpsim=(1.-16.*xxx)**(-0.25)/xxx -(1.-16.*xxx0)**(-0.25)/xxx
dpsih=1./sqrt(1.-16.*xxx)/xxx -1./sqrt(1.-16.*xxx0)/xxx
f=rib*psim**2./psih-xxx
df=rib*(2.*dpsim*psim*psih-dpsih*psim**2.) /psih**2.-1.
xxxp=xxx
xxx=xxxp-f/df
if(xxx <= -10.) xxx=-10.
end do
else if(rib >= 0.142857) then
xxx=0.714
psim=alog((z+z0)/z0)+7.*xxx
psih=psim+0.4*b1
else
al=alog((z+z0)/z0)
xkb=0.4*b1
dd=-4.*rib*7.*xkb*al+(al+xkb)**2.
if(dd <= 0.) dd=0.
xxx=(al+xkb-2.*rib*7.*al-sqrt(dd))/(2.*(rib*7.**2-7.))
psim=alog((z+z0)/z0)+7.*min(xxx,0.714)
psih=psim+0.4*b1
end if
us=0.4*ua/psim
if(us <= 0.01) us=0.01
ts=0.4*(ta-tsf)/psih
cd=us*us/ua**2.
alpha=rho*cp*0.4*us/psih
return
end subroutine mos

subroutine multi_layer(km,bound,g0,cap,aks,tsl,dz,delt,tslend)
implicit none
real, intent(in)                   :: g0
real, intent(in)                   :: cap
real, intent(in)                   :: aks
real, intent(in)                   :: delt
real, intent(in)                   :: tslend
integer, intent(in)                :: km
integer, intent(in)                :: bound
real, dimension(km), intent(in)    :: dz
real, dimension(km), intent(inout) :: tsl
real, dimension(km)                :: a, b, c, d, x, p, q
real                               :: dzend
integer                            :: k
dzend=dz(km)
a(1) = 0.0
b(1) = cap*dz(1)/delt +2.*aks/(dz(1)+dz(2))
c(1) = -2.*aks/(dz(1)+dz(2))
d(1) = cap*dz(1)/delt*tsl(1) + g0
do k=2,km-1
a(k) = -2.*aks/(dz(k-1)+dz(k))
b(k) = cap*dz(k)/delt + 2.*aks/(dz(k-1)+dz(k)) + 2.*aks/(dz(k)+dz(k+1))
c(k) = -2.*aks/(dz(k)+dz(k+1))
d(k) = cap*dz(k)/delt*tsl(k)
end do
if(bound == 1) then
a(km) = -2.*aks/(dz(km-1)+dz(km))
b(km) = cap*dz(km)/delt + 2.*aks/(dz(km-1)+dz(km))
c(km) = 0.0
d(km) = cap*dz(km)/delt*tsl(km)
else
a(km) = -2.*aks/(dz(km-1)+dz(km))
b(km) = cap*dz(km)/delt + 2.*aks/(dz(km-1)+dz(km)) + 2.*aks/(dz(km)+dzend)
c(km) = 0.0
d(km) = cap*dz(km)/delt*tsl(km) + 2.*aks*tslend/(dz(km)+dzend)
end if
p(1) = -c(1)/b(1)
q(1) =  d(1)/b(1)
do k=2,km
p(k) = -c(k)/(a(k)*p(k-1)+b(k))
q(k) = (-a(k)*q(k-1)+d(k))/(a(k)*p(k-1)+b(k))
end do
x(km) = q(km)
do k=km-1,1,-1
x(k) = p(k)*x(k+1)+q(k)
end do
do k=1,km
tsl(k) = x(k)
end do
return
end subroutine multi_layer

subroutine force_restore(cap,aks,delt,s,r,h,le,tslend,tsp,ts)
real, intent(in)  :: cap,aks,delt,s,r,h,le,tslend,tsp
real, intent(out) :: ts
real              :: c1,c2
c2=24.*3600./2./3.14159
c1=sqrt(0.5*c2*cap*aks)
ts = tsp + delt*( (s+r-h-le)/c1 -(tsp-tslend)/c2 )
end subroutine force_restore

subroutine sfcdif_urb (zlm,z0,thz0,thlm,sfcspd,akanda,akms,akhs,rlmo,cd)
implicit none
real     wwst, wwst2, g, vkrm, excm, beta, btg, elfc, wold, wnew
real     pihf, epsu2, epsust, epsit, epsa, ztmin, ztmax, hpbl, sqvisc
real     ric, rric, fhneu, rfc,rlmo_thr, rfac, zz, pslmu, pslms, pslhu, pslhs
real     xx, pspmu, yy, pspms, psphu, psphs, zlm, z0, thz0, thlm
real     sfcspd, akanda, akms, akhs, zu, zt, rdz, cxch
real     dthv, du2, btgh, wstar2, ustar, zslu, zslt, rlogu, rlogt
real     rlmo, zetalt, zetalu, zetau, zetat, xlu4, xlt4, xu4, xt4
real     xlu, xlt, xu, xt, psmz, simm, pshz, simh, ustark, rlmn, rlma
integer  itrmx, ilech, itr
real,    intent(out) :: cd
parameter (wwst = 1.2,wwst2 = wwst * wwst,g = 9.8,vkrm = 0.40, excm = 0.001 ,beta = 1./270.,btg = beta * g,elfc = vkrm * btg ,wold =.15,wnew = 1. - wold,itrmx = 05, pihf = 3.14159265/2.)
parameter (epsu2 = 1.e-4,epsust = 0.07,epsit = 1.e-4,epsa = 1.e-8 ,ztmin = -5.,ztmax = 1.,hpbl = 1000.0 ,sqvisc = 258.2)
parameter (ric = 0.183,rric = 1.0/ ric,fhneu = 0.8,rfc = 0.191 ,rlmo_thr = 0.001,rfac = ric / (fhneu * rfc * rfc))
pspmu (xx)= -2.* log ( (xx +1.)*0.5) - log ( (xx * xx +1.)*0.5) +2.* atan (xx) - pihf
pspms (yy)= 5.* yy
psphu (xx)= -2.* log ( (xx * xx +1.)*0.5)
psphs (yy)= 5.* yy
ilech = 0
zu = z0
rdz = 1./ zlm
cxch = excm * rdz
dthv = thlm - thz0
du2 = max (sfcspd * sfcspd,epsu2)
btgh = btg * hpbl
if (btgh * akhs * dthv .ne. 0.0) then
wstar2 = wwst2* abs (btgh * akhs * dthv)** (2./3.)
else
wstar2 = 0.0
end if
ustar = max (sqrt (akms * sqrt (du2+ wstar2)),epsust)
zt = exp (2.0-akanda*(sqvisc**2 * ustar * z0)**0.25)* z0
zslu = zlm + zu
zslt = zlm + zt
rlogu = log (zslu / zu)
rlogt = log (zslt / zt)
rlmo = elfc * akhs * dthv / ustar **3
do itr = 1,itrmx
zetalt = max (zslt * rlmo,ztmin)
rlmo = zetalt / zslt
zetalu = zslu * rlmo
zetau = zu * rlmo
zetat = zt * rlmo
if (rlmo .lt. 0.0)then
xlu4 = 1. -16.* zetalu
xlt4 = 1. -16.* zetalt
xu4 = 1. -16.* zetau
xt4 = 1. -16.* zetat
xlu = sqrt (sqrt (xlu4))
xlt = sqrt (sqrt (xlt4))
xu = sqrt (sqrt (xu4))
xt = sqrt (sqrt (xt4))
psmz = pspmu (xu)
simm = pspmu (xlu) - psmz + rlogu
pshz = psphu (xt)
simh = psphu (xlt) - pshz + rlogt
else
zetalu = min (zetalu,ztmax)
zetalt = min (zetalt,ztmax)
psmz = pspms (zetau)
simm = pspms (zetalu) - psmz + rlogu
pshz = psphs (zetat)
simh = psphs (zetalt) - pshz + rlogt
end if
ustar = max (sqrt (akms * sqrt (du2+ wstar2)),epsust)
zt = exp (2.0-akanda*(sqvisc**2 * ustar * z0)**0.25)* z0
zslt = zlm + zt
rlogt = log (zslt / zt)
ustark = ustar * vkrm
akms = max (ustark / simm,cxch)
akhs = max (ustark / simh,cxch)
if (btgh * akhs * dthv .ne. 0.0) then
wstar2 = wwst2* abs (btgh * akhs * dthv)** (2./3.)
else
wstar2 = 0.0
end if
rlmn = elfc * akhs * dthv / ustar **3
rlma = rlmo * wold+ rlmn * wnew
rlmo = rlma
end do
cd = ustar*ustar/sfcspd**2
end subroutine sfcdif_urb

subroutine direvap (edir,etp,smc,shdfac,smcmax,smcdry,fxexp)
real, intent(in)  :: etp,smc,shdfac,smcmax,smcdry,fxexp
real, intent(out) :: edir
real              :: fx, sratio
sratio = (smc - smcdry) / (smcmax - smcdry)
if (sratio > 0.) then
fx = sratio**fxexp
fx = max ( min ( fx, 1. ) ,0. )
else
fx = 0.
endif
edir = fx * ( 1.0- shdfac ) * etp * 0.001
end subroutine direvap

subroutine transp (ett,et,ec,shdfac,etp1,cmc,cfactr,cmcmax,lai,rsmin,rsmax,rgl,sx, ts,ta,qa,smc,smcwlt,smcref,cpp,ps,ch,epsv,delt, nroot,nsoil, dzvr, zsoil, hs)
integer, intent(in)   :: nroot, nsoil
real, intent(in)  :: shdfac,etp1,cmc,cfactr,cmcmax,lai,rsmin,rsmax,rgl,sx,ta
real, intent(in)  :: ts,qa, smcwlt, smcref, cpp, ps,ch, epsv, delt, hs
real, dimension(1:nsoil), intent(in)   :: zsoil, dzvr, smc
real, dimension(1:nsoil), intent(inout):: et
real, intent(out) :: ec, ett
real              :: rc, rcs, rct, rcq, rcsoil, ff, ws, slv, desdt
real              :: sigma, pc, cmc2ms, sgx, denom, rtx, ett1
integer           :: k
real, dimension(1:nroot) ::  part, gx
slv    = 2.501e+6
sigma  = 5.67e-8
ett    = 0.0
do k = 1, nsoil
et(k) = 0.
end do
rcs = 0.0
rct = 0.0
rcq = 0.0
rcsoil = 0.0
ff  = 0.55*2.0* sx*697.7 * 60/ (rgl * lai)
rcs = (ff + rsmin / rsmax) / (1.0+ ff)
rcs = max (rcs,0.0001)
rct = 1.0- 0.0016* ( (298 - ta)**2.0)
rct = max (rct,0.0001)
ea = 6.11*exp((2.5*10.**6./461.51)*(ta-273.15)/(273.15*ta) )
ws = 0.622*ea/1013
rcq = 1.0/ (1.0+ hs * (ws - qa))
rcq = max (rcq,0.01)
do k = 1, nroot
gx(k) = (smc(k) - smcwlt) / (smcref - smcwlt)
if (gx(k) >  1.) gx(k) = 1.
if (gx(k) <  0.) gx(k) = 0.
part (k) = ( -dzvr (k)/ zsoil (3)) * gx(k)
end do
sgx =0.0
do k = 1, nroot
sgx    = sgx    + gx (k)
rcsoil = rcsoil + part (k)
end do
sgx =sgx / nroot
rcsoil = max (rcsoil,0.0001)
rc = rsmin / (lai * rcs * rct * rcq * rcsoil)
desdt = 0.622*slv*ea/461.51/ta/ta/1013
delta = (slv / cpp)* desdt
rr = (4.* epsv *sigma * 287.04 / cpp)* (ta **4.)/ (ts * ch) + 1.0
pc = (rr + delta)/ (rr * (1. + rc * ch) + delta)
if (cmc .ne. 0.0) then
ett1 = shdfac * pc * etp1 * (1.0- (cmc / cmcmax) ** cfactr) * 0.001
else
ett1 = shdfac * pc * etp1 * 0.001
endif
denom = 0.
do k = 1, nroot
rtx= (-dzvr (k)/ zsoil (3)) + gx(k) - sgx
gx (k) = gx (k) * max ( rtx, 0. )
denom  = denom + gx (k)
end do
if (denom .le. 0.0) denom =1.
do k = 1, nroot
et(k) = ett1 * gx (k) / denom
ett   = ett + et (k)
end do
if (cmc > 0.0) then
ec = shdfac * ( ( cmc / cmcmax ) ** cfactr ) * etp1 * 0.001
else
ec = 0.0
end if
cmc2ms = cmc / delt
ec   = min ( cmc2ms, ec )
end subroutine transp

subroutine smflx (smcp,smc,nsoil,cmcp,cmc,dt,prcp1,zsoil, smcmax,bexp,smcwlt,dksat,dwsat, shdfac,cmcmax,runoff1,runoff2,runoff3, edir,ec,et,drip)
implicit none
integer, intent(in)   :: nsoil
integer               :: i,k
real, intent(in)      :: bexp, cmcmax, dksat,dwsat, dt, ec, edir, prcp1, shdfac, smcmax, smcwlt
real, intent(out)     :: drip, runoff1, runoff2, runoff3
real, intent(in)      :: cmcp
real, intent(out)     :: cmc
real, dimension(1:nsoil), intent(in)   :: zsoil, et
real, dimension(1:nsoil), intent(in)   :: smcp
real, dimension(1:nsoil), intent(out)  :: smc
real, dimension(1:nsoil)               :: ai, bi, ci, stcf,rhsts, rhstt
real                  :: excess,pcpdrp,rhsct,trhsct
rhsct = shdfac * prcp1 * 0.001 /3600. - ec
drip = 0.
trhsct = dt * rhsct
excess = cmcp + trhsct
if (excess > cmcmax) drip = excess - cmcmax
pcpdrp = (1. - shdfac) * prcp1 * 0.001 /3600. + drip / dt
call srt (rhstt,edir,et,smcp,nsoil,pcpdrp,zsoil,dwsat,dksat, smcmax,bexp,runoff1,runoff2,dt,smcwlt,ai,bi,ci)
call sstep (smcp,smc,cmcp,cmc,rhstt,rhsct,dt,nsoil,smcmax, cmcmax,runoff3,zsoil,ai,bi,ci)
end subroutine smflx

subroutine srt (rhstt,edir,et,smcp,nsoil,pcpdrp,zsoil,dwsat, dksat,smcmax,bexp,runoff1, runoff2,dt,smcwlt,ai,bi,ci)
implicit none
integer, intent(in)       :: nsoil
integer                   :: k, ks
real, intent(in)          :: bexp, dksat, dt, dwsat, edir, pcpdrp, smcmax, smcwlt
real, intent(out)         :: runoff1, runoff2
real, dimension(1:nsoil), intent(in)   :: smcp, zsoil, et
real, dimension(1:nsoil), intent(out)  :: rhstt
real, dimension(1:nsoil), intent(out)  :: ai, bi, ci
real, dimension(1:nsoil)  :: ddmax
real                      :: dd, ddt, ddz, ddz2, denom, denom2, dsmdz, dsmdz2, dt1, infmax,mxsmc,mxsmc2,numer,pddum, px,smcav, sstt, par, val, wcnd, wcnd2, wdf, wdf2,kdt
pddum = pcpdrp
runoff1 = 0.0
par = 2.0e-6
if (pcpdrp /=  0.0) then
smcav = smcmax - smcwlt
ddmax (1) = - zsoil (1)* smcav
ddmax (1) = ddmax (1)* (1.0- (smcp (1) - smcwlt)/ smcav)
ddmax (2) = (zsoil (1) - zsoil (2))* smcav
ddmax (2) = ddmax (2)* (1.0- (smcp (2) - smcwlt)/ smcav)
ddmax (3) = (zsoil (2) - zsoil (3))* smcav
ddmax (3) = ddmax (3)* (1.0- (smcp (3) - smcwlt)/ smcav)
dd = ddmax(1)+ddmax(2)+ddmax(3)
dt1 = dt/86400
kdt = 3.0 * dksat / par
val = (1. - exp ( - kdt * dt1))
ddt = dd * val
px = pcpdrp * dt
if (px <  0.0) px = 0.0
infmax = (px * (ddt / (px + ddt)))/ dt
mxsmc = smcp (1)
call wdfcnd (wdf,wcnd,mxsmc,smcmax,bexp,dksat,dwsat)
infmax = max (infmax,wcnd)
infmax = min (infmax,px/dt)
if (pcpdrp >  infmax) then
runoff1  = pcpdrp - infmax
pddum = infmax
end if
end if
call wdfcnd (wdf,wcnd,smcp(1),smcmax,bexp,dksat,dwsat)
ddz = 1. / ( - .5 * zsoil (2) )
ai (1) = 0.0
bi (1) = wdf * ddz / ( - zsoil (1) )
ci (1) = - bi (1)
dsmdz = (smcp (1) - smcp (2) )/( - 0.5 * zsoil(2))
rhstt (1) = (wdf * dsmdz + wcnd- pddum + edir + et(1))/ zsoil (1)
sstt = wdf * dsmdz + wcnd+ edir + et(1)
ddz2 = 0.0
do k = 2,nsoil-1
denom2 = (zsoil (k -1) - zsoil (k))
if (k /= nsoil-1) then
mxsmc2 = smcp (k)
call wdfcnd (wdf2,wcnd2,mxsmc2,smcmax,bexp,dksat,dwsat)
denom = (zsoil (k -1) - zsoil (k +1))
dsmdz2 = (smcp (k) - smcp (k +1)) / (denom * 0.5)
ddz2 = 2.0 / denom
ci (k) = - wdf2 * ddz2 / denom2
else
call wdfcnd (wdf2,wcnd2,smcp(nsoil-1),smcmax,bexp,dksat,dwsat)
dsmdz2 = 0.0
ci (k) = 0.0
end if
numer = (wdf2 * dsmdz2) - (wdf * dsmdz) - wcnd+ et(k)
rhstt (k) = numer / ( - denom2)
ai (k) = - wdf * ddz / denom2
bi (k) = - ( ai (k) + ci (k) )
if (k .eq. nsoil-1) then
runoff2 = 0.0
end if
if (k .ne. nsoil-1) then
wdf = wdf2
wcnd = wcnd2
dsmdz = dsmdz2
ddz = ddz2
end if
end do
end subroutine srt

subroutine sstep (smcp,smc,cmcp,cmc,rhstt,rhsct,dt, nsoil,smcmax,cmcmax,runoff3,zsoil, ai,bi,ci)
implicit none
integer, intent(in)       :: nsoil
integer                   :: i, k, kk11
real, intent(in)          :: cmcmax, dt, smcmax
real, intent(out)         :: runoff3
real, intent(in)          :: cmcp
real, intent(out)         :: cmc
real, dimension(1:nsoil), intent(in)     :: smcp, zsoil
real, dimension(1:nsoil), intent(out)    :: smc
real, dimension(1:nsoil), intent(inout)  :: rhstt
real, dimension(1:nsoil), intent(inout)  :: ai, bi, ci
real, dimension(1:nsoil)  :: rhsttin, smcout,smcin
real, dimension(1:nsoil)  :: ciin
real                      :: ddz, rhsct, wplus, stot
do k = 1,nsoil-1
rhstt (k) = rhstt (k) * dt
ai (k) = ai (k) * dt
bi (k) = 1. + bi (k) * dt
ci (k) = ci (k) * dt
end do
do k = 1,nsoil-1
rhsttin (k) = rhstt (k)
end do
do k = 1,nsoil-1
ciin (k) = ci (k)
end do
call rosr12 (ci,ai,bi,ciin,rhsttin,rhstt,nsoil-1)
wplus = 0.0
runoff3 = 0.
ddz = - zsoil (1)
do k = 1,nsoil-1
if (k /= 1) ddz = zsoil (k - 1) - zsoil (k)
smcout (k) = smcp (k) + ci (k) + wplus / ddz
stot = smcout (k)
if (stot > smcmax) then
if (k .eq. 1) then
ddz = - zsoil (1)
else
kk11 = k - 1
ddz = - zsoil (k) + zsoil (kk11)
end if
wplus = (stot - smcmax) * ddz
else
wplus = 0.
end if
smc (k) = max ( min (stot,smcmax),0.066 )
end do
runoff3 = wplus
cmc = cmcp + dt * rhsct
if (cmc < 1.e-20) cmc = 0.0
cmc = min (cmc,cmcmax)
end subroutine sstep

subroutine wdfcnd (wdf,wcnd,smc,smcmax,bexp,dksat,dwsat)
implicit none
real     bexp
real     dksat
real     dwsat
real     expon
real     factr1
real     factr2
real     smc
real     smcmax
real     wcnd
real     wdf
factr1 = 0.05 / smcmax
factr2 = smc / smcmax
factr1 = min(factr1,factr2)
expon  = bexp + 2.0
wdf    = dwsat * factr2 ** expon
expon  = (2.0 * bexp) + 3.0
wcnd   = dksat * factr2 ** expon
end subroutine wdfcnd

subroutine rosr12 (p,a,b,c,d,delta,nsoil)
implicit none
integer, intent(in)   :: nsoil
integer               :: k, kk
real, dimension(1:nsoil), intent(in):: a, b, d
real, dimension(1:nsoil),intent(inout):: c,p,delta
c (nsoil) = 0.0
p (1) = - c (1) / b (1)
delta (1) = d (1) / b (1)
do k = 2,nsoil
p (k) = - c (k) * ( 1.0 / (b (k) + a (k) * p (k -1)) )
delta (k) = (d (k) - a (k)* delta (k -1))* (1.0/ (b (k) + a (k) * p (k -1)))
end do
p (nsoil) = delta (nsoil)
do k = 2,nsoil
kk = nsoil - k + 1
p (kk) = p (kk) * p (kk +1) + delta (kk)
end do
end subroutine rosr12

subroutine shflx (ssoil,stc,smc,smcmax,nsoil,t1,dt,yy,zz1,zsoil, tbot,zbot,smcwlt,df1,quartz,csoil,capr)
implicit none
integer, intent(in)   :: nsoil
integer               :: i
real, intent(in)      :: df1,dt,smcmax, smcwlt, tbot,yy, zbot,zz1, quartz
real, intent(in)      :: csoil, capr
real, intent(inout)   :: t1
real, intent(out)     :: ssoil
real, dimension(1:nsoil), intent(in)    :: smc,zsoil
real, dimension(1:nsoil), intent(inout) :: stc
real, dimension(1:nsoil)             :: ai, bi, ci, stcf,rhsts
call hrt (rhsts,stc,smc,smcmax,nsoil,zsoil,yy,zz1,tbot, zbot,dt,df1,ai,bi,ci,quartz,csoil,capr)
call hstep (stcf,stc,rhsts,dt,nsoil,ai,bi,ci)
do i = 1,nsoil
stc (i) = stcf (i)
enddo
t1 = (yy + (zz1- 1.0) * stc (1)) / zz1
ssoil = df1 * (stc (1) - t1) / (0.5 * zsoil (1))
end subroutine shflx

subroutine hrt (rhsts,stc,smc,smcmax,nsoil,zsoil,yy,zz1, tbot,zbot,dt,df1,ai,bi,ci,quartz,csoil,capr)
implicit none
logical              :: itavg
integer, intent(in)  :: nsoil
integer              :: i, k
real, intent(in)     :: df1, dt,smcmax ,tbot,yy,zz1, zbot, quartz, csoil, capr
real, dimension(1:nsoil), intent(in)   :: smc,stc,zsoil
real, dimension(1:nsoil), intent(out)  :: rhsts
real, dimension(1:nsoil), intent(out)  :: ai, bi,ci
real                 :: ddz, ddz2, denom, df1k, dtsdz,df1n, dtsdz2,hcpct,qtot,ssoil,sice,tavg,tbk, tbk1,tsnsr,tsurf
real, parameter      :: cair = 1004.0, ch2o = 4.2e6
itavg = .true.
hcpct = smc (1)* ch2o + (1.0- smcmax)* csoil + (smcmax - smc (1)) * cair
ddz = 1.0 / ( -0.5 * zsoil (2) )
ai (1) = 0.0
ci (1) = (df1 * ddz) / (zsoil (1) * hcpct)
bi (1) = - ci (1) + df1 / (0.5 * zsoil (1) * zsoil (1)* hcpct * zz1)
dtsdz = (stc (1) - stc (2)) / (-0.5 * zsoil (2))
ssoil = df1 * (stc (1) - yy) / (0.5 * zsoil (1) * zz1)
denom = (zsoil (1) * hcpct)
rhsts (1) = (df1 * dtsdz - ssoil) / denom
qtot = -1.0* rhsts (1)* denom
if (itavg) then
tsurf = (yy + (zz1-1) * stc (1)) / zz1
call tbnd (stc (1),stc (2),zsoil,zbot,1,nsoil,tbk)
endif
ddz2 = 0.0
df1n = df1
do k = 2,nsoil
if (k < nsoil-1 ) then
hcpct = smc (k)* ch2o + (1.0- smcmax)* csoil + (smcmax - smc ( k))* cair
call tdfcnd  (df1k, smc(k), quartz, smcmax)
denom = 0.5 * ( zsoil (k -1) - zsoil (k +1) )
dtsdz2 = (stc (k) - stc (k +1) ) / denom
ddz2 = 2. / (zsoil (k -1) - zsoil (k +1))
ci (k) = - df1k * ddz2 / ( (zsoil (k -1) - zsoil (k)) * hcpct)
if (itavg) then
call tbnd (stc (k),stc (k +1),zsoil,zbot,k,nsoil,tbk1)
end if
elseif (k == nsoil-1) then
hcpct = smc (k)* ch2o + (1.0- smcmax)* csoil + (smcmax- smc ( k))* cair
call tdfcnd  (df1k, smc(k), quartz, smcmax)
denom = 0.5 * ( zsoil (k -1) - zsoil (k +1) )
dtsdz2 = (stc (k) - stc (k +1) ) / denom
ddz2 = 2. / (zsoil (k -1) - zsoil (k +1))
ci (k) = - df1k * ddz2 / ( (zsoil (k -1) - zsoil (k)) * hcpct)
if (itavg) then
call tbnd (stc (k),tbot,zsoil,zbot,k,nsoil,tbk1)
end if
else
hcpct = capr * 4.1868 * 1.e6
df1k  = 3.24
denom = .5 * (zsoil (k -1) + zsoil (k)) - zbot
dtsdz2 = (stc (k) - tbot) / denom
ci (k) = 0.
if (itavg) then
call tbnd (stc (k),tbot,zsoil,zbot,k,nsoil,tbk1)
end if
end if
denom = ( zsoil (k) - zsoil (k -1) ) * hcpct
rhsts (k) = ( df1k * dtsdz2- df1n * dtsdz ) / denom
qtot = -1.0* denom * rhsts (k)
ai (k) = - df1n * ddz / ( (zsoil (k -1) - zsoil (k)) * hcpct)
bi (k) = - (ai (k) + ci (k))
tbk = tbk1
df1n = df1k
dtsdz = dtsdz2
ddz = ddz2
end do
end subroutine hrt

subroutine hstep (stcout,stcin,rhsts,dt,nsoil,ai,bi,ci)
implicit none
integer, intent(in)  :: nsoil
integer              :: k
real, dimension(1:nsoil), intent(in):: stcin
real, dimension(1:nsoil), intent(out):: stcout
real, dimension(1:nsoil), intent(inout):: rhsts
real, dimension(1:nsoil), intent(inout):: ai,bi,ci
real, dimension(1:nsoil) :: rhstsin
real, dimension(1:nsoil) :: ciin
real                 :: dt
do k = 1,nsoil
rhsts (k) = rhsts (k) * dt
ai (k) = ai (k) * dt
bi (k) = 1. + bi (k) * dt
ci (k) = ci (k) * dt
end do
do k = 1,nsoil
rhstsin (k) = rhsts (k)
end do
do k = 1,nsoil
ciin (k) = ci (k)
end do
call rosr12 (ci,ai,bi,ciin,rhstsin,rhsts,nsoil)
do k = 1,nsoil
stcout (k) = stcin (k) + ci (k)
end do
end subroutine hstep

subroutine tbnd (tu,tb,zsoil,zbot,k,nsoil,tbnd1)
implicit none
integer, intent(in)       :: nsoil
integer                   :: k
real, intent(in)          :: tb, tu, zbot
real, intent(out)         :: tbnd1
real, dimension(1:nsoil), intent(in)   :: zsoil
real                      :: zb, zup
if (k == 1) then
zup = 0.
else
zup = zsoil (k -1)
end if
if (k ==  nsoil) then
zb = 2.* zbot - zsoil (k)
else
zb = zsoil (k +1)
end if
tbnd1 = tu + (tb - tu)* (zup - zsoil (k))/ (zup - zb)
end subroutine tbnd

subroutine tdfcnd (df, smc, qz, smcmax)
implicit none
real, intent(in)          :: qz,  smc, smcmax
real, intent(out)         :: df
real                      :: ake, gammd, thkdry, thko, thkqtz,thksat,thks,thkw,satratio
satratio = smc / smcmax
thkw = 0.57
thko = 2.0
thkqtz = 7.7
thks = (thkqtz ** qz)* (thko ** (1. - qz))
thksat = thks ** (1. - smcmax)* thkw ** (smcmax)
gammd = (1. - smcmax)*2700.
thkdry = (0.135* gammd+ 64.7)/ (2700. - 0.947* gammd)
if ( satratio >  0.1 ) then
ake = log10 (satratio) + 1.0
else
ake = 0.0
end if
df = ake * (thksat - thkdry) + thkdry
end subroutine tdfcnd
'''



def _floors(chs, chs2, cqs2):
    floor = F(1.0e-2)
    return tuple(floor if F(x) < floor else F(x) for x in (chs, chs2, cqs2))


def _renewal(urban):
    """Driver assignments following urban, including optional supplied state.

    Pass the urban outputs, optionally merged with its renewed state. Values
    are copied to the driver's *_urb2d and *_urb3d names without blending.
    """
    result = {}
    renewed = 'ts psim psih gz1oz0 u10 v10 th2 q2 ust'.split()
    renewed += [k for k in STATE_FIELDS if k not in _ALIASES]
    for k in renewed:
        if k in urban:
            result[k + ('_urb3d' if k in LAYER_FIELDS else '_urb2d')] = (
                np.array(urban[k], dtype=np.float32, copy=True)
                if k in LAYER_FIELDS else F(urban[k]))
    result['akms_urb2d'] = F(F(F(0.4)*F(urban['ust'])) /
                                F(F(urban['gz1oz0'])-F(urban['psim'])))
    for k in 'cmr chr cmgr chgr cmc chc'.split():
        if k in urban:
            result[k + '_sfcdif'] = F(urban[k])
    result['znt'] = F(urban['znt'])
    return result


def _driver_forcing(*, ta, qa, u1, v1, soldn, glw, rainbl, dt, rhoo,
                    za, declin, cosz, omg, xlat, znt, chs, chs2):
    values = {k:F(v) for k,v in locals().items()}
    u1, v1 = values['u1'], values['v1']
    ua = F(sqrtf(F(F(powf(u1,F(2)))+F(powf(v1,F(2))))))
    if ua < F(1): ua = F(1)
    values.update(ua=ua, rain=F(F(values['rainbl']/values['dt'])*F(3600)),
                  delt=values['dt'], ssg=values['soldn'], llg=values['glw'])
    result = {k+'_urb':values[k] for k in FORCING_FIELDS}
    result['ssgd_urb'] = F(F(0.8)*values['soldn'])
    result['ssgq_urb'] = F(values['soldn']-result['ssgd_urb'])
    return result


def noah_blend(urban: Mapping[str, object], *, frc_urb, t1, sheat,
               eta_kinematic, eta, ssoil, albedok, q1, sfctmp, q2k,
               sfcprs, zlvl, soldn, rainbl, emissi, ust, u1, v1,
               glw, dt, declin, cosz, omg, xlat, znt, chs, chs2, cqs2):
    """Pure scalar Noah driver reference (WRF 1317-1600).

    ``urban`` contains urban_step's outputs and optionally its renewed state.
    Returns the blended surface fields, urban call inputs as *_urb, and
    state/diagnostic renewal as *_urb2d (or *_urb3d for optional arrays).
    ``emissi`` is Noah's rural emissivity; this arm does not blend it.
    """
    f, rural = F(frc_urb), F(F(1)-F(frc_urb))
    chs, chs2, cqs2 = _floors(chs, chs2, cqs2)
    rhoo = F(F(sfcprs)/F(F(F(287.04)*F(sfctmp))*F(F(1)+F(F(0.61)*F(q2k)))))
    result = _driver_forcing(ta=sfctmp,qa=q2k,u1=u1,v1=v1,soldn=soldn,
        glw=glw,rainbl=rainbl,dt=dt,rhoo=rhoo,za=zlvl,declin=declin,
        cosz=cosz,omg=omg,xlat=xlat,znt=znt,chs=chs,chs2=chs2)
    for name,uk,rv in [('albedo','alb',albedok),('hfx','sh',sheat),
                       ('qfx','lh_kinematic',eta_kinematic),('lh','lh',eta),
                       ('grdflx','g',ssoil),('tsk','ts',t1),('q1','qs',q1),
                       ('ust','ust',ust)]:
        result[name] = F(F(f*F(urban[uk]))+F(rural*F(rv)))
    result['qsfc'] = F(result['q1']/F(F(1)-result['q1']))
    result.update(chs=chs,chs2=chs2,cqs2=cqs2,emissi=F(emissi))
    result.update(_renewal(urban))
    return result


def noahmp_blend(urban: Mapping[str, object], *, frc_urb, albedo, hfx,
                 qfx, lh, grdflx, tsk, qsfc, ust, t3d, qv, u1, v1,
                 swdown, glw, rainbl, dt, p8w_lower, p8w_upper, dz8w,
                 declin, cosz, omg, xlat, znt, chs, chs2, cqs2):
    """Pure scalar option-1 Noah-MP driver reference (WRF 3374-3598)."""
    f, rural = F(frc_urb), F(F(1)-F(frc_urb))
    chs, chs2, cqs2 = _floors(chs, chs2, cqs2)
    chs2 = cqs2
    qa = F(F(qv)/F(F(1)+F(qv)))
    rhoo = F(F(F(F(p8w_upper)+F(p8w_lower))*F(0.5)) /
             F(F(F(287.04)*F(t3d))*F(F(1)+F(F(0.61)*qa))))
    result = _driver_forcing(ta=t3d,qa=qa,u1=u1,v1=v1,soldn=swdown,
        glw=glw,rainbl=rainbl,dt=dt,rhoo=rhoo,za=F(F(0.5)*F(dz8w)),
        declin=declin,cosz=cosz,omg=omg,xlat=xlat,znt=znt,chs=chs,chs2=chs2)
    for name,uk,rv in [('albedo','alb',albedo),('hfx','sh',hfx),
                       ('qfx','lh_kinematic',qfx),('lh','lh',lh),
                       ('tsk','ts',tsk),('qsfc','qs',qsfc),('ust','ust',ust)]:
        result[name] = F(F(f*F(urban[uk]))+F(rural*F(rv)))
    result['grdflx'] = F(F(f*F(F(urban['g'])*F(-1)))+F(rural*F(grdflx)))
    result.update(chs=chs,chs2=chs2,cqs2=cqs2)
    result.update(_renewal(urban))
    return result


def noah_overrides(urban: Mapping[str, np.float32], *, chs, akms_urb2d):
    """Surface driver Noah overrides, WRF 3001-3021."""
    result = {k:F(urban[k]) for k in 'u10 v10 psim psih gz1oz0'.split()}
    result.update(akhs=F(chs),akms=F(akms_urb2d))
    return result


RCP = F(F(287)/F(F(F(7)*F(287))/F(2)))


def noahmp_overrides(urban: Mapping[str, np.float32], *, frc_urb, fvegxy,
                     t2mvxy, t2mbxy, q2mvxy, q2mbxy, psfc, chs, akms_urb2d):
    """Surface driver Noah-MP option-1 overrides, WRF 3383-3404."""
    f, v = F(frc_urb), F(fvegxy)
    r, bare = F(F(1)-f), F(F(1)-v)
    exner = F(powf(F(F(1e5)/F(psfc)),RCP))
    result = noah_overrides(urban,chs=chs,akms_urb2d=akms_urb2d)
    result['q2'] = F(F(F(F(v*F(q2mvxy))+F(bare*F(q2mbxy)))*r)+F(F(urban['q2'])*f))
    result['t2'] = F(F(F(F(v*F(t2mvxy))+F(bare*F(t2mbxy)))*r)+F(F(F(urban['th2'])/exner)*f))
    result['th2'] = F(result['t2']*exner)
    return result
