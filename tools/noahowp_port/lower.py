#!/usr/bin/env python3
"""Build-time lowering of the pinned Apache Noah-OWP numeric modules to CUDA.

No native executable or libf2c is linked or shipped. f2c is an offline compiler
pass after the explicitly supported F90 constructs below are lowered. Any new
syntax is rejected by this pass or f2c -u. Generated metadata maps every field
back to its original type and Fortran lower bound.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

PIN = '0242a96c2a90b2c1501ff56a1e457a932a49c8ac'
GROUPS = ['Levels', 'Domain', 'Options', 'Parameters', 'Forcing', 'Energy', 'Water']
MODULES = ['UtilitiesModule', 'AtmProcessing', 'ForcingModule', 'InterceptionModule',
           'CanopyWaterModule', 'EnergyModule', 'ThermalPropertiesModule',
           'PrecipHeatModule', 'ShortwaveRadiationModule', 'AlbedoModule',
           'EtFluxModule', 'SnowSoilTempModule', 'SnowWaterModule', 'SnowWaterRenew',
           'SnowLayerChange', 'SoilWaterModule', 'SoilWaterMovement',
           'SoilWaterRetentionCoeff', 'SurfaceRunoffModule',
           'SurfaceRunoffInfiltration', 'SubsurfaceRunoffModule', 'WaterModule']


def split(s, delim=','):
    out, start, depth, quote = [], 0, 0, None
    for i, c in enumerate(s):
        if quote:
            if c == quote: quote = None
        elif c in "'\"": quote = c
        elif c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == delim and depth == 0:
            out.append(s[start:i].strip()); start = i + 1
    out.append(s[start:].strip())
    return out


def statements(path):
    # cpp selects the native BMI forcing branch; no source include dependencies.
    src = subprocess.check_output(['cpp', '-P', '-traditional-cpp',
          '-DNGEN_FORCING_ACTIVE', str(path)], text=True)
    buf = ''
    for line in src.splitlines():
        quote, clean = None, ''
        for c in line:
            if quote:
                clean += c
                if c == quote: quote = None
            elif c in "'\"": quote = c; clean += c
            elif c == '!': break
            else: clean += c
        line = clean.strip()
        if not line: continue
        more = line.endswith('&')
        buf += ' ' + line.removesuffix('&').lstrip('&')
        if not more:
            for s in split(buf.strip(), ';'):
                if s: yield s.lower()
            buf = ''
    if buf: raise ValueError(f'unterminated continuation: {path}')


def decl(s):
    if '::' in s:
        attrs, body = s.split('::', 1)
    else:
        old = re.match(r'^(double precision|real\*8|real|integer|logical|character)\s+(.+)$', s)
        if not old: return None
        attrs, body = old[1], old[2]
    typ = re.match(r'\s*(double precision|real\*8|real|integer|logical|character)', attrs)
    if not typ: return None
    typ = typ[1]
    dim = re.search(r'dimension\s*\((.*?)\)', attrs)
    entries = []
    for ent in split(body):
        name = re.match(r'(\w+)\s*(\([^)]*\))?\s*(?:=\s*(.*))?$', ent)
        if not name: raise ValueError(('declaration', s, ent))
        dims = name[2][1:-1] if name[2] else dim[1] if dim else None
        entries.append((name[1], dims, name[3]))
    return typ, attrs, entries


def bound(s, nsoil, nsnow):
    s = s.replace('namelist%nsoil', str(nsoil)).replace('namelist%nsnow', str(nsnow))
    s = s.replace('levels%nsoil', str(nsoil)).replace('levels%nsnow', str(nsnow))
    if not re.fullmatch(r'[\d\s+*/()-]+', s): raise ValueError(('nonconstant field bound', s))
    return int(eval(s, {'__builtins__': {}}, {}))


def fields(src, nsoil, nsnow):
    result = []
    for group in GROUPS:
        allstat = list(statements(src / (group + 'Type.f90')))
        inside = False
        allocations = {}
        for s in allstat:
            m = re.match(r'allocate\(this%(\w+)\s*\((.*?)\)\)', s)
            if m: allocations[m[1]] = m[2]
        for s in allstat:
            if re.match(r'type,\s*public\s*::', s): inside = True; continue
            if inside and s == 'contains': break
            if not inside: continue
            d = decl(s)
            if not d: raise ValueError(('type declaration', s))
            typ, attrs, entries = d
            if typ == 'character': continue  # parsed calendar components are numeric carriers
            for name, dims, init in entries:
                if name == 'sim_datetimes': continue  # host initialization schedule, not physics
                if dims == ':' or 'allocatable' in attrs:
                    dims = allocations[name]
                bounds = []
                if dims:
                    for dim in split(dims):
                        parts = dim.split(':')
                        lo, hi = parts if len(parts) == 2 else ('1', parts[0])
                        bounds.append([bound(lo, nsoil, nsnow), bound(hi, nsoil, nsnow)])
                result.append(dict(name=group.lower()+'.'+name, symbol=f'f{len(result):04}',
                    type=typ, bounds=bounds))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--f2c', type=Path, required=True)
    ap.add_argument('--nsoil', type=int, default=4)
    ap.add_argument('--nsnow', type=int, default=3)
    a = ap.parse_args()
    if subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('source revision differs from licence-reviewed pin')
    src = a.source/'src'; a.work.mkdir(parents=True, exist_ok=True); a.output.mkdir(parents=True, exist_ok=True)
    ff = fields(src, a.nsoil, a.nsnow)
    ff.sort(key=lambda f: 0 if f['type'] in ('real*8','double precision') else 1)
    fmap = {x['name'].replace('.', '%'): x['symbol'] for x in ff}
    fmeta = {x['symbol']: x for x in ff}
    units = []
    constants = {}
    for module in MODULES:
        current = None
        for s in statements(src/(module+'.f90')):
            m = re.match(r'(?:(integer|real|double precision)\s+)?(subroutine|function)\s+(\w+)\s*\((.*?)\)$', s)
            if m:
                current = dict(module=module, name=m[3], kind=m[2], ret=m[1], args=split(m[4]), body=[])
                units.append(current); continue
            if re.match(r'end\s+(subroutine|function)', s): current = None; continue
            if current is not None: current['body'].append(s)
            elif (d := decl(s)) and 'parameter' in d[1]:
                for name, dims, value in d[2]: constants[name] = value
    # String parsing belongs to setup and has no caller in the physics chain.
    units = [u for u in units if u['name'] != 'minutes_between']
    for i, u in enumerate(units): u['symbol'] = f'r{i:04}'
    lookup = {}
    for u in units: lookup.setdefault(u['name'], []).append(u)
    def routine(name, module):
        choices = lookup[name]
        own = [u for u in choices if u['module'] == module]
        if len(own) == 1: return own[0]
        if len(choices) == 1: return choices[0]
        raise ValueError(('ambiguous call', name, module))
    def replace(s, module):
        s = re.sub(r'\b\w+%\w+\b', lambda m: fmap.get(m[0], m[0]), s)
        s = re.sub(r'\b\w+\b', lambda m: '('+constants[m[0]]+')' if m[0] in constants else routine(m[0], module)['symbol'] if m[0] in lookup else m[0], s)
        s = re.sub(r'real\((\w+),\s*kind\s*=\s*8\)', r'dble(\1)', s)
        s = s.replace('modulo(total, 1440)', 'mod(mod(total,1440)+1440,1440)')
        for x, y in [('==','.eq.'),('/=','.ne.'),('>=','.ge.'),('<=','.le.'),('>','.gt.'),('<','.lt.')]: s = s.replace(x,y)
        if '%' in s: raise ValueError(('unknown model field', s))
        return s
    def fixed(s):
        label = re.match(r'^(\d+)\s+(.*)$', s)
        prefix = label[1].rjust(5)+' ' if label else '      '
        body = label[2] if label else s
        return '\n'.join((prefix if i == 0 else '     &')+body[i:i+66] for i in range(0,len(body),66))
    common = []
    for f in ff:
        typ = 'double precision' if f['type'] == 'real*8' else f['type']
        dims = ','.join(f'{lo}:{hi}' for lo,hi in f['bounds'])
        common += [typ+' '+f['symbol']+('('+dims+')' if dims else '')]
    common += ['integer status', 'common /model/ '+','.join(f['symbol'] for f in ff)+',status']
    result, diagnostics = [], []
    for u in units:
        structured = set()
        arrays = {x['symbol']: x['bounds'] for x in ff if x['bounds']}
        declarations, executable, data, initializers = [], [], [], []
        aliases = {}
        for s in u['body']:
            if re.match(r'type\s*\(',s):
                structured.update(split(s.split('::')[1])); continue
            if s == 'implicit none' or s.startswith('use '): continue
            if s.startswith('associate('):
                for ent in split(s[len('associate('):-1]):
                    name, value = ent.split('=>'); aliases[name.strip()] = value.strip()
                continue
            if s == 'end associate': continue
            # Substitute aliases only inside their lexical block. Both blocks
            # occupy the rest of their separate routines in this source pin.
            if aliases:
                for name in sorted(aliases, key=len, reverse=True):
                    s = re.sub(r'(?<![%\w])'+name+r'\b', aliases[name], s)
            if (d := decl(s)):
                typ, attrs, entries = d
                if typ == 'character': continue
                typ = 'double precision' if typ == 'real*8' else typ
                for name, dims, value in entries:
                    if u['kind'] == 'function' and name == u['name']:
                        if typ != (u['ret'] or 'real') or dims or value:
                            raise ValueError(('function result declaration', s))
                        continue
                    dims = replace(dims, u['module']) if dims else None
                    if dims:
                        if name not in u['args']:
                            dims = re.sub(r'\bnsnow\b', str(a.nsnow), dims)
                            dims = re.sub(r'\bnsoil\b', str(a.nsoil), dims)
                        # Derived-type profile counts specialize the kernel,
                        # while explicit dummy bounds remain native adjustable.
                        for k,v in [(fmap['levels%nsoil'],a.nsoil),(fmap['levels%nsnow'],a.nsnow)]: dims=re.sub(r'\b'+k+r'\b',str(v),dims)
                        bounds = [p.split(':') if ':' in p else ['1',p] for p in split(dims)]
                        arrays[replace(name,u['module'])] = bounds
                    var = replace(name,u['module'])+('('+dims+')' if dims else '')
                    declarations.append(typ+' '+var)
                    if value:
                        value = replace(value,u['module'])
                        if 'parameter' in attrs:
                            if value.startswith('(/'):
                                data.append('data '+replace(name,u['module'])+' /'+value[2:-2]+'/')
                            elif typ == 'real':
                                # f2c folds constant expressions in host double,
                                # unlike native default REAL arithmetic. Keep
                                # REAL parameters as per-call local assignments.
                                initializers.append(replace(name,u['module'])+'='+value)
                            else: declarations.append('parameter ('+replace(name,u['module'])+'='+value+')')
                        else: data.append('data '+replace(name,u['module'])+' /'+value+'/')
                continue
            if s.startswith('parameter ') or s.startswith('data '): data.append(replace(s,u['module'])); continue
            if s.startswith('write') or s.startswith('print') or s.startswith('call wrf_error'):
                diagnostics.append({'routine':u['name'],'statement':s,'handling':'resident diagnostic bit'})
                executable.append('status = ior(status,1)'); continue
            if s.startswith('stop'):
                executable += ['status = ior(status,2)', 'return']; continue
            # Eliminate derived-type arguments from calls and signatures.
            if (m := re.search(r'call\s+(\w+)\s*\((.*)\)$',s)):
                args = [x for x in split(m[2]) if x not in structured]
                s = s[:m.start()]+'call '+m[1]+'('+','.join(args)+')'
            executable.append(replace(s,u['module']))
        args = [x for x in u['args'] if x not in structured]
        signature = ('subroutine' if u['kind']=='subroutine' else (u['ret'] or 'real')+' function')+' '+u['symbol']+'('+','.join(args)+')'
        # Every user function is explicitly typed; -u rejects unknown symbols.
        funcs = [x for x in units if x['kind']=='function' and x is not u]
        funcsdecl = [(x['ret'] or 'real')+' '+x['symbol'] for x in funcs]
        out = [signature, 'implicit none', *common, 'integer ixarr', 'real red001,red002,red003,red004', *funcsdecl, *declarations, *data]
        executable = initializers + executable
        # Elementary array expressions and reductions (all rank one at pin).
        def scalarize(expr, idx):
            ranges=[]
            for name, dims in sorted(arrays.items(),key=lambda x:len(x[0]),reverse=True):
                if len(dims)!=1: raise ValueError('rank != 1')
                lo,hi=dims[0]
                pattern=r'\b'+name+r'\b(\s*\([^()]*\))?'
                def sub(m):
                    subscript = m[1]
                    if subscript and ':' not in subscript: return m[0]
                    start,end = lo,hi
                    if subscript:
                        start,end=subscript.strip()[1:-1].split(':')
                        start=start.strip() or str(lo);end=end.strip() or str(hi)
                    ranges.append((str(start),str(end)))
                    return name+'(('+str(start)+')+('+idx+')-1)'
                expr=re.sub(pattern,sub,expr)
            return expr,ranges
        selects=[]; loops=[]; label=9000
        for s in executable:
            s = re.sub(r'^\w+:\s*do', 'do', s)
            s = re.sub(r'^end do\s+\w+$', 'enddo', s)
            s = re.sub(r'^exit\s+\w+$', 'exit', s)
            if s.startswith('select case'):
                selects.append([s[s.index('(')+1:-1],False]); continue
            if s.startswith('case'):
                sel,seen=selects[-1]
                if s=='case default': out.append('else')
                else:
                    vals=split(s[s.index('(')+1:-1]); cond=' .or. '.join('('+sel+'.eq.'+x+')' for x in vals)
                    out.append(('else if' if seen else 'if')+' ('+cond+') then')
                selects[-1][1]=True;continue
            if s=='end select': selects.pop();out.append('endif');continue
            if s=='do' or s.startswith('do while'):
                label+=3;loops.append((label,label+1,label+1))
                out.append(str(label)+' continue')
                if s!='do': out.append('if (.not. '+s[s.index('('):]+') goto '+str(label+1))
                continue
            if s.startswith('do '):
                # A counted DO terminal label still executes its increment.
                # EXIT must jump beyond that label, rather than act as CYCLE.
                label+=3;loops.append((None,label+1,label+2));out.append('do '+str(label+1)+' '+s[3:]);continue
            if s in {'end do','enddo'}:
                start,end,after=loops.pop()
                if start: out.append('goto '+str(start))
                out.append(str(end)+' continue')
                if after != end: out.append(str(after)+' continue')
                continue
            if s=='exit': out.append('goto '+str(loops[-1][2]));continue
            if s=='cycle':
                if loops[-1][0] is None: raise ValueError('cycle inside counted do unsupported')
                out.append('goto '+str(loops[-1][0]));continue
            if s.endswith(' exit'):
                out.append(s[:-5]+' goto '+str(loops[-1][2]));continue
            # Lower reductions wherever they occur, including IF conditions.
            redcount=0
            while (red := re.search(r'\b(sum|maxval|minval)\(', s)):
                depth=1; end=red.end()
                while depth:
                    if s[end]=='(': depth+=1
                    elif s[end]==')': depth-=1
                    end+=1
                val,ranges=scalarize(s[red.end():end-1],'ixarr')
                if not ranges: raise ValueError(('reduction without arrays',s))
                redcount+=1
                if redcount>4: raise ValueError('too many reductions')
                temp=f'red{redcount:03}'
                lo,hi=ranges[0];extent='('+str(hi)+')-('+str(lo)+')+1'
                initial='0.0' if red[1]=='sum' else '-3.402823466e38' if red[1]=='maxval' else '3.402823466e38'
                combine=temp+'+'+val if red[1]=='sum' else ('max' if red[1]=='maxval' else 'min')+'('+temp+','+val+')'
                out += [temp+'='+initial,'do ixarr=1,'+extent,temp+'='+combine,'enddo']
                s=s[:red.start()]+temp+s[end:]
            m=re.match(r'^(\w+(?:\([^=]*\))?)\s*=\s*(.*)$',s)
            if m:
                lhs,rhs=m[1].strip(),m[2].strip()
                left,ranges=scalarize(lhs,'ixarr')
                if ranges:
                    right,rr=scalarize(rhs,'ixarr');lo,hi=ranges[0]
                    out += ['do ixarr=1,('+str(hi)+')-('+str(lo)+')+1',left+'='+right,'enddo'];continue
            out.append(s)
        if loops or selects: raise ValueError(('unclosed block',u['name']))
        # Expose each default-REAL literal as a readonly DATA local so the
        # offline f2c pass cannot evaluate compound expressions in double.
        # CUDA subsequently folds float operands with native REAL rounding.
        literals = {}
        literal_pattern = r'(?<![\w.])(?:\d+\.\d*|\.\d+|\d+[eEdD][+-]?\d+)(?:[eEdD][+-]?\d+)?(?![\w.])'
        def preserve_real(m):
            value = m[0]
            if 'd' in value.lower(): return value
            if value not in literals: literals[value] = f'rc{len(literals):04}'
            return literals[value]
        # Locate the declaration/executable boundary without assuming its size.
        prefix_count = len([signature, 'implicit none', *common, 'integer ixarr', 'real red001,red002,red003,red004', *funcsdecl, *declarations, *data])
        body = [re.sub(literal_pattern, preserve_real, s) for s in out[prefix_count:]]
        literal_decls = ['real '+name for name in literals.values()]
        literal_data = ['data '+name+' /'+value+'/' for value,name in literals.items()]
        out = out[:prefix_count-len(data)] + literal_decls + data + literal_data + body + ['end']
        result += [fixed(s) for s in out]
    fortran=a.work/'model.f';fortran.write_text('\n'.join(result)+'\n')
    subprocess.run([str(a.f2c.resolve()),'-a','-R','-r','-u','-w','model.f'],cwd=a.work,check=True)
    schema=dict(source_revision=PIN,nsoil=a.nsoil,nsnow=a.nsnow,fields=ff,
                routines=[{k:u[k] for k in ('module','name','symbol','kind','ret')} for u in units],
                source_sha256={m+'.f90':hashlib.sha256((src/(m+'.f90')).read_bytes()).hexdigest() for m in MODULES},
                diagnostic_handling=diagnostics)
    (a.output/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    (a.output/'lowered.c').write_text((a.work/'model.c').read_text())

if __name__=='__main__': main()
