"""Emit a gfortran probe for constant float subexpressions in the CUDA port."""
from pathlib import Path
import json
import re
import sys

path = Path(sys.argv[1])
source = path.read_text()
constants = dict(CP="1004.5", XLV="2.5e6", RHOWATER="1000.0")


def split_args(s):
    level, start, parts = 0, 0, []
    for i, c in enumerate(s):
        if c == '(':
            level += 1
        elif c == ')':
            level -= 1
        elif c == ',' and level == 0:
            parts.append(s[start:i].strip());start = i+1
    parts.append(s[start:].strip())
    return parts


def expression(s):
    s = s.strip()
    if s in constants:
        return constants[s]
    if re.fullmatch(r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?f', s):
        return s[:-1] if '.' in s or 'e' in s.lower() else s[:-1]+'.0'
    if s.startswith('(') and s.endswith(')'):
        return expression(s[1:-1])
    if s.startswith('-'):
        return '-(' + expression(s[1:]) + ')'
    match = re.fullmatch(r'(FADD|FSUB|FMUL|FDIV)\((.*)\)', s, re.S)
    if match:
        a, b = split_args(match[2])
        return '((' + expression(a) + ')' + {'FADD': '+', 'FSUB': '-', 'FMUL': '*', 'FDIV': '/'}[match[1]] + '(' + expression(b) + '))'
    raise ValueError(s)


for name, value in re.findall(r'^#define\s+(\w+)\s+([\d.eE+\-]+f)\b', source, re.M):
    constants[name] = value[:-1]
for _ in range(4):
    for decl in re.findall(r'const real\s+([^;]+);', source):
        for item in split_args(decl):
            if '=' not in item:
                continue
            name, value = item.split('=', 1)
            try:
                constants[name.strip()] = expression(value)
            except ValueError:
                pass

sites, expressions = [], {}
for m in re.finditer(r'\b(FADD|FSUB|FMUL|FDIV)\(', source):
    depth, pos = 1, m.end()
    while depth:
        if source[pos] == '(':
            depth += 1
        elif source[pos] == ')':
            depth -= 1
        pos += 1
    try:
        expr = expression(source[m.start():pos])
    except ValueError:
        continue
    name = expressions.setdefault(expr, f'fold_{len(expressions):03}')
    sites.append(dict(start=m.start(), end=pos, name=name, expression=expr))

program = '''program folded_words
use module_model_constants,only: R_D,CP,XLV,RHOWATER,XLF,STBOLT
implicit none
real,parameter :: capa=R_D/CP
write(*,'(A,1X,Z8.8)') 'CAPA',transfer(capa,0)
write(*,'(A,1X,Z8.8)') 'CP',transfer(CP,0)
write(*,'(A,1X,Z8.8)') 'ROWLIW',transfer(RHOWATER*XLF,0)
write(*,'(A,1X,Z8.8)') 'XLV_RHOWATER',transfer(XLV*RHOWATER,0)
write(*,'(A,1X,Z8.8)') 'STBOLT',transfer(STBOLT,0)
'''
for expr, name in expressions.items():
    program += f"write(*,'(A,1X,Z8.8)') '{name}',transfer({expr},0)\n"
program += 'end program\n'
path.with_name('folded_words.F90').write_text(program, encoding='utf-8', newline='\n')
path.with_name('folded-sites.json').write_text(json.dumps(sites, indent=2))
print(len(sites), 'constant sites,', len(expressions), 'distinct expressions')
