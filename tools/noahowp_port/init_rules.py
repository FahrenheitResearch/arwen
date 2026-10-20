#!/usr/bin/env python3
"""Offline generation of native initialization metadata, never a runtime path."""
import json
import re
import subprocess
import sys
from pathlib import Path
source, output = map(Path, sys.argv[1:])
if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != '0242a96c2a90b2c1501ff56a1e457a932a49c8ac':
    raise SystemExit('initialization metadata requires the pinned upstream revision')
rules = {'source_revision': '0242a96c2a90b2c1501ff56a1e457a932a49c8ac',
         'defaults': {}, 'transfer': {}, 'parameter_tables': {}, 'constants': {}}
for group in ('levels', 'domain', 'options', 'parameters', 'forcing', 'energy', 'water'):
    text = (source/'src'/f'{group.title()}Type.f90').read_text().lower()
    for line in text.split('subroutine paramread(', 1)[0].splitlines():
        line = line.split('!')[0]
        # Includes the initialization after ALLOCATE(...); ignores declarations.
        for name, value in re.findall(r'this%(\w+)\s*(?:\([^)]*\))?\s*=\s*([^;]+)', line):
            if value.strip().startswith(('huge(', '.true.', '.false.')):
                rules['defaults'][f'{group}.{name}'] = value.strip()
            elif re.fullmatch(r'[+\-]?[\d.]+(?:e[+\-]?\d+)?', value.strip()):
                rules['defaults'][f'{group}.{name}'] = value.strip()
    transfer = text.split('subroutine inittransfer(', 1)
    if len(transfer) == 2:
        for name, value in re.findall(r'this%(\w+)\s*=\s*namelist%(\w+)', transfer[1]):
            rules['transfer'][f'{group}.{name}'] = value
    if group == 'parameters':
        body = text.split('subroutine paramread(', 1)[1]
        for line in body.splitlines():
            line = line.split('!')[0].strip()
            m = re.fullmatch(r'this%(\w+)(?:\(([^)]*)\))?\s*=\s*(.*)', line)
            if not m:
                continue
            name, target_index, value = m.groups()
            t = re.fullmatch(r'(\w+)_table(?:\((.*)\))?', value)
            key = f'parameters.{name}'
            if t:
                rules['parameter_tables'].setdefault(key, []).append(
                    {'target_index': target_index, 'table': t[1], 'index': t[2]})
            elif re.fullmatch(r'[+\-]?[\d.]+(?:e[+\-]?\d+)?|\.false\.|\.true\.', value):
                rules['constants'][key] = value
# The straight constant assignments in initialize_from_file override type defaults.
text = (source/'src'/'RunModule.f90').read_text().lower().split('end subroutine initialize_from_file')[0]
for line in text.splitlines():
    line = line.split('!')[0].strip()
    m = re.fullmatch(r'(water|energy|forcing|domain)%(\w+)\s*=\s*([+\-]?[\d.]+(?:[ed][+\-]?\d+)?|\.false\.|\.true\.)', line)
    if m:
        rules['constants'][f'{m[1]}.{m[2]}'] = m[3]
# Rain threshold is conditional and is evaluated by Rust, not this metadata.
rules['constants'].pop('parameters.rain_snow_thresh', None)
output.write_text(json.dumps(rules, indent=2, sort_keys=True)+'\n')
