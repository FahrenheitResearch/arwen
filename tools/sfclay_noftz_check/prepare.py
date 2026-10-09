"""Prepare added classic/revised inputs and a CK-corrected private referee."""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
L = Path(os.environ.get('SFCLAY_CHECK_ROOT', '/work/pool-sfclay-noftz-mynn-mm5-eta'))
WRF = Path(os.environ['WRF_SOURCE_ROOT'])
from tools.sfclay_classic_wrf461_oracle import make_columns as c
d = L / 'oracles/classic'
s = (WRF / 'phys/module_sf_sfclay.F').read_text()
old = 'PSIQ10=GZ10OZ0(I)-PSIH(I)+GZ0OZQ'
assert s.count(old) == 1
source = d / 'module_sf_sfclay_ck.F'
source.write_text(s.replace(old, 'PSIQ10=GZ10OZ0(I)-PSIH10(I)+GZ0OZQ'))
flags = ['gfortran', '-O0', '-cpp', '-DEM_CORE=1', '-ffree-form', '-ffree-line-length-none']
subprocess.run(flags + ['-c', str(source), '-o', 'ck.o'], cwd=d, check=True)
subprocess.run(flags + ['-I.', str(ROOT / 'tools/sfclay_classic_wrf461_oracle/run_sfclay_classic.F90'), 'ck.o', '-o', 'run_corrected'], cwd=d, check=True)
fringe = [c.col(xland=1.5,hfx=75.,qfx=.0001,qsfc=.01),
          c.col(qv=1e-40,qsfc=1e-40,qfx=1e-40),
          c.col(mol=-1e-40,hfx=1e-40,qfx=1e-40,tsk=285.,qsfc=.005)]
lines = ['1 7', f'subnormal 3 1 0 1 {c.word(3000.)}']
lines.extend(' '.join(str(c.word(v[f])) for f in c.INPUT_FIELDS) for v in fringe)
(d / 'fringe-columns.txt').write_text('\n'.join(lines) + '\n')
for inputs, outputs in [('columns.txt', 'wrf-ck-corrected.txt'), ('fringe-columns.txt', 'fringe-corrected.txt')]:
    subprocess.run([str(d / 'run_corrected'), str(d / inputs), str(d / outputs)], check=True)

import _sfclayrev_oracle as r
base = r.load_fixture()
d = L / 'oracles/revised-subnormal'
d.mkdir(exist_ok=True)
rows = []
for bits in [1, 2, 0xfff, 0x7fffff, 0x800000, 0x800001]:
    for water in [1., 2.]:
        v = {k: np.float32(a[0]) for k, a in base.inputs.items()}
        small = np.array([bits], np.uint32).view(np.float32)[0]
        v.update(qv=small, qsfc=small, qfx=small, hfx=small,
                 mol=-small, xland=np.float32(water), tsk=np.float32(285.))
        rows.append(v)
lines = ['# case ' + ' '.join(r.INPUT_FIELDS)]
for i, v in enumerate(rows, 1):
    lines.append(str(i) + ' ' + ' '.join(f'{int(v[f].view(np.uint32)):08X}' for f in r.INPUT_FIELDS))
(d / 'sfclayrev-inputs.hex').write_text('\n'.join(lines) + '\n')
(d / 'sfclayrev-cases.csv').write_text('case,label\n' + '\n'.join(f'{i},subnormal_{i}' for i in range(1, len(rows)+1)) + '\n')
subprocess.run([str(L / 'oracles/revised/run_sfclayrev'), str(d / 'sfclayrev-inputs.hex'), str(d / 'sfclayrev-outputs.hex')], check=True)
