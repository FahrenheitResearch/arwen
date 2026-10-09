"""Grade all requested subnormal cases with raw words and hash actual fields."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
L = Path(os.environ.get('SFCLAY_CHECK_ROOT', '/work/pool-sfclay-noftz-mynn-mm5-eta'))
unit = sys.argv[1]
mode = 'strict' if os.environ.get('GPUWM_WRF_EXACT') == '1' else 'default'
digest = hashlib.sha256()
def words(got, want):
    a = np.ascontiguousarray(got).view(np.uint32)
    b = np.ascontiguousarray(want).view(np.uint32)
    digest.update(a.tobytes())
    return int(np.count_nonzero(a != b))

if unit == 'mynn_surface':
    p = subprocess.run([sys.executable, str(Path(__file__).with_name('mynn_surface.py')), 'subnormal', 'run'])
    row = json.loads((L / 'receipts' / f'subnormal-{mode}.json').read_text())
    bad = row['independent_audit']['mismatched']
    actual_sha256 = row['independent_audit'].get('actual_sha256')
elif unit == 'mynn_pbl':
    subprocess.run([sys.executable, str(Path(__file__).with_name('pbl_run.py'))], check=True)
    row = json.loads((L / 'receipts' / f'extra-{os.environ.get("GPUWM_WRF_EXACT", "default")}.json').read_text())
    bad = sum(r['outputs']['qsq']['bits'] for r in row['configs'])
    assert len(row['configs']) == 28
    actual_sha256 = hashlib.sha256(''.join(r.get('actual_sha256','') for r in row['configs']).encode()).hexdigest()
elif unit == 'classic':
    from tools.sfclay_classic_wrf461_oracle import validate_sfclay_classic_oracle as c
    original = c.woof_step
    def step(*args, **kwargs):
        got = original(*args, **kwargs)
        digest.update(got.tobytes())
        return got
    c.woof_step = step
    row = c.grade(L / 'oracles/classic/fringe-columns.txt', L / 'oracles/classic/fringe-corrected.txt')
    bad = sum(sum(row[m]['differing_words'].values()) for m in ('free', 'replay'))
elif unit == 'revised':
    import _sfclayrev_oracle as c
    fixture = c.load_fixture(L / 'oracles/revised-subnormal')
    got = c.port_outputs(fixture)
    fields = __import__('gpuwm.core.physics_inventory', fromlist=['SFCLAY_OUTPUTS']).SFCLAY_OUTPUTS
    counts = {}
    for j, name in enumerate(fields):
        a, b = got[..., j], fixture.outputs[..., j]
        if name == 'lh':
            a, b = a[:5], b[:5]  # WRF flux-off LH is an unassigned local.
        counts[name] = words(a, b)
    bad = sum(counts.values())
    row = dict(per_field=counts, compared=got.size - got.shape[1] * got.shape[2])
elif unit == 'eta':
    from gpuwm.verify import myjsfc_oracle as c
    fixture = c.load(L / 'extras/extras.npz')
    fixture = {k:v for k,v in fixture.items() if k.startswith('subnormal_probe/') or k.startswith('tables/') or k == 'fields'}
    old = c._stats
    def stats(got, want):
        out = old(got, want)
        out['mismatch'] = words(got, want)
        return out
    c._stats = stats
    row = c.measure(fixture)
    bad = sum(sum(v['mismatch'] for v in row[m].values()) for m in ('replay', 'free_run'))
else:
    raise ValueError(unit)
row['mismatches'] = bad
row['actual_sha256'] = locals().get('actual_sha256') or digest.hexdigest()
(L / 'receipts' / f'{unit}-{mode}.json').write_text(json.dumps(row, indent=1) + '\n')
print(unit, mode, 'mismatches', bad, 'actual_sha256', row['actual_sha256'])
raise SystemExit(bool(bad))
