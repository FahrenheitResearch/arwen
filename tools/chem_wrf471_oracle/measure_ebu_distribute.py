"""Measure the driver subset and deliberately fire its CPU parity test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from gpuwm.verify.chem_oracle import load, ulp_table
from gpuwm.verify.chem_plumerise_ref import frp_driver_override, ebu_distribute

parser = argparse.ArgumentParser()
parser.add_argument('--kind8', required=True)
parser.add_argument('--mutate', action='store_true')
args = parser.parse_args()
root = Path('tests/data/oracles/chem/smoke/ebu_distribute_gsl')
cases = load(root)
totals = {k: {'max_ulp': 0, 'n_nonzero': 0, 'n': 0} for k in
          ('k_min', 'k_max', 'flam_frac', 'ebu')}
for case in cases.values():
    k1,k2,fraction = frp_driver_override(case['frp_inst'],case['plume_k_min'],
        case['plume_k_max'],case['kpbl'],case['uspdavg2d'],case['hpbl2d'],
        wind_eff_opt=int(case['wind_eff_opt']))
    actual = dict(k_min=np.int32(k1), k_max=np.int32(k2), flam_frac=fraction,
                  ebu=ebu_distribute(k1,k2,fraction,case['ebu_in'],case['z_at_w']))
    for k,a in actual.items():
        stats = ulp_table(a,case[k])
        totals[k]['max_ulp'] = max(totals[k]['max_ulp'],stats['max_ulp'])
        totals[k]['n_nonzero'] += stats['n_nonzero']
        totals[k]['n'] += stats['n']
kind8=load(args.kind8)
diffs={k: {'max_ulp':0,'n_nonzero':0,'n':0} for k in totals}
for c in cases:
    for k in diffs:
        d=ulp_table(kind8[c][k],cases[c][k])
        diffs[k]['max_ulp']=max(diffs[k]['max_ulp'],d['max_ulp'])
        diffs[k]['n_nonzero']+=d['n_nonzero']; diffs[k]['n']+=d['n']
print(json.dumps({'cpu':totals,'kind8_cast_to_f4_vs_kind4':diffs},indent=2))
if not args.mutate:
    sys.exit(0)
path=Path('gpuwm/verify/chem_plumerise_ref.py')
original=path.read_bytes()
try:
    mutated=original.replace(b'fraction = f(.85)',b'fraction = f(.84)')
    assert mutated!=original
    path.write_bytes(mutated)
    result=subprocess.run([sys.executable,'-B','-m','pytest','-q',
        'tests/test_chem_plumerise_frp_parity.py','-k','cpu_driver_parity'],
        env={**os.environ,'GPUWM_NO_LOCAL_GPU':'1'},capture_output=True,text=True)
    print(result.stdout)
    assert result.returncode==1, result.stderr
finally:
    path.write_bytes(original)
