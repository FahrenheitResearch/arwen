"""Fire each CPU full-solver gate and restore the exact reference bytes."""
from pathlib import Path
import subprocess
import sys
import os
import json

p=Path('gpuwm/verify/chem_plumerise_ref.py'); original=p.read_bytes(); result={}
try:
    for arm,test in [('wrf','tests/test_chem_plumerise_wrf471_parity.py'),('gsl','tests/test_chem_plumerise_frp_parity.py')]:
        start=original.index(('def '+arm+'_burn(').encode())
        end=original.index(b'\ndef ',start+5)
        body=original[start:end]
        changed=body.replace(b'word(0x447a0000)',b'word(0x00000000)')
        if body==changed: raise RuntimeError('burn mutation target absent: the parity negative control would not change fire water')
        p.write_bytes(original[:start]+changed+original[end:])
        run=subprocess.run([sys.executable,'-B','-m','pytest','-q',test,'-k','cpu_solver_parity'],env={**os.environ,'GPUWM_NO_LOCAL_GPU':'1'},capture_output=True,text=True)
        result[arm]={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        if run.returncode!=1: raise RuntimeError('mutated fire-water source did not fail parity: the oracle gate would miss changed plume physics')
        p.write_bytes(original)
finally:
    p.write_bytes(original)
Path('.plume-evidence/full-mutation.json').write_text(json.dumps(result,indent=2))
print({k:v['returncode'] for k,v in result.items()})
