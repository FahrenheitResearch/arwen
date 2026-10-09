"""A one-bit pi/2 plant must fail the complete CK-corrected classic gate."""
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
L=Path(os.environ.get('SFCLAY_CHECK_ROOT','/work/pool-sfclay-noftz-mynn-mm5-eta'))
p=ROOT/'gpuwm/core/kernels/sfclay_classic.cuh'
original=p.read_bytes()
old=b'0x3FC90FDBu'
assert original.count(old)==1
assert (0x3FC90FDB^0x3FC90FDA)==1
receipt={'original_sha256':hashlib.sha256(original).hexdigest(),'word_xor':1}
mode='strict' if os.environ.get('GPUWM_WRF_EXACT')=='1' else 'default'
try:
    p.write_bytes(original.replace(old,b'0x3FC90FDAu'))
    sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
    from tools.sfclay_classic_wrf461_oracle import validate_sfclay_classic_oracle as c
    fixture=ROOT/'tests/data/oracles/sfclay_classic'
    measured=c.grade(fixture/'columns.txt',fixture/'wrf-ck-corrected.txt')
    receipt['table_mismatches']=measured['table']
    receipt['output_mismatches']={m:measured[m]['differing_words'] for m in ('free','replay')}
    assert measured['table']['psim']['differing']==749
    run=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_sfclay_classic_wrf461_parity.py','-k','stability_tables or every_output_word'],cwd=ROOT,text=True,capture_output=True)
    (L/'logs'/f'plant-{mode}.log').write_text(run.stdout+run.stderr)
    receipt['exit']=run.returncode
    assert run.returncode==1,run.stdout+run.stderr
    assert '3 failed' in run.stdout,run.stdout+run.stderr
finally:
    p.write_bytes(original)
    receipt['restored_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    assert p.read_bytes()==original
    (L/'receipts'/f'plant-{mode}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
