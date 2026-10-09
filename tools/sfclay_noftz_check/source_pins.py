"""Record source identities and update only the measured surface/PBL pins."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
L=Path(os.environ.get('SFCLAY_CHECK_ROOT','/work/pool-sfclay-noftz-mynn-mm5-eta'))
from gpuwm.core import kernels
rows={}
p=ROOT/'tests/test_mp8_frozen.py'
text=p.read_bytes()
for name in ('mynn_pbl','mynn_surface','sfclay'):
    path=ROOT/'gpuwm/core/kernels'/(name+'.cu')
    raw=hashlib.sha256(path.read_bytes()).hexdigest()
    composed=hashlib.sha256(kernels.module_source(name).encode()).hexdigest()
    rows[name]=dict(file_sha256=raw,compiled_source_sha256=composed,options=kernels.module_options(name))
    start=text.index(("    '"+name+"': (").encode())
    end=text.find(b"\n    '",start+10)
    block=text[start:end]
    old=re.findall(rb"'[0-9a-f]{64}'",block)
    assert len(old)==2,(name,old)
    block=block.replace(old[0],("'"+raw+"'").encode()).replace(old[1],("'"+composed+"'").encode())
    text=text[:start]+block+text[end:]
p.write_bytes(text)
p=ROOT/'tests/test_kernel_source_freeze_per_module.py'
text=p.read_bytes()
for name in ('sfclay_classic.cuh','surface_subnormal.cuh'):
    raw=hashlib.sha256((ROOT/'gpuwm/core/kernels'/name).read_bytes()).hexdigest()
    rows[name]=dict(file_sha256=raw)
    pattern=(rb'"'+re.escape(name.encode())+rb'": "[0-9a-f]{64}"')
    replacement=('"'+name+'": "'+raw+'"').encode()
    if re.search(pattern,text):text=re.sub(pattern,replacement,text)
    else:
        anchor=b'    "mynn_libm.cuh":'
        at=text.index(anchor)
        text=text[:at]+b'    '+replacement+b',\n'+text[at:]
p.write_bytes(text)
(L/'receipts/source-pins.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
