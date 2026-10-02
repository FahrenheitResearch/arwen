"""Record the exact engine source measured by the coupling regression."""
import hashlib
import json
from pathlib import Path
from gpuwm.core.kernels import module_source

root=Path(__file__).resolve().parents[2]
if not (root/"gpuwm/core/kernels/openbc.cu").is_file():
    root=Path.cwd()
receipt={}
for name in ("openbc","acoustic"):
    raw=root/"gpuwm/core/kernels"/(name+".cu")
    receipt[name]={"file_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(),"compiled_source_sha256":hashlib.sha256(module_source(name).encode("utf-8")).hexdigest()}
print(json.dumps(receipt,indent=2))
