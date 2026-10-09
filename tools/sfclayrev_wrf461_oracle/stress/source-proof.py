"""Small source receipt, without launching CUDA or compiling anything."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from gpuwm.core import kernels

repo = Path(__file__).resolve().parents[3]
paths = ('gpuwm/core/kernels/sfclay.cu', 'gpuwm/core/sfclay.py',
         'tools/sfclayrev_wrf461_oracle/run_sfclayrev.F90',
         'tests/data/oracles/sfclayrev/nonfinite-inputs.hex')
receipt = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
    text=True).strip(), python=sys.version,
    module_options=list(kernels.module_options('sfclay')),
    module_source_sha256=hashlib.sha256(kernels.module_source('sfclay').encode()).hexdigest(),
    files={path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths})
print(json.dumps(receipt, indent=2))
