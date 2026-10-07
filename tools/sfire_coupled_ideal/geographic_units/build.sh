#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=''
engine=$(pwd)
native="$engine/tools/sfire_coupled_ideal/geographic_units/native-build"
python3 tools/sfire_coupled_ideal/geographic_units/extract.py "$native"
cd "$native"
nice -n 10 gfortran -O0 -ffp-contract=off -ffree-line-length-none -fcheck=all native.F90 -o native-oracle
./native-oracle
python3 - <<'PY'
from pathlib import Path
import hashlib,json,subprocess
p=Path('native-oracle')
r=dict(compiler=subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
       flags=['-O0','-ffp-contract=off','-ffree-line-length-none','-fcheck=all'],
       command='gfortran FLAGS native.F90 -o native-oracle; ./native-oracle',
       executable_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
       wrapper_sha256=hashlib.sha256(Path('native.F90').read_bytes()).hexdigest())
Path('build.json').write_text(json.dumps(r,indent=2)+'\n')
PY
cd "$engine"
python3 tools/sfire_coupled_ideal/geographic_units/pack.py "$native" tools/sfire_coupled_ideal/geographic_units/fixtures
