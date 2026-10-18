#!/usr/bin/env bash
# Fresh corrected referee objects in a private build, never the WRF tree.
# usage: build_corrected.sh WRF_SOURCE_ROOT REGIME_BUILD [PYTHON]
set -euo pipefail
source_root=$(realpath "$1")
build_dir=$(realpath "$2")
python=${3:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$build_dir/corrected"
"$python" - "$source_root" "$build_dir" <<'PY'
from pathlib import Path
import hashlib, json, re, sys
src = Path(sys.argv[1]) / 'phys/module_cu_gf_sh.F'
build = Path(sys.argv[2])
raw = src.read_bytes()
assert hashlib.sha256(raw).hexdigest() == '0dd1ed7c620b39e13f7ef77dc4ff1a269fa1f8d746492d360e203df0215b529a'
s = raw.decode()
old = 'k22(i)=maxloc(HEO_CUP(i,2:kbmax(i)),1)'
assert s.count(old) == 1
# The pressure difference uses k+1. Cap the loop in BOTH referee arms.
pattern = r'(?im)^(\s*do\s+k\s*=\s*1\s*,\s*)kpbl\(i\)(\s*)$'
safe, count = re.subn(pattern, r'\1min(kpbl(i),ktf-1)\2', s)
assert count == 1, count
corrected = safe.replace(old, old + '+1')
for name, text in [('safe', safe), ('corrected', corrected)]:
    (build/'corrected'/f'{name}.F').write_text(text)
(build/'corrected'/'provenance.json').write_text(json.dumps(dict(
    original_sha256=hashlib.sha256(raw).hexdigest(),
    safe_sha256=hashlib.sha256(safe.encode()).hexdigest(),
    corrected_sha256=hashlib.sha256(corrected.encode()).hexdigest(),
    changes=['closure k <= min(kpbl,ktf-1) in both arms',
             'MAXLOC section lower bound offset +1 in corrected arm']), indent=2)+'\n')
PY
cd "$build_dir"
objects=(ref/module_gfs_machine.o ref/module_gfs_physcons.o ref/module_cu_gf_deep.o)
tail=(ref/module_cu_gf_wrfdrv.o ref/run_gf_columns.o gf_crgamma_tgammaf.o)
for arm in safe corrected; do
    gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -fcheck=bounds \
        -I ref -J corrected -o "corrected/$arm.o" "corrected/$arm.F"
    gfortran -o "run_$arm" "${objects[@]}" "corrected/$arm.o" "${tail[@]}" -lquadmath
    "./run_$arm" gf-regime-inputs.bin "gf-regime-wrf-$arm-k0.bin" 0
done
sha256sum corrected/*.F corrected/*.o run_safe run_corrected \
    gf-regime-wrf-{safe,corrected}-k0.bin > corrected/sha256sums.txt
