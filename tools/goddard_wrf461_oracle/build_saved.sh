#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); W=$(realpath "$1"); B=$(realpath -m "$2"); mkdir -p "$B"
python - "$W" "$B" <<'PY'
from pathlib import Path
import sys
w,b=map(Path,sys.argv[1:]);s=(w/'phys/module_ra_goddard.F').read_text()
s=s.replace('public :: init_goddardrad, goddardrad','public :: init_goddardrad, goddardrad, swrad,lwrad,fp_kind,fast_overcast')
s=s.replace(' call sw_uvpar (np,wh,oh,dp, &',' taux = taucl(:,:,8)\n call sw_uvpar (np,wh,oh,dp, &')
(b/'module_ra_goddard.F').write_text(s)
PY
cd "$B"; F=(-O2 -fno-tree-vectorize -ffp-contract=off -fconvert=big-endian -frecord-marker=4 -ffree-form -ffree-line-length-none -fcheck=bounds)
gfortran "${F[@]}" -c "$D/stub_wrf.F90"
gfortran "${F[@]}" -cpp -DWRF_CHEM=0 -DRWORDSIZE=4 $(nf-config --fflags) -c module_ra_goddard.F
gfortran "${F[@]}" -c "$D/run_saved.F90"
gfortran -o run_saved stub_wrf.o module_ra_goddard.o run_saved.o $(nf-config --flibs)
