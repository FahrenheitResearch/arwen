#!/usr/bin/env bash
set -euo pipefail
[ "$#" = 2 ] || { echo 'usage: build.sh WRF_ROOT BUILD_DIR' >&2; exit 2; }
D=$(cd -- "$(dirname -- "$0")" && pwd)
W=$(realpath "$1")
B=$(realpath -m "$2")
[ "$(git -C "$W" rev-parse HEAD)" = d66e442fccc04111067e29274c9f9eaccc3cef28 ] || { echo 'wrong WRF commit' >&2; exit 2; }
git -C "$W" diff --quiet HEAD -- phys/module_ra_goddard.F run/BROADBAND_CLOUD_GODDARD.bin || { echo 'edited WRF source or cloud table' >&2; exit 2; }
mkdir -p "$B"
# Visibility only. No changes to expressions, constants or solver kind.
python - "$W" "$B" <<'PY'
import pathlib,sys
w,b=map(pathlib.Path,sys.argv[1:])
s=(w/'phys/module_ra_goddard.F').read_text()
old='public :: init_goddardrad, goddardrad'
assert s.count(old)==1
s=s.replace(old,old+', swrad, lwrad, opt_cloud, fp_kind, fast_overcast')
import os
if os.environ.get('GODDARD_DEFINED_TAUX')=='1':
    needle=' call sw_uvpar (np,wh,oh,dp, &'
    assert s.count(needle)==1
    s=s.replace(needle,' taux = taucl(:,:,8)\n'+needle)
(b/'module_ra_goddard.F').write_text(s)
PY
cd "$B"
F=(-O2 -fno-tree-vectorize -ffp-contract=off -fconvert=big-endian -frecord-marker=4 -ffree-form -ffree-line-length-none -fcheck=bounds)
FC=${FC:-gfortran}
"$FC" "${F[@]}" -c "$D/stub_wrf.F90"
"$FC" "${F[@]}" -cpp -DWRF_CHEM=0 -DRWORDSIZE=4 $(nf-config --fflags) -c module_ra_goddard.F
"$FC" "${F[@]}" -c "$D/run_columns.F90"
"$FC" "${F[@]}" -o run_columns stub_wrf.o module_ra_goddard.o run_columns.o $(nf-config --flibs)
if nm -D run_columns | grep -q _ZGV; then echo 'vector math reached oracle' >&2; exit 3; fi
ln -s "$W/run/BROADBAND_CLOUD_GODDARD.bin" BROADBAND_CLOUD_GODDARD.bin
./run_columns columns.bin | tee run.log
sha256sum "$W/phys/module_ra_goddard.F" "$W/run/BROADBAND_CLOUD_GODDARD.bin" "$D/build.sh" "$D/stub_wrf.F90" "$D/run_columns.F90" module_ra_goddard.F run_columns columns.bin | tee SHA256SUMS
"$FC" --version | head -1 > toolchain.txt
printf '%s\n' "${F[*]} -cpp -DWRF_CHEM=0 -DRWORDSIZE=4" >> toolchain.txt
