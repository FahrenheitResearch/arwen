#!/usr/bin/env bash
# Build the NSSL option-18 column oracle against official WRF v4.6.1.
#
#   build.sh WRF_SRC_ROOT WRF_SERIAL_BUILD_ROOT BUILD_DIR
#
# WRF_SRC_ROOT           pristine checkout, commit d66e442f (source pin)
# WRF_SERIAL_BUILD_ROOT  a stock `./compile em_real` tree of the same commit
#                        (configure 32, GNU serial) holding phys/
#                        module_mp_nssl_2mom.{f90,o,mod}
#
# Two drivers are linked, both from the same WRF-preprocessed source:
#   nssl2_column_oracle_stock   links the stock build's OWN object file
#                               (WRF's flags: -O2 -ftree-vectorize
#                               -funroll-loops ...), i.e. the code wrf.exe runs
#   nssl2_column_oracle_scalar  recompiles that .f90 with the same flags plus
#                               -fno-tree-vectorize, so every libm call is the
#                               scalar libm entry (no libmvec)
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 WRF_SRC_ROOT WRF_SERIAL_BUILD_ROOT BUILD_DIR" >&2; exit 2
fi
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
src=$(realpath "$1"); bld=$(realpath "$2"); out=$(realpath -m "$3")
commit=d66e442fccc04111067e29274c9f9eaccc3cef28
# The pinned bytes are the CRLF checkout (tools/nssl2_wrf461_composed_replay);
# a Linux checkout carries the same text with LF endings.
sha_lf=1eb1b138b75ff3b0cfe33c23779f4ec9b72e57a5455a53ef11c9e55ae0f42722
sha_crlf=5aaae368289694c929d38365d77d445e4f22291a30a48555df7a21d470b72ae3
[ "$(git -C "$src" rev-parse HEAD)" = "$commit" ] || { echo "WRF source is not $commit" >&2; exit 2; }
for root in "$src" "$bld"; do
  s=$(sha256sum "$root/phys/module_mp_nssl_2mom.F" | cut -d' ' -f1)
  [ "$s" = "$sha_lf" ] || [ "$s" = "$sha_crlf" ] || { echo "unexpected NSSL source in $root: $s" >&2; exit 2; }
done
[ -e "$out" ] && { echo "refusing to reuse $out" >&2; exit 2; }
mkdir -p "$out/stock" "$out/scalar" "$out/stages"
cd "$out"

stock_flags=(-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none
             -fconvert=big-endian -frecord-marker=4 -fallow-argument-mismatch
             -fallow-invalid-boz)
drv_flags=(-O2 -cpp -ffree-form -ffree-line-length-none)

# stock: WRF's own object and module
cp "$bld/phys/module_mp_nssl_2mom.o" "$bld/phys/module_mp_nssl_2mom.mod" stock/
( cd stock
  gfortran "${drv_flags[@]}" -c "$here/../nssl2_wrf461_composed_replay/stub_wrf.F90" -o stub_wrf.o
  gfortran "${drv_flags[@]}" -c "$here/stub_wrf3.F90" -o stub_wrf3.o
  gfortran "${drv_flags[@]}" -I. -c "$here/column_oracle.F90" -o column_oracle.o
  gfortran -o ../nssl2_column_oracle_stock column_oracle.o module_mp_nssl_2mom.o stub_wrf.o stub_wrf3.o )

# scalar: same preprocessed source, same flags, no vectoriser
cp "$bld/phys/module_mp_nssl_2mom.f90" scalar/
( cd scalar
  gfortran "${stock_flags[@]}" -fno-tree-vectorize -c module_mp_nssl_2mom.f90
  gfortran "${drv_flags[@]}" -c "$here/../nssl2_wrf461_composed_replay/stub_wrf.F90" -o stub_wrf.o
  gfortran "${drv_flags[@]}" -c "$here/stub_wrf3.F90" -o stub_wrf3.o
  gfortran "${drv_flags[@]}" -I. -c "$here/column_oracle.F90" -o column_oracle.o
  gfortran -o ../nssl2_column_oracle_scalar column_oracle.o module_mp_nssl_2mom.o stub_wrf.o stub_wrf3.o )

# fixed: the reference -- the scalar build of WRF's source with the four
# documented defects corrected (wrf_bugfix.py; README "WRF defects")
mkdir -p fixed dump
python3 "$here/wrf_bugfix.py" scalar/module_mp_nssl_2mom.f90 fixed/module_mp_nssl_2mom.f90
( cd fixed
  gfortran "${stock_flags[@]}" -fno-tree-vectorize -c module_mp_nssl_2mom.f90
  gfortran "${drv_flags[@]}" -c "$here/../nssl2_wrf461_composed_replay/stub_wrf.F90" -o stub_wrf.o
  gfortran "${drv_flags[@]}" -c "$here/stub_wrf3.F90" -o stub_wrf3.o
  gfortran "${drv_flags[@]}" -I. -c "$here/column_oracle.F90" -o column_oracle.o
  gfortran -o ../nssl2_column_oracle_fixed column_oracle.o module_mp_nssl_2mom.o stub_wrf.o stub_wrf3.o )

# dump: diagnostic only -- the fixed build plus three stage-dump CALLs and the
# GS-variable dump (instrument_gs.py); run_oracle.sh checks its final output
# is byte-identical to the fixed reference
( cd dump
  python3 "$here/instrument_gs.py" ../fixed/module_mp_nssl_2mom.f90       "$here/../../gpuwm/core/kernels/nssl2_fused_gs.cu" module_mp_nssl_2mom.f90 names.json
  gfortran "${stock_flags[@]}" -fno-tree-vectorize -c module_mp_nssl_2mom.f90
  gfortran "${drv_flags[@]}" -c "$here/gs_dump.F90" -o gs_dump.o
  gfortran "${drv_flags[@]}" -c "$here/../nssl2_wrf461_composed_replay/stub_wrf.F90" -o stub_wrf.o
  gfortran "${drv_flags[@]}" -c "$here/stub_wrf3.F90" -o stub_wrf3.o
  gfortran "${drv_flags[@]}" -I. -c "$here/column_oracle.F90" -o column_oracle.o
  gfortran -o ../nssl2_column_oracle_dump column_oracle.o module_mp_nssl_2mom.o gs_dump.o stub_wrf.o stub_wrf3.o )

{
  echo "wrf_commit $commit"
  sha256sum "$src/phys/module_mp_nssl_2mom.F" "$bld/phys/module_mp_nssl_2mom.f90" \
            "$bld/phys/module_mp_nssl_2mom.o" scalar/module_mp_nssl_2mom.o \
            fixed/module_mp_nssl_2mom.f90             nssl2_column_oracle_stock nssl2_column_oracle_scalar nssl2_column_oracle_fixed
  gfortran --version | head -n1
  ldd --version | head -n1
  echo "libmvec entries in stock object: $(nm -u stock/module_mp_nssl_2mom.o | grep -c _ZGV || true)"
  echo "libmvec entries in scalar object: $(nm -u scalar/module_mp_nssl_2mom.o | grep -c _ZGV || true)"
  echo "libm imports (stock): $(nm -u stock/module_mp_nssl_2mom.o | awk '{print $2}' | grep -v '^_gfortran\|^mem\|^malloc\|^free\|^wrf_' | tr '\n' ' ')"
  echo "FMA instructions (stock object): $(objdump -d stock/module_mp_nssl_2mom.o | grep -c 'vfmadd\|vfmsub\|vfnmadd\|vfnmsub' || true)"
} > BUILD_RECEIPT.txt
cat BUILD_RECEIPT.txt
