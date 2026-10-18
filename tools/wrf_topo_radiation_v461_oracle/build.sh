#!/usr/bin/env bash
# Build the WRF v4.6.1 slope_rad / topo_shading column oracle.
#
#   bash build.sh <WRF_V461_SOURCE_ROOT> <BUILD_DIR>
#
# The WRF statements are cut verbatim from a pristine v4.6.1 checkout
# (extract.py checks every file and block hash) and compiled with WRF's own
# GNU flags from arch/configure.defaults, exactly as wrf.exe compiles them:
# -O2 -ftree-vectorize -funroll-loops, big-endian records, no -march, so no
# fused multiply-add can be emitted (the objects are checked for one).
#
# Two executables, same objects:
#   oracle         linked against the box's stock libm, as wrf.exe runs.
#                  This is the reference: WRF v4.6.1 on this box.
#   oracle_crlibm  the same objects linked ahead of libm_cr.c (correctly
#                  rounded float transcendentals), a diagnostic that splits
#                  a libm difference from an arithmetic one.
#
# Ubuntu's gfortran pre-includes math-vector-fortran.h, so a loop it
# vectorises may call a SIMD math entry point (_ZGV*) instead of the scalar
# one.  The build lists any such call in the WRF objects: there must be
# none, or the scalar libm is not the whole story.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$1"; out="$2"
python3 "$here/extract.py" "$src" "$out"
cp "$here/../wrf_topo_radiation_v471_oracle/driver.F90" \
   "$here/../wrf_topo_radiation_v471_oracle/libm_cr.c" "$out/"
cd "$out"
wrf_flags=(-ffree-form -ffree-line-length-none -O2 -ftree-vectorize
           -funroll-loops -fconvert=big-endian -frecord-marker=4 -w)
gfortran -cpp -DEM_CORE=1 "${wrf_flags[@]}" -c module_model_constants.F
gfortran "${wrf_flags[@]}" -c wrf_topo_blocks.F90
gfortran "${wrf_flags[@]}" -c wrf_topo_inline.F90
gfortran "${wrf_flags[@]}" -c wrf_blend_terrain.F90
# The driver only does I/O: native byte order for the stream files.
gfortran -O0 -c driver.F90
objs=(module_model_constants.o wrf_topo_blocks.o wrf_topo_inline.o
      wrf_blend_terrain.o)
if objdump -d "${objs[@]}" | grep -Eqi 'vfn?m(add|sub)'; then
  echo "refusing: a WRF object holds a fused multiply-add" >&2
  exit 1
fi
nm -u "${objs[@]}" | awk '{print $NF}' | grep -E '^_ZGV' | sort -u \
  > simd_math_calls.txt || true
echo "SIMD math entry points called by the WRF objects:" \
     "$(wc -l < simd_math_calls.txt)"
nm -u "${objs[@]}" | awk '{print $NF}' \
  | grep -E '^(sin|cos|tan|asin|acos|atan|atan2|exp|log|pow|sincos)f?$' \
  | sort -u > scalar_math_calls.txt || true
echo "scalar math entry points: $(tr '\n' ' ' < scalar_math_calls.txt)"
gcc -O2 -c libm_cr.c
gfortran -o oracle driver.o "${objs[@]}" -lm
gfortran -o oracle_crlibm driver.o "${objs[@]}" libm_cr.o -lm
echo "built $out/oracle and $out/oracle_crlibm"
gfortran --version | head -1
ldd --version | head -1
