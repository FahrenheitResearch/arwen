#!/usr/bin/env bash
# Build the WRF v4.7.1 slope_rad / topo_shading oracle.
#
#   bash build.sh <WRF_SOURCE_ROOT> <BUILD_DIR>
#
# WRF_SOURCE_ROOT holds phys/module_radiation_driver.F,
# phys/module_surface_driver.F, dyn_em/start_em.F, dyn_em/nest_init_utils.F
# and share/module_model_constants.F at tag v4.7.1; extract.py refuses any
# other bytes.  The WRF statements are compiled with WRF's own gfortran
# flags (configure.defaults, Linux x86_64 gfortran: -O2 -ftree-vectorize
# -funroll-loops, big-endian records; no -march, so no fused multiply-add
# can be emitted, and the objects are checked for one anyway).
#
# Two executables:
#   oracle        glibc's libm, as wrf.exe runs.
#   oracle_crlibm the same objects linked against libm_cr.c, whose float
#                 sinf/cosf/.../powf are the correctly rounded value of the
#                 double function.  The port evaluates its transcendentals
#                 that way, so this build separates any libm difference
#                 from an arithmetic one: the port must match it bit for
#                 bit, and the stock build measures the libm seam.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$1"; out="$2"
python3 "$here/extract.py" "$src" "$out"
cp "$here/driver.F90" "$here/libm_cr.c" "$out/"
cd "$out"
wrf_flags=(-ffree-form -ffree-line-length-none -O2 -ftree-vectorize
           -funroll-loops -fconvert=big-endian -frecord-marker=4 -w)
gfortran -cpp -DEM_CORE=1 "${wrf_flags[@]}" -c module_model_constants.F
gfortran "${wrf_flags[@]}" -c wrf_topo_blocks.F90
gfortran "${wrf_flags[@]}" -c wrf_topo_inline.F90
gfortran "${wrf_flags[@]}" -c wrf_blend_terrain.F90
# The driver only does I/O: native byte order for the stream files.
gfortran -O0 -c driver.F90
if objdump -d module_model_constants.o wrf_topo_blocks.o wrf_topo_inline.o wrf_blend_terrain.o \
        | grep -Eqi 'vfn?m(add|sub)'; then
  echo "refusing: a WRF object holds a fused multiply-add" >&2
  exit 1
fi
gcc -O2 -c libm_cr.c
gfortran -o oracle driver.o wrf_topo_blocks.o wrf_topo_inline.o \
         wrf_blend_terrain.o module_model_constants.o -lm
gfortran -o oracle_crlibm driver.o wrf_topo_blocks.o wrf_topo_inline.o \
         wrf_blend_terrain.o module_model_constants.o libm_cr.o -lm
echo "built $out/oracle and $out/oracle_crlibm"
gfortran --version | head -1
ldd --version | head -1
