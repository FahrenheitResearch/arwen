#!/usr/bin/env bash
# Build and run the terrain-drag column oracles against WRF v4.7.1.
#
#   bash tools/terrain_drag_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT holds the files SOURCES.sha256 pins (fetch_sources.sh lays
# them out); every one is checked before anything compiles, so a fixture can
# never come from an edited file.  What is compiled, all byte-unmodified:
#
#   phys/ccpp_kind_types.F           (-DRWORDSIZE=4, WRF's default: kind_phys
#                                     is real(4))
#   phys/physics_mmm/bl_ysu.F90      YSU, for the topo_wind arm (ctopo/ctopo2)
#   phys/physics_mmm/bl_gwdo.F90     gwd_opt = 1 scheme body
#   phys/module_bl_gwdo.F            gwd_opt = 1 WRF wrapper (gwdo)
#   phys/module_bl_gwdo_gsl.F        gwd_opt = 3, the GSL drag suite
#   dyn_em/start_em.F:1539-1626      LAP_HGT, CTOPO, CTOPO2 (cut by extract.py)
#
# plus ../urban_wrf471_oracle/oracle_io.F90 (the fixture writer), gwd_cases.F90
# (the orographic-drag columns) and the run_*.F90 drivers here.  Each driver
# writes its fixture under BUILD_DIR/fixtures; copy that directory to
# gpuwm/data/terrain_drag/oracle/ to publish it.
#
# Reference build: -O0.  WRF builds phys/ at -O2 -ftree-vectorize, where
# gfortran may call glibc's 4-ULP vector expf/powf/logf (libmvec).  The script
# fails if an -O0 object references a _ZGV* symbol, proves with a positive
# control that the grep can see one on this toolchain, and records which
# vector symbols WRF's own -O2 flags would pull in (libmvec-report.txt).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- byte identity to the pinned sources ------------------------------------
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -rf fixtures *.o *.mod run_*
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"

ff=(-ffree-form -ffree-line-length-none)
gfortran -c -O0 -cpp -DRWORDSIZE=4 "${ff[@]}" \
    "${source_root}/phys/ccpp_kind_types.F"
gfortran -c -O0 -cpp "${ff[@]}" -o bl_ysu.o \
    "${source_root}/phys/physics_mmm/bl_ysu.F90"
gfortran -c -O0 -cpp "${ff[@]}" -o bl_gwdo.o \
    "${source_root}/phys/physics_mmm/bl_gwdo.F90"
gfortran -c -O0 -cpp "${ff[@]}" -o module_bl_gwdo.o \
    "${source_root}/phys/module_bl_gwdo.F"
gfortran -c -O0 -cpp "${ff[@]}" -o module_bl_gwdo_gsl.o \
    "${source_root}/phys/module_bl_gwdo_gsl.F"
gfortran -c -O0 "${ff[@]}" wrf_topo_wind_inline.F90
gfortran -c -O0 "${ff[@]}" "${script_dir}/../urban_wrf471_oracle/oracle_io.F90"
gfortran -c -O0 "${ff[@]}" "${script_dir}/gwd_cases.F90"

wrf_objects=(bl_ysu.o bl_gwdo.o module_bl_gwdo.o module_bl_gwdo_gsl.o
             wrf_topo_wind_inline.o)

# --- libmvec receipts --------------------------------------------------------
: > libmvec-report.txt
for obj in "${wrf_objects[@]}"; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 4
    fi
    echo "# ${obj} -O0 libm symbols:" >> libmvec-report.txt
    nm -u "${obj}" | grep -E 'expf|powf|logf|sqrtf|tanhf|atan2f|sinf|cosf|cbrtf' \
        >> libmvec-report.txt || true
done
for src in phys/physics_mmm/bl_ysu.F90 phys/physics_mmm/bl_gwdo.F90 \
           phys/module_bl_gwdo_gsl.F; do
    gfortran -c -O2 -ftree-vectorize -funroll-loops -cpp "${ff[@]}" \
        -I "${build_dir}" -o O2vec.o "${source_root}/${src}"
    echo "# ${src} at WRF's -O2 -ftree-vectorize -funroll-loops:" >> libmvec-report.txt
    nm -u O2vec.o | grep -E 'expf|powf|logf|sqrtf|tanhf|atan2f|_ZGV' \
        >> libmvec-report.txt || true
    rm -f O2vec.o
done
cat > libmvec_positive_control.F90 <<'EOF'
subroutine positive_control(x, n)
  integer, intent(in) :: n
  real, intent(inout) :: x(n)
  integer :: i
  do i = 1, n
    x(i) = exp(x(i))
  end do
end subroutine positive_control
EOF
gfortran -c -Ofast -ftree-vectorize -o libmvec_positive_control.o \
    libmvec_positive_control.F90
if ! nm -u libmvec_positive_control.o | grep -q '_ZGV'; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi
echo "# positive control (plain expf loop at -Ofast):" >> libmvec-report.txt
nm -u libmvec_positive_control.o | grep -E 'expf|_ZGV' >> libmvec-report.txt

# --- drivers -------------------------------------------------------------------
for drv in run_topo_static run_ysu_topo run_gwdo run_gwdo_gsl; do
    gfortran -c -O0 "${ff[@]}" "${script_dir}/${drv}.F90"
done
gfortran -o run_topo_static run_topo_static.o wrf_topo_wind_inline.o oracle_io.o
gfortran -o run_ysu_topo run_ysu_topo.o bl_ysu.o ccpp_kind_types.o oracle_io.o
gfortran -o run_gwdo run_gwdo.o gwd_cases.o module_bl_gwdo.o bl_gwdo.o \
    ccpp_kind_types.o oracle_io.o
gfortran -o run_gwdo_gsl run_gwdo_gsl.o gwd_cases.o module_bl_gwdo_gsl.o oracle_io.o

mkdir -p fixtures
for drv in run_topo_static run_ysu_topo run_gwdo run_gwdo_gsl; do
    "./${drv}" fixtures
done

# --- receipts -------------------------------------------------------------------
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > fixtures/TOOLCHAIN.txt
cp libmvec-report.txt fixtures/
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" ) \
    > fixtures/SOURCES.sha256
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh extract.py gwd_cases.F90 run_*.F90 \
    ../urban_wrf471_oracle/oracle_io.F90 ) > fixtures/HARNESS.sha256
( cd fixtures && find . -type f \( -name '*.bin' -o -name 'MANIFEST.txt' \) \
    | LC_ALL=C sort | xargs sha256sum ) \
    > fixtures/FIXTURES.sha256
echo "fixtures in ${build_dir}/fixtures"
