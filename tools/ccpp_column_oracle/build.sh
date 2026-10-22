#!/usr/bin/env bash
# Build and run the CCPP column oracle for the GSL drag suite.
#
#   bash tools/ccpp_column_oracle/build.sh CCPP_SOURCE_ROOT BUILD_DIR INPUT_DIR [IMX:DELTIM ...]
#
# CCPP_SOURCE_ROOT holds the ccpp-physics files SOURCES.sha256 pins
# (fetch_sources.sh lays them out); every one is checked before anything
# compiles.  INPUT_DIR is the column set WOOF's own drag kernel is held to,
# gpuwm/data/terrain_drag/oracle/gwdo/inputs.  Compiled byte-unmodified at
# -O0 twice: once with kind_phys = 8, the precision the UFS model builds its
# physics at, and once with kind_phys = 4 (-DSINGLE_PREC, machine.F's own
# switch), the precision WOOF and WRF run, so a code difference and a
# precision difference can be told apart.
#
# Each IMX:DELTIM case runs into fixtures/imx<IMX>_dt<DELTIM>.  IMX is the
# host's global x count the scheme scales its mesoscale coefficient by
# (RRFS v1: 4 x 3950 = 15800); DELTIM is the physics step (RRFS v1: 36 s;
# the WRF fixture WOOF is held to: 20 s).  The default cases are RRFS's own
# pair, RRFS's IMX at the fixture's step, and the first receipt's 192 at
# 20 s, which reproduces that receipt.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "usage: build.sh CCPP_SOURCE_ROOT BUILD_DIR INPUT_DIR [IMX:DELTIM ...]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
input_dir=$(realpath "$3")
shift 3
cases=("$@")
[[ ${#cases[@]} -eq 0 ]] && cases=(15800:36 15800:20 192:20)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a ccpp-physics source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}
for prec in r8 r4; do
    flags=()
    [[ "${prec}" == r4 ]] && flags=(-DSINGLE_PREC)
    out="${build_dir}/${prec}"
    rm -rf "${out}"; mkdir -p "${out}/fixtures"
    ( cd "${out}"
      gfortran -c -O0 -cpp "${flags[@]}" "${source_root}/physics/hooks/machine.F"
      gfortran -c -O0 -cpp -ffree-line-length-none "${flags[@]}" \
          "${source_root}/physics/GWD/drag_suite.F90"
      gfortran -c -O0 -ffree-line-length-none "${script_dir}/run_drag_suite.F90"
      gfortran -o run_drag_suite run_drag_suite.o drag_suite.o machine.o
      for c in "${cases[@]}"; do
          imx=${c%%:*}; dt=${c##*:}
          mkdir -p "fixtures/imx${imx}_dt${dt}"
          ./run_drag_suite "${input_dir}" "fixtures/imx${imx}_dt${dt}" "${imx}" "${dt}"
      done )
done
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "host: $(uname -srm)"
} > "${build_dir}/TOOLCHAIN.txt"
( cd "${script_dir}" && sha256sum build.sh fetch_sources.sh run_drag_suite.F90 \
    compare_drag_suite.py woof_drag_columns.py ) > "${build_dir}/HARNESS.sha256"
echo "cases: ${cases[*]}" >> "${build_dir}/TOOLCHAIN.txt"
echo "fixtures in ${build_dir}/r8/fixtures and ${build_dir}/r4/fixtures"
