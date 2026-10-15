#!/usr/bin/env bash
# Build and run the Grell 3D (cu_physics = 5) oracle against the
# byte-unmodified WRF v4.6.1 phys/module_cu_g3.F.
#
# THE FILE SET IS ONE FILE.  module_cu_g3.F USEs nothing: G3DRV,
# CUP_enss_3d, every cup_* helper, g3init, massflx_stats, cup_axx and
# conv_grell_spread3d live in that one module, and `nm -u` on the object is
# the receipt (libm and libgfortran only).
#
# THE CPP DEFINES ARE REQUIRED.  G3DRV's high-resolution forcing
# average (ave_f_t/ave_f_q, :252-281) sits inside `#if ( EM_CORE == 1 )`.
# cpp evaluates an undefined identifier as 0, so a build that forgets
# -DEM_CORE=1 compiles the block OUT and the cugd_avedx > 1 arm silently
# feeds the scheme zero forcing.  WRF's ARW build defines it through
# COREDEFS (configure.wrf).  WRF_DFI_RADAR is left undefined, as WRF's own
# default build leaves it, so the cap-suppress arm is compiled out.
#
# TWO BUILDS, ONE ANSWER.  The reference objects are -O0 (the house rule:
# no vectoriser, no contraction).  A second build of the same pinned source
# at the scalar WRF tree's own FCOPTIM (-O2 -fno-fast-math
# -ffp-contract=off -fno-tree-vectorize, /work/wrf/build/scalar) runs the
# same inputs and the script FAILS unless every output word agrees.  When
# WRF_BUILD_F90 points at the preprocessed module_cu_g3.f90 the scalar WRF
# build actually compiled, a third build of THAT text joins the agreement.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [FIXTURE_DIR]" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
fixture_dir=${3:-}
if [[ -n "$fixture_dir" ]]; then fixture_dir=$(realpath -m "$fixture_dir"); fi
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

declare -A pinned=(
  ["phys/module_cu_g3.F"]="68089efd157753e8d050c19d4e65f711838147381639014f5f0be40576a337e9"
)
for rel in "${!pinned[@]}"; do
    actual=$(sha256sum "${source_root}/${rel}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$rel]}" ]]; then
        echo "${rel} is ${actual}, not the pinned ${pinned[$rel]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"/{O0,O2,wrf,ub-poison}
export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"
cd "${build_dir}"
cpp_defs=(-DEM_CORE=1 -DNMM_CORE=0 -DDA_CORE=0 -DWRFPLUS=0 -DRWORDSIZE=4
          -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4)
ffree=(-ffree-form -ffree-line-length-none)

build_one() {  # dir flags...
    local dir=$1; shift
    ( cd "${dir}"
      gfortran -c "$@" -cpp "${cpp_defs[@]}" "${ffree[@]}" \
          -o module_cu_g3.o "${source_root}/phys/module_cu_g3.F"
      gfortran -c "$@" -cpp "${ffree[@]}" -I . \
          -o run_g3.o "${script_dir}/run_g3.F90"
      gfortran -o run_g3 module_cu_g3.o run_g3.o )
}
# THE ONE UNINITIALISED READ IS PINNED, AND A CONTROL PROVES IT IS THE ONLY
# ONE.  With ishallow = 0, G3DRV writes K22/KBCON/KTOP_SHALLOW from locals
# CUP_enss_3d never assigns (module_cu_g3.F:487-490 vs :1223) -- MEASURED,
# the -O0 and -O2 builds write different stack words there.  The reference
# pins every uninitialised local to zero, which is G3DRV's own "no shallow
# cloud" value (:237-241) and the value the port implements (a declared
# divergence, never bit-exact to a bug); the ub-poison build runs the same
# source with every such local poisoned and ub_control.py fails the build
# if anything but those three fields, in the band, under ishallow = 0, moves.
build_one O0 -O0 -finit-local-zero
build_one O2 -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize -finit-local-zero
build_one ub-poison -O0 -finit-integer=-123456 -finit-real=nan -finit-logical=true

have_wrf=0
if [[ -n "${WRF_BUILD_F90:-}" && -f "${WRF_BUILD_F90}" ]]; then
    ( cd wrf
      gfortran -c -O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize \
          -finit-local-zero "${ffree[@]}" -o module_cu_g3.o "${WRF_BUILD_F90}"
      gfortran -c -O0 -cpp "${ffree[@]}" -I . -o run_g3.o "${script_dir}/run_g3.F90"
      gfortran -o run_g3 module_cu_g3.o run_g3.o )
    have_wrf=1
    sha256sum "${WRF_BUILD_F90}" > wrf-f90-sha256.txt
fi

# --- receipts ---------------------------------------------------------------
nm -u O0/module_cu_g3.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol" >&2; exit 4
fi
gfortran --version | head -1 > compiler.txt

# --- inputs and runs --------------------------------------------------------
mkdir -p cases
python3 "${script_dir}/gen_g3_cases.py" cases
for f in cases/g3-in-*.bin; do
    cfg=$(basename "$f" .bin); cfg=${cfg#g3-in-}
    read -r -a dims <<< "$(python3 -c 'from g3_fixture_io import read_input; import sys; d=read_input(sys.argv[1]); print(*(d[k] for k in ("maxiens","maxens","maxens2","maxens3","ensdim","periodic_x","periodic_y")))' "$f")"
    ./O0/run_g3 "$f" "cases/g3-out-${cfg}.bin" "${dims[@]}"
    ./O2/run_g3 "$f" "cases/g3-out-O2-${cfg}.bin" "${dims[@]}"
    if ! cmp -s "cases/g3-out-${cfg}.bin" "cases/g3-out-O2-${cfg}.bin"; then
        echo "FATAL: -O0 and scalar -O2 builds disagree on config ${cfg}" >&2
        exit 6
    fi
    ./ub-poison/run_g3 "$f" "cases/g3-out-poison-${cfg}.bin" "${dims[@]}"
    python3 "${script_dir}/ub_control.py" "$f" "cases/g3-out-${cfg}.bin"         "cases/g3-out-poison-${cfg}.bin" "${dims[@]}"
    if [[ ${have_wrf} -eq 1 ]]; then
        ./wrf/run_g3 "$f" "cases/g3-out-wrf-${cfg}.bin" "${dims[@]}"
        if ! cmp -s "cases/g3-out-${cfg}.bin" "cases/g3-out-wrf-${cfg}.bin"; then
            echo "FATAL: the WRF-preprocessed build disagrees on config ${cfg}" >&2
            exit 7
        fi
    fi
done

if [[ -n "${fixture_dir}" ]]; then
    python3 "${script_dir}/pack_g3_fixture.py" cases "${fixture_dir}"
fi
sha256sum cases/g3-in-*.bin cases/g3-out-?.bin compiler.txt undefined-O0.txt \
    > oracle-sha256sums.txt
echo "built and ran in ${build_dir} (wrf-preprocessed cross-build: ${have_wrf})"
