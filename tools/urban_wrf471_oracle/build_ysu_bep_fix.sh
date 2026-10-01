#!/usr/bin/env bash
# Build and run the YSU flag_bep column oracle twice: once from the
# byte-unmodified phys/physics_mmm/bl_ysu.F90 (the stock fixture, which must
# reproduce gpuwm/data/urban/oracle/bep/ysu_bep/columns byte for byte), and
# once with the one-line change gpuwm carries as a declared divergence
# (gpuwm/core/kernels/ysu.cu, tests/test_ysu_bep_rural_drag.py).
#
#   bash tools/urban_wrf471_oracle/build_ysu_bep_fix.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT is the tree build.sh names (WRF v4.7.1 with phys/physics_mmm
# at MMM-physics 20240626-MPASv8.2).  Only the two files run_ysu_bep.F90 links
# are read, each checked against SOURCES.sha256 before anything compiles.
#
# THE CHANGE.  bl_ysu.F90:1313-1314 removes only the URBAN fraction of YSU's
# own surface drag from the first-level diagonal,
#     ad(i,1) = ad(i,1) - bepswitch*frc_urb1d(i)*(fric*vconvlim + ...)
# while module_sf_noahdrv.F:1708-1711 already folds the rural drag,
# (1-frc)*(-ust*ust)/dz8w/|U|, into a_u_bep/a_v_bep, which :1359-1368 add to
# the same diagonal: the rural surface drag is counted twice.  The fixed
# build drops the frc_urb1d(i)* factor, so YSU's own drag is removed whole and
# the surface drag enters once, through a_u_bep, as heat does through b_t_bep.
# Nothing else in the file changes; the script checks that the edit landed
# exactly once and that the two sources differ in that one line only.
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: build_ysu_bep_fix.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for rel in phys/ccpp_kind_types.F phys/physics_mmm/bl_ysu.F90; do
    want=$(awk -v n="$rel" '$2 == n {print $1; exit}' "$here/SOURCES.sha256")
    got=$(sha256sum "$src/$rel" | cut -d' ' -f1)
    if [[ -z "$want" || "$want" != "$got" ]]; then
        echo "$rel: sha256 $got, pinned ${want:-nothing}" >&2
        exit 3
    fi
done

defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="$defines -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
for arm in stock fix; do
    dir="$build/$arm"
    rm -rf "$dir"
    mkdir -p "$dir/fixtures"
    cd "$dir"
    cp "$src/phys/physics_mmm/bl_ysu.F90" bl_ysu.F90
    if [[ "$arm" == fix ]]; then
        sed -i '1313s/bepswitch\*frc_urb1d(i)\* &$/bepswitch* \&/' bl_ysu.F90
        [[ "$(sed -n 1313p bl_ysu.F90)" == "        ad(i,1) = ad(i,1) - bepswitch* &" ]] \
            || { echo "line 1313 moved; the fix did not apply" >&2; exit 4; }
        changed=$(diff "$src/phys/physics_mmm/bl_ysu.F90" bl_ysu.F90 | grep -c '^[<>]' || true)
        [[ "$changed" -eq 2 ]] || { echo "the fix touched $changed lines, not one" >&2; exit 4; }
    fi
    gfortran -c -O0 -cpp $defines $free -fallow-argument-mismatch "$src/phys/ccpp_kind_types.F"
    gfortran -c -O0 -cpp $defines $free -fallow-argument-mismatch bl_ysu.F90
    nm -u bl_ysu.o | grep -q '_ZGV' && { echo "-O0 bl_ysu.o pulled in a libmvec symbol" >&2; exit 4; }
    gfortran -c -O0 $free "$here/oracle_io.F90"
    gfortran -c -O0 -cpp $defines $free "$here/run_ysu_bep.F90"
    gfortran -o run_ysu_bep ccpp_kind_types.o bl_ysu.o oracle_io.o run_ysu_bep.o
    mkdir -p fixtures/ysu_bep
    ./run_ysu_bep fixtures/ysu_bep > run_ysu_bep-stdout.txt
    sha256sum bl_ysu.F90 > bl_ysu.sha256
done
{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# stock bl_ysu.F90: $(cut -d' ' -f1 "$build/stock/bl_ysu.sha256")"
    echo "# fixed bl_ysu.F90: $(cut -d' ' -f1 "$build/fix/bl_ysu.sha256")"
} > "$build/compiler.txt"
echo "ysu bep oracles written: $build/stock/fixtures/ysu_bep, $build/fix/fixtures/ysu_bep"
