#!/usr/bin/env bash
# Build and run the Shin-Hong PBL oracle against the byte-unmodified WRF
# v4.6.1 phys/module_bl_shinhong.F.
#
# Why there is no stub_wrf.F90 here.  module_bl_shinhong.F has no USE
# statement at all and every CALL it makes (shinhong2d, tridi1n, tridin_ysu,
# mixlen, prodq2, vdifq) plus the five partition functions live inside the
# module, so -- like bl_ysu.F90 and unlike RUC / Noah-MP -- the whole scheme
# compiles as it ships.  `nm -u` on the object is the receipt: it must list
# nothing but libm and libgfortran.
#
# The byte pin is the FILE, not a git commit: the module's sha256 is checked
# against the value recorded from the v4.6.1 release tarball
# (v4.6.1.tar.gz, sha256 b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab
# 562ce9dc50b5f2), so the fixture cannot come from an edited file no matter
# how the tree was obtained.
#
# libmvec: same guard as the YSU oracle.  The reference is built at -O0 and
# this script FAILS if any _ZGV* symbol appears in the -O0 object; throw-away
# -O2 -ftree-vectorize and -Ofast objects are built purely as evidence of
# what WRF's own phys/ setting would have done on this toolchain, and the
# positive control proves the grep can fire at all.
#
# pow semantics: module_bl_shinhong.F is full of real-exponent powers
# (**2., **3., **4., **h1, **h2, **(-1./4.), **(-1./2.), **(-0.18), **1.5,
# **0.875, **(pfac_q-pfac) == **0.).  pow_probe.F90 prints the bit pattern
# of every such form beside its algebraic decompositions on THIS toolchain
# at -O0, so the port matches measured semantics instead of guessed ones.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
shinhong_source="${source_root}/phys/module_bl_shinhong.F"
pinned_sha256="99f44dbeb5e586b96be14424b8ab27c9986ffbd81f007f41fb8528d8ea466d56"

test -f "${shinhong_source}"

# --- byte identity to the pinned v4.6.1 release ------------------------------
actual_sha256=$(sha256sum "${shinhong_source}" | awk '{print $1}')
if [[ "${actual_sha256}" != "${pinned_sha256}" ]]; then
    echo "module_bl_shinhong.F is ${actual_sha256}, not the pinned" \
         "${pinned_sha256}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the reference build: -O0, no vectoriser --------------------------------
gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -o module_bl_shinhong_O0.o "${shinhong_source}"
gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/run_bl_shinhong.F90"
gfortran -o run_bl_shinhong module_bl_shinhong_O0.o run_bl_shinhong.o
gfortran -O0 -o pow_probe "${script_dir}/pow_probe.F90"

# --- receipts ---------------------------------------------------------------
nm -u module_bl_shinhong_O0.o | sed 's/^ *//' | sort > undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi
if grep -Eqv '^(U )?(_gfortran_|__gfortran_|GCC_|expf|powf|sqrtf|tanhf|logf|cbrtf|__powf_finite|__expf_finite|_ZGV|__stack_chk_fail|memset|memcpy|memmove|malloc|free)' \
        <(sed 's/^U *//' undefined-O0.txt); then
    echo "note: undefined symbols beyond libm/libgfortran are listed in" \
         "undefined-O0.txt; review before trusting the no-stub claim" >&2
fi

# The vectorised builds exist only as evidence; they are never linked.
gfortran -c -O2 -ftree-vectorize -cpp -ffree-form -ffree-line-length-none \
    -o module_bl_shinhong_O2vec.o "${shinhong_source}"
nm -u module_bl_shinhong_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
gfortran -c -Ofast -ftree-vectorize -cpp -ffree-form -ffree-line-length-none \
    -o module_bl_shinhong_Ofast.o "${shinhong_source}"
nm -u module_bl_shinhong_Ofast.o | sed 's/^ *//' | sort > undefined-Ofast.txt

# Positive control: the grep must be able to find _ZGV* on this toolchain at
# all, or its silence on module_bl_shinhong_O0.o means nothing.
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard on" \
         "module_bl_shinhong_O0.o is vacuous on this toolchain" >&2
    cat undefined-control.txt >&2
    exit 5
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|tanhf|_ZGV' undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize (WRF's own phys/ setting):"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|tanhf|_ZGV' undefined-O2vec.txt || true
    echo "# -Ofast -ftree-vectorize:"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|tanhf|_ZGV' undefined-Ofast.txt || true
    echo "# positive control (plain expf loop at -Ofast), proves the grep works:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- run --------------------------------------------------------------------
./pow_probe > pow-probe.txt
./run_bl_shinhong shinhong-levels.csv shinhong-surface.csv shinhong-partition.csv

sha256sum "${shinhong_source}" \
    "${script_dir}/run_bl_shinhong.F90" \
    "${script_dir}/pow_probe.F90" \
    shinhong-levels.csv shinhong-surface.csv shinhong-partition.csv \
    pow-probe.txt > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
