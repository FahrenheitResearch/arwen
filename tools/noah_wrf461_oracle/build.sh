#!/usr/bin/env bash
# Build and run the Noah LSM oracle against the byte-unmodified WRF v4.6.1
# phys/module_sf_noahlsm.F and phys/module_sf_noahdrv.F.
#
# WHAT IS AND IS NOT STUBBED.  module_sf_noahlsm.F USEs module_model_constants
# and module_wrf_error, so unlike bl_ysu.F90 this needs a stub file -- but a
# much smaller one than the RUC or Noah-MP harnesses.  Both of those modules
# are compiled HERE FROM THE PINNED TREE rather than replaced:
#
#   * module_model_constants.F carries CP, R_D, XLF, XLV, RHOWATER, STBOLT and
#     KARMAN, which are the constants the reference is measured in.  A stub is
#     free to get them wrong -- tools/ruc_wrf461_oracle/stub_wrf.F90 writes
#     `r_d = 287.0` and `cp = 1004.5` as independent literals where WRF derives
#     `cp = 7.*r_d/2.` from r_d, which is the same number only by luck.
#   * frame/module_wrf_error.F compiles standalone once DM_PARALLEL is
#     undefined, and it defines wrf_error_fatal, so stubbing that would be
#     stubbing a file that already builds.
#
# What is left in stub_wrf.F90 is service only: wrf_abort, wrf_debug, the
# single-rank forms of the wrf_dm_bcast_* shims, and abort-only link targets
# for the urban/BEP/GFDL packages this harness deliberately does not build.
#
# COMPILER DEFINES.  -Dwrfmodel is not optional cosmetics: WRF's own
# arch/postamble:26 passes it, and without it module_sf_noahdrv.F's LSMINIT
# and SOIL_VEG_GEN_PARM are preprocessed away entirely and the object exports
# only `lsm`.  A harness that missed that would have had to hand-fill the
# parameter tables, i.e. to invent them.
#
# libmvec.  WRF compiles phys/ at -O2 with -ftree-vectorize, where gfortran's
# auto-vectoriser can replace scalar expf/powf/logf with glibc's vector forms
# (_ZGVbN4v_expf, ...), whose accuracy contract is 4 ULP rather than the
# scalar routines' ~0.5.  That silently moves the reference a bit-parity gate
# is measured against.  The oracle is therefore built at -O0 and this script
# FAILS if any _ZGV* symbol appears in an -O0 object.  It also builds
# throw-away -O2 -ftree-vectorize and -Ofast objects purely as evidence, and a
# positive control that MUST emit _ZGV* -- without which the grep's silence on
# the Noah objects would prove nothing.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"

sources=(
    share/module_model_constants.F
    frame/module_wrf_error.F
    phys/module_sf_noahlsm.F
    phys/module_sf_noahlsm_glacial_only.F
    phys/module_sf_noahdrv.F
    run/VEGPARM.TBL
    run/SOILPARM.TBL
    run/GENPARM.TBL
)
for rel in "${sources[@]}"; do
    test -f "${source_root}/${rel}"
done

# --- byte identity to the pinned commit -------------------------------------
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not the pinned ${pinned_commit}" >&2
    exit 3
fi
if ! git -C "${source_root}" diff --quiet HEAD -- "${sources[@]}"; then
    echo "one of the Noah sources differs from the pinned commit" >&2
    git -C "${source_root}" diff --name-only HEAD -- "${sources[@]}" >&2
    exit 3
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

# WRF's own ARCHFLAGS for phys/ (arch/postamble), minus the I/O and chemistry
# switches nothing in the Noah column reads.
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4"
defines="${defines} -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"

# --- the reference build: -O0, no vectoriser --------------------------------
gfortran -c -O0 -cpp ${free} -fallow-argument-mismatch \
    "${script_dir}/stub_wrf.F90"
gfortran -c -O0 -cpp ${defines} ${free} \
    "${source_root}/share/module_model_constants.F"
gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch \
    "${source_root}/frame/module_wrf_error.F"
gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I "${build_dir}" \
    "${source_root}/phys/module_sf_noahlsm.F"
gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I "${build_dir}" \
    "${source_root}/phys/module_sf_noahlsm_glacial_only.F"
gfortran -c -O0 -cpp ${defines} ${free} -fallow-argument-mismatch -I "${build_dir}" \
    "${source_root}/phys/module_sf_noahdrv.F"
gfortran -c -O0 ${free} -I "${build_dir}" "${script_dir}/run_lsm.F90"
gfortran -o run_lsm stub_wrf.o module_model_constants.o module_wrf_error.o \
    module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o \
    module_sf_noahdrv.o run_lsm.o

# The driver must actually export the routines that read the tables; if
# -Dwrfmodel were ever dropped this catches it before a fixture is written.
for sym in lsm soil_veg_gen_parm lsminit; do
    if ! nm module_sf_noahdrv.o | grep -q "__module_sf_noahdrv_MOD_${sym}\$"; then
        echo "module_sf_noahdrv.o does not export ${sym}; is -Dwrfmodel set?" >&2
        exit 6
    fi
done

# --- receipts ---------------------------------------------------------------
for obj in module_sf_noahlsm module_sf_noahdrv module_sf_noahlsm_glacial_only; do
    nm -u "${obj}.o" | sed 's/^ *//' | sort > "undefined-O0-${obj}.txt"
    if grep -q '_ZGV' "undefined-O0-${obj}.txt"; then
        echo "-O0 object ${obj}.o pulled in a libmvec vector symbol:" >&2
        grep '_ZGV' "undefined-O0-${obj}.txt" >&2
        exit 4
    fi
done

# The vectorised builds exist only as evidence; they are never linked.
gfortran -c -O2 -ftree-vectorize -cpp ${defines} ${free} -fallow-argument-mismatch \
    -I "${build_dir}" -o noahlsm_O2vec.o "${source_root}/phys/module_sf_noahlsm.F"
nm -u noahlsm_O2vec.o | sed 's/^ *//' | sort > undefined-O2vec.txt
gfortran -c -Ofast -ftree-vectorize -cpp ${defines} ${free} -fallow-argument-mismatch \
    -I "${build_dir}" -o noahlsm_Ofast.o "${source_root}/phys/module_sf_noahlsm.F"
nm -u noahlsm_Ofast.o | sed 's/^ *//' | sort > undefined-Ofast.txt

# Positive control: the grep must be able to find _ZGV* on this toolchain at
# all, or its silence on the Noah objects means nothing.
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard on the" \
         "Noah objects is vacuous on this toolchain" >&2
    cat undefined-control.txt >&2
    exit 5
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 module_sf_noahlsm libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|_ZGV' \
        undefined-O0-module_sf_noahlsm.txt || true
    echo "# -O0 module_sf_noahdrv libm symbols:"
    grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|_ZGV' \
        undefined-O0-module_sf_noahdrv.txt || true
    echo "# -O2 -ftree-vectorize (WRF's own phys/ setting):"
    grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|_ZGV' undefined-O2vec.txt || true
    echo "# -Ofast -ftree-vectorize:"
    grep -E 'expf|powf|logf|log10f|atanf|sqrtf|cbrtf|_ZGV' undefined-Ofast.txt || true
    echo "# positive control (plain expf loop at -Ofast), proves the grep works:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- run --------------------------------------------------------------------
cp "${source_root}/run/VEGPARM.TBL" "${source_root}/run/SOILPARM.TBL" \
   "${source_root}/run/GENPARM.TBL" .

# Four passes over the same fixture state, differing only in the four driver
# switches gpuwm's launch_noah also exposes.  Each is a separate process, so
# no pass can inherit the previous pass's mutated state.
./run_lsm noah-lsm.csv          1 F F F > run-lsm-stdout.txt
./run_lsm noah-lsm-thcnd2.csv   2 F F F > run-lsm-thcnd2-stdout.txt
./run_lsm noah-lsm-frpcpn.csv   1 T F F > run-lsm-frpcpn-stdout.txt
./run_lsm noah-lsm-monalb.csv   1 F T T > run-lsm-monalb-stdout.txt

# A switch the fixture cannot discriminate is a switch the gate cannot check.
# This fired for real: with soil types 3 and 4 absent from the fixture,
# noah-lsm-thcnd2.csv came out BYTE-IDENTICAL to noah-lsm.csv, because
# TDFCND's opt_thcnd == 2 arm is reachable only for those two soil types
# (module_sf_noahlsm.F:4173).  run_lsm.F90 cases 41 and 42 exist to make this
# check pass, and it stays here so the coverage cannot silently regress.
for variant in thcnd2 frpcpn monalb; do
    if cmp -s noah-lsm.csv "noah-lsm-${variant}.csv"; then
        echo "noah-lsm-${variant}.csv is byte-identical to noah-lsm.csv:" \
             "no fixture column reaches what that switch changes" >&2
        exit 7
    fi
done

sha256sum "${source_root}/share/module_model_constants.F" \
    "${source_root}/frame/module_wrf_error.F" \
    "${source_root}/phys/module_sf_noahlsm.F" \
    "${source_root}/phys/module_sf_noahlsm_glacial_only.F" \
    "${source_root}/phys/module_sf_noahdrv.F" \
    "${script_dir}/stub_wrf.F90" \
    "${script_dir}/run_lsm.F90" \
    "${script_dir}/build.sh" \
    VEGPARM.TBL SOILPARM.TBL GENPARM.TBL \
    noah-lsm.csv noah-lsm-thcnd2.csv noah-lsm-frpcpn.csv noah-lsm-monalb.csv \
    > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
echo "oracle built in ${build_dir}"
