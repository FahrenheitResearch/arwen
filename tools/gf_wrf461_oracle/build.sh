#!/usr/bin/env bash
# Build and run the Grell-Freitas cumulus oracle against the byte-unmodified
# WRF v4.6.1 phys/module_cu_gf_*.F.
#
# Why there is no stub_wrf.F90 here.  With WRF_CHEM = 0 the GF scheme is
# self-contained: module_cu_gf_deep.F and module_cu_gf_sh.F declare every
# constant they use as a local parameter and their only USE
# (module_cu_gf_ctrans) sits inside a `#if ( WRF_CHEM == 1)` guard, and
# module_cu_gf_wrfdrv.F's single USE is module_gfs_physcons -- 40 lines of
# parameters over the 16-line module_gfs_machine.  Both GFS files are
# compiled here from the SAME pinned tarball, so they are inside the byte pin
# rather than being hand-written stand-ins.  `nm -u` on the objects is the
# receipt: it must list nothing but libm and libgfortran.
#
# The byte pin is the FILE SET, not a git commit: each module's sha256 is
# checked against the value recorded from the v4.6.1 release tarball
# (v4.6.1.tar.gz, sha256 b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab
# 562ce9dc50b5f2, the same tarball the Shin-Hong oracle pinned), so the
# fixture cannot come from an edited file no matter how the tree was
# obtained.
#
# module_cu_gf_ctrans.F is pinned but NOT compiled.  It is only reachable at
# WRF_CHEM = 1, where it drags in module_HLawConst, module_state_description,
# module_scalar_tables, module_chem_utilities and module_ctrans_aqchem -- the
# whole WRF-Chem stack.  ArWen has no chemistry, so the chem arm is out of
# the port's scope; its digest is recorded so a later phase can prove which
# bytes it declined to port.
#
# libmvec: same guard as the YSU and Shin-Hong oracles.  The reference is
# built at -O0 and this script FAILS if any _ZGV* symbol appears in an -O0
# object; throw-away -O2 -ftree-vectorize and -Ofast objects are built purely
# as evidence of what WRF's own phys/ setting would have done on this
# toolchain, and the positive control proves the grep can fire at all.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- the pinned WRF v4.6.1 Grell-Freitas file set ---------------------------
# Note the v4.6.1 spelling: the WRF driver module is module_cu_gf_wrfdrv.F.
# There is no module_cu_gf_driver.F in this release -- that name belongs to
# the CCPP/UFS packaging of the same scheme.
declare -A pinned=(
  [module_cu_gf_wrfdrv]="0cead99c913b6a74f3a492aea1e1c7ef6d34f95402574336022d1a6c716a4382"
  [module_cu_gf_deep]="798ad7271963757b88be36ce01e24222f7af045dc0692908d6e7c270d6b4ea11"
  [module_cu_gf_sh]="0dd1ed7c620b39e13f7ef77dc4ff1a269fa1f8d746492d360e203df0215b529a"
  [module_cu_gf_ctrans]="61587f4f011de884388c3065fefb750e5de94953e947a0dc4793a847c6093c1e"
  [module_gfs_physcons]="2b8ef99663ba62748b57c7002f5e5b8c9338f2a825c1d32c113bd4e3bff2eb52"
  [module_gfs_machine]="300ee4f75663e89d34c8a9b8733cc07a5dd48c3d973da019d31cb76524babe25"
)

for name in "${!pinned[@]}"; do
    src="${source_root}/phys/${name}.F"
    test -f "${src}"
    actual=$(sha256sum "${src}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$name]}" ]]; then
        echo "phys/${name}.F is ${actual}, not the pinned ${pinned[$name]}" >&2
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the reference build: -O0, no vectoriser --------------------------------
# Compile order is the USE order: machine -> physcons -> deep -> sh -> wrfdrv.
# WRF_CHEM and WRF_DFI_RADAR are left undefined, so the cpp guards evaluate
# `0 == 1` and the chem and DFI-radar arms are compiled out exactly as they
# are in an ArWen-shaped WRF build.
for name in module_gfs_machine module_gfs_physcons \
            module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
        -o "${name}_O0.o" "${source_root}/phys/${name}.F"
done

gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/gf_cases.F90"
for prog in run_cu_gf run_cup_gf run_gf_stages run_gf_shallow; do
    gfortran -c -O0 -ffree-form -ffree-line-length-none \
        -I "${build_dir}" "${script_dir}/${prog}.F90"
    gfortran -o "${prog}" \
        module_gfs_machine_O0.o module_gfs_physcons_O0.o \
        module_cu_gf_deep_O0.o module_cu_gf_sh_O0.o module_cu_gf_wrfdrv_O0.o \
        gf_cases.o "${prog}.o"
done

# --- the libm probes --------------------------------------------------------
# pow_probe measures what gfortran emits for every power form in the scheme;
# gf_libm_probe is the C-side glibc ground truth the float32 reference is
# built against.  Neither is built with -ffast-math: that would let the
# compiler fold or substitute the calls and stop measuring anything.
# pow_probe USEs module_gfs_physcons so it can print the GFS constants at
# their declared real(8) width.  They are parameters, so nothing needs
# linking -- but the .mod files do need to be findable.
gfortran -O0 -ffree-form -ffree-line-length-none -I "${build_dir}" \
    -o gf_pow_probe "${script_dir}/pow_probe.F90"
gcc -O2 -o gf_libm_probe "${script_dir}/gf_libm_probe.c" -lm

# --- receipts ---------------------------------------------------------------
: > undefined-O0.txt
for name in module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    nm -u "${name}_O0.o" | sed 's/^ *//' | sed "s|^|${name} |" >> undefined-O0.txt
done
sort -o undefined-O0.txt undefined-O0.txt
if grep -q '_ZGV' undefined-O0.txt; then
    echo "-O0 object pulled in a libmvec vector symbol:" >&2
    grep '_ZGV' undefined-O0.txt >&2
    exit 4
fi

# The vectorised builds exist only as evidence; they are never linked.
mkdir -p vecevidence
: > undefined-O2vec.txt
: > undefined-Ofast.txt
for name in module_gfs_machine module_gfs_physcons \
            module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    gfortran -c -O2 -ftree-vectorize -cpp -ffree-form -ffree-line-length-none \
        -J vecevidence -o "vecevidence/${name}_O2vec.o" \
        "${source_root}/phys/${name}.F"
done
for name in module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    nm -u "vecevidence/${name}_O2vec.o" | sed 's/^ *//' \
        | sed "s|^|${name} |" >> undefined-O2vec.txt
done
rm -rf vecevidence && mkdir -p fastevidence
for name in module_gfs_machine module_gfs_physcons \
            module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    gfortran -c -Ofast -ftree-vectorize -cpp -ffree-form \
        -ffree-line-length-none -J fastevidence \
        -o "fastevidence/${name}_Ofast.o" "${source_root}/phys/${name}.F"
done
for name in module_cu_gf_deep module_cu_gf_sh module_cu_gf_wrfdrv; do
    nm -u "fastevidence/${name}_Ofast.o" | sed 's/^ *//' \
        | sed "s|^|${name} |" >> undefined-Ofast.txt
done
rm -rf fastevidence

# Positive control: the grep must be able to find _ZGV* on this toolchain at
# all, or its silence on the -O0 objects means nothing.
gfortran -c -Ofast -ftree-vectorize \
    -o libmvec_positive_control.o "${script_dir}/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o | sed 's/^ *//' | sort > undefined-control.txt
if ! grep -q '_ZGV' undefined-control.txt; then
    echo "libmvec positive control produced no _ZGV symbol; the guard on the" \
         "-O0 objects is vacuous on this toolchain" >&2
    cat undefined-control.txt >&2
    exit 5
fi

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# -O0 libm symbols (THE REFERENCE):"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|tanhf|tgammaf|__powisf2|_ZGV' \
        undefined-O0.txt || true
    echo "# -O2 -ftree-vectorize (WRF's own phys/ setting):"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|tanhf|tgammaf|__powisf2|_ZGV' \
        undefined-O2vec.txt || true
    echo "# -Ofast -ftree-vectorize:"
    grep -E 'expf|powf|logf|sqrtf|cbrtf|tanhf|tgammaf|__powisf2|_ZGV' \
        undefined-Ofast.txt || true
    echo "# positive control (plain expf loop at -Ofast), proves the grep works:"
    grep -E 'expf|_ZGV' undefined-control.txt || true
} > libmvec-report.txt
cat libmvec-report.txt

# --- run --------------------------------------------------------------------
./run_cu_gf gf-levels.csv gf-surface.csv gf-isolation.csv
./run_cup_gf gf-stage-levels.csv gf-stage-surface.csv gf-stage-consistency.csv
./run_gf_stages gf-deep-levels.csv gf-deep-surface.csv gf-deep-consistency.csv
./run_gf_shallow gf-shallow-levels.csv gf-shallow-surface.csv \
    gf-shallow-consistency.csv
./gf_pow_probe > gf-pow-probe.txt

# The stage oracle proves its own decomposition: it reruns GFDRV on every
# column and compares its reconstruction of the driver's output algebra
# against the driver's own answer, bitwise.
#
# 208 of the 216 rows are exact.  The 8 that are not are a MEASURED fact, not
# a tolerance: they carry a 1-2 ULP difference in the deep mass flux `xmb`
# that comes from how GFDRV evaluates `omeg = -g*rho*w` (:471) and
# `dhdt = cp*rthblten*pi + xlv*rqvblten` (:416).  Those constants are the GFS
# physcons ones, declared real(kind_phys) = real(8) but INITIALISED FROM
# real(4) LITERALS (`con_g = 9.80665e+0`, not `9.80665d+0`), so they carry
# float32 values widened to double, and the expression rounds differently
# from either a pure-float32 or a pure-float64 spelling.  The measurement:
#   both constants real(8):            8 rows differ
#   omeg real(4), dhdt real(8):        4 rows differ
#   omeg real(8), dhdt real(4):       14 rows differ
#   both real(4):                     10 rows differ
# No spelling reaches 0, so this is recorded rather than tuned -- picking the
# variant with the smallest count would be fitting the fixture, not matching
# WRF.  The source-faithful real(8) spelling is what ships.
#
# Consequence for the port: on those 8 rows the GFDRV fixture is authoritative
# and the stage fixture is not.  gf-stage-consistency.csv marks them.
#
# The guard fails only if the disagreement GROWS past what was measured, which
# would mean the decomposition has drifted rather than that WRF is subtle.
bad=$(awk -F, 'NR>1 && $4+0 != 0' gf-stage-consistency.csv | wc -l)
echo "stage decomposition: ${bad} of $(( $(wc -l < gf-stage-consistency.csv) - 1 )) rows differ from GFDRV"
if [[ "${bad}" -gt 8 ]]; then
    echo "stage decomposition drifted: ${bad} rows differ, 8 were measured" >&2
    awk -F, 'NR>1 && $4+0 != 0' gf-stage-consistency.csv | head -20 >&2
    exit 6
fi

# run_gf_stages replicates CUP_gf's own body statement by statement and then
# calls the real cup_gf on the same prepared column.  Unlike the GFDRV
# decomposition above, this one shares every constant and every expression
# with the module it is replicating, so there is no mixed-precision residual
# to allow for: the tolerance is ZERO differing words, on every row.  A
# nonzero count means the replication has drifted and its per-stage captures
# must not be used to grade a port.
baddeep=$(awk -F, 'NR>1 && $4+0 != 0' gf-deep-consistency.csv | wc -l)
echo "deep stage replication: ${baddeep} of $(( $(wc -l < gf-deep-consistency.csv) - 1 )) rows differ from cup_gf"
if [[ "${baddeep}" -ne 0 ]]; then
    echo "deep stage replication drifted from cup_gf on ${baddeep} rows" >&2
    awk -F, 'NR>1 && $4+0 != 0' gf-deep-consistency.csv | head -20 >&2
    exit 7
fi

# run_gf_shallow holds itself to the same bar against CUP_gf_sh, and carries a
# second column: whether the module's own answer moves with dx.  CUP_gf_sh has
# no dx argument and nothing GFDRV hands it depends on grid spacing, so it
# cannot -- and the capture is keyed by case alone on the strength of that.
# Both counts must be zero.
badshal=$(awk -F, 'NR>1 && $3+0 != 0' gf-shallow-consistency.csv | wc -l)
echo "shallow stage replication: ${badshal} of $(( $(wc -l < gf-shallow-consistency.csv) - 1 )) rows differ from CUP_gf_sh"
if [[ "${badshal}" -ne 0 ]]; then
    echo "shallow stage replication drifted from CUP_gf_sh on ${badshal} rows" >&2
    awk -F, 'NR>1 && $3+0 != 0' gf-shallow-consistency.csv | head -20 >&2
    exit 8
fi
baddx=$(awk -F, 'NR>1 && $4+0 != 0' gf-shallow-consistency.csv | wc -l)
echo "shallow dx independence: ${baddx} rows move with dx"
if [[ "${baddx}" -ne 0 ]]; then
    echo "CUP_gf_sh moved with dx on ${baddx} rows; the case-keyed capture is" \
         "no longer sufficient" >&2
    awk -F, 'NR>1 && $4+0 != 0' gf-shallow-consistency.csv | head -20 >&2
    exit 9
fi

# The C probe answers to the same glibc the Fortran oracle linked against.
# Feeding it the gamma arguments the scheme can actually reach makes the
# tgammaf question a measurement rather than an assumption.
awk '/^pgamma/ {print "g " $2; print "g " $3; print "g " $4}' \
    gf-pow-probe.txt | sort -u > gamma-queries.txt || true
if [[ -s gamma-queries.txt ]]; then
    ./gf_libm_probe < gamma-queries.txt > gamma-glibc.txt
fi

# The receipt ships in the wheel, so the two roots are replaced by the
# placeholders every other scheme's oracle receipt uses.  Without this the
# listing carries whatever absolute paths this machine happened to have,
# which is a build-host detail rather than provenance, and the release
# snapshot's developer-path check refuses it.  The hashes are unchanged.
sha256sum \
    "${source_root}/phys/module_cu_gf_wrfdrv.F" \
    "${source_root}/phys/module_cu_gf_deep.F" \
    "${source_root}/phys/module_cu_gf_sh.F" \
    "${source_root}/phys/module_cu_gf_ctrans.F" \
    "${source_root}/phys/module_gfs_physcons.F" \
    "${source_root}/phys/module_gfs_machine.F" \
    "${script_dir}/gf_cases.F90" \
    "${script_dir}/run_cu_gf.F90" \
    "${script_dir}/run_cup_gf.F90" \
    "${script_dir}/run_gf_stages.F90" \
    "${script_dir}/run_gf_shallow.F90" \
    "${script_dir}/pow_probe.F90" \
    "${script_dir}/gf_libm_probe.c" \
    gf-levels.csv gf-surface.csv gf-isolation.csv \
    gf-stage-levels.csv gf-stage-surface.csv gf-stage-consistency.csv \
    gf-deep-levels.csv gf-deep-surface.csv gf-deep-consistency.csv \
    gf-shallow-levels.csv gf-shallow-surface.csv gf-shallow-consistency.csv \
    gf-pow-probe.txt \
  | sed -e "s#${source_root}#<wrf-4.6.1-checkout>#" \
        -e "s#${script_dir%/*/*}#<oracle-workspace>#" \
  > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
gcc --version | head -1 >> compiler.txt
echo "oracle built in ${build_dir}"
