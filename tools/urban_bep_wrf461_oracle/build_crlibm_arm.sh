#!/usr/bin/env bash
# The same v4.6.1 BEP oracle, linked with correctly rounded acosf, asinf,
# atanf and tanf from CORE-MATH instead of the build host's C library.
#
#   bash tools/urban_bep_wrf461_oracle/build_crlibm_arm.sh \
#        WRF_SOURCE_ROOT CORE_MATH_REPO BUILD_DIR
#
# CORE_MATH_REPO is a clone of https://gitlab.inria.fr/core-math/core-math.git.
# Each function is taken at a pinned commit: asinf bc385c2, acosf 56dd347,
# atanf a8066a5, tanf 59d21d7.  Nothing from CORE-MATH enters the repository
# or the package; it only builds this reference arm on the oracle host.
#
# Why: WRF's BEP calls acos/asin/atan/tan on REAL arguments, which gfortran
# lowers to the C library's acosf/asinf/atanf/tanf.  Whether those are
# correctly rounded depends on the C library version of the host that builds
# WRF (Ubuntu 24.04's glibc 2.39: not; the v4.7.1 oracle host's glibc 2.43:
# yes).  This arm pins the WRF side to correctly rounded results so the BEP
# code itself is graded, and build.sh's plain arm records the host-library
# distance.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: build_crlibm_arm.sh WRF_SOURCE_ROOT CORE_MATH_REPO BUILD_DIR" >&2
    exit 2
fi
src=$(realpath "$1"); cm=$(realpath "$2"); build=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "${build}/crlibm"
cd "${build}/crlibm"
for pair in asin:bc385c2 acos:56dd347 atan:a8066a5 tan:59d21d7; do
    fn=${pair%%:*}; commit=${pair##*:}
    git -C "${cm}" show "${commit}:src/binary32/${fn}/${fn}f.c" > "${fn}f.c"
    gcc -c -O2 -fno-builtin -fno-fast-math -o "cr_${fn}f.o" "${fn}f.c"
done
cat > crlibm_shim.c <<'SHIM'
/* The four C-library entry points gfortran's REAL acos/asin/atan/tan call,
   answered by CORE-MATH's correctly rounded routines. */
float cr_acosf(float); float cr_asinf(float);
float cr_atanf(float); float cr_tanf(float);
float acosf(float x) { return cr_acosf(x); }
float asinf(float x) { return cr_asinf(x); }
float atanf(float x) { return cr_atanf(x); }
float tanf(float x) { return cr_tanf(x); }
SHIM
gcc -c -O0 -fno-builtin -o crlibm_shim.o crlibm_shim.c
sha256sum asinf.c acosf.c atanf.c tanf.c crlibm_shim.c > crlibm-sha256.txt
ORACLE_LINK_EXTRA="${build}/crlibm/crlibm_shim.o ${build}/crlibm/cr_asinf.o ${build}/crlibm/cr_acosf.o ${build}/crlibm/cr_atanf.o ${build}/crlibm/cr_tanf.o" \
    bash "${here}/build.sh" "${src}" "${build}"
cp "${build}/crlibm/crlibm-sha256.txt" "${build}/"
