#!/usr/bin/env bash
# Build and run the NSAS (cu_physics = 96) column oracle against WRF v4.6.1.
#
#   bash build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD]
#
# THREE ARMS, ONE ANSWER.  The referee is arm "wrfobj": the harness linked
# against the module_cu_nsas.o that the scalar WRF build compiled into its
# wrf.exe (default /work/wrf/build/scalar/WRF-4.6.1, FCOPTIM = -O2
# -fno-fast-math -ffp-contract=off -fno-tree-vectorize, gfortran 13.3.0,
# glibc 2.39 libm).  Two independent compiles of the same WRF-preprocessed
# source check it: arm "o2" (the scalar FCFLAGS again, from source) and arm
# "o0" (no optimisation at all).  The script FAILS unless all three write
# byte-identical outputs, so an answer that depends on how gfortran
# scheduled the arithmetic cannot become a fixture.
#
# The scheme file has no USE and no framework dependency
# (`nm -u module_cu_nsas.o` is expf, powf and the allocator), so the harness
# links the scheme and stops.  It is pinned by digest below; the build's
# copy of it must carry the same digest, and the build's preprocessed .f90
# is what arms o2/o0 compile, so all three arms see the same text.
#
# libmvec: an -O0 object must call no _ZGV* vector routine; checked.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR [WRF_SCALAR_BUILD]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
wrf_build=$(realpath "${3:-/work/wrf/build/scalar/WRF-4.6.1}")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tree=$(cd "$here/../.." && pwd)

pinned_src=092cfc6685a7006ecf9200b0dbb9296f0e7bb6af11e46bbbd92a843a4d1b911a
src="$source_root/phys/module_cu_nsas.F"
got=$(sha256sum "$src" | cut -d' ' -f1)
[[ "$got" == "$pinned_src" ]] || { echo "module_cu_nsas.F digest $got != $pinned_src" >&2; exit 1; }
got=$(sha256sum "$wrf_build/phys/module_cu_nsas.F" | cut -d' ' -f1)
[[ "$got" == "$pinned_src" ]] || { echo "build copy digest $got != $pinned_src" >&2; exit 1; }
for f in module_cu_nsas.o module_cu_nsas.mod module_cu_nsas.f90; do
    [[ -s "$wrf_build/phys/$f" ]] || { echo "missing $wrf_build/phys/$f" >&2; exit 1; }
done

FCBASE="-w -ffree-form -ffree-line-length-none -fallow-argument-mismatch -fallow-invalid-boz"
FCOPTIM_SCALAR="-O2 -fno-fast-math -ffp-contract=off -fno-tree-vectorize"

mkdir -p "$build_dir"/{wrfobj,o2,o0}
{
    echo "gfortran: $(gfortran --version | head -1)"
    echo "glibc: $(ldd --version | head -1)"
    echo "wrf build: $wrf_build"
    echo "module_cu_nsas.F sha256: $pinned_src"
    echo "module_cu_nsas.o sha256: $(sha256sum "$wrf_build/phys/module_cu_nsas.o" | cut -d' ' -f1)"
    echo "module_cu_nsas.f90 sha256: $(sha256sum "$wrf_build/phys/module_cu_nsas.f90" | cut -d' ' -f1)"
    echo "FCOPTIM (scalar referee): $FCOPTIM_SCALAR"
} > "$build_dir/compiler.txt"

# arm wrfobj: the WRF build's own object and module file
cp "$wrf_build/phys/module_cu_nsas.o" "$wrf_build/phys/module_cu_nsas.mod" "$build_dir/wrfobj/"
(cd "$build_dir/wrfobj" && gfortran $FCOPTIM_SCALAR $FCBASE -I. -c "$here/run_nsas.F90" -o run_nsas.o \
    && gfortran -o run_nsas run_nsas.o module_cu_nsas.o -lm)

# arms o2 and o0: the build's preprocessed text, compiled here
for arm in o2 o0; do
    opt=$FCOPTIM_SCALAR
    [[ $arm == o0 ]] && opt="-O0 -ffp-contract=off"
    (cd "$build_dir/$arm" && cp "$wrf_build/phys/module_cu_nsas.f90" . \
        && gfortran $opt $FCBASE -c module_cu_nsas.f90 -o module_cu_nsas.o \
        && gfortran $opt $FCBASE -I. -c "$here/run_nsas.F90" -o run_nsas.o \
        && gfortran -o run_nsas run_nsas.o module_cu_nsas.o -lm)
done
if nm "$build_dir/o0/module_cu_nsas.o" | grep -q '_ZGV'; then
    echo "libmvec symbol in the -O0 object" >&2; exit 1
fi
nm -u "$build_dir/wrfobj/module_cu_nsas.o" | sed 's/^ *//' > "$build_dir/undefined-symbols.txt"

export PYTHONPATH="$tree${PYTHONPATH:+:$PYTHONPATH}"
py=${PYTHON:-python3}
status=0
for nz in 40 60; do
    "$py" -m gpuwm.verify.nsas_oracle --nz $nz --out "$build_dir/cases-nz$nz.bin"
    for arm in wrfobj o2 o0; do
        "$build_dir/$arm/run_nsas" "$build_dir/cases-nz$nz.bin" "$build_dir/out-nz$nz-$arm.bin"
    done
    for arm in o2 o0; do
        if cmp -s "$build_dir/out-nz$nz-wrfobj.bin" "$build_dir/out-nz$nz-$arm.bin"; then
            echo "nz=$nz: arm $arm byte-identical to wrfobj"
        else
            echo "nz=$nz: arm $arm DIFFERS from wrfobj" >&2; status=1
        fi
    done
done
sha256sum "$build_dir"/cases-nz*.bin "$build_dir"/out-nz*-wrfobj.bin > "$build_dir/oracle-sha256sums.txt"
cat "$build_dir/oracle-sha256sums.txt"
exit $status
