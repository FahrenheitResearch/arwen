#!/usr/bin/env bash
# Build and run the gwd_opt = 1 (KIM orographic drag) column oracle from
# byte-unmodified WRF v4.6.1.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# WRF_SOURCE_ROOT is a WRF v4.6.1 checkout (commit d66e442f...).  Compiled,
# all byte-unmodified and pinned by SOURCES.sha256:
#   phys/ccpp_kind_types.F           (-DRWORDSIZE=4, WRF's default build:
#                                     kind_phys is real(4))
#   share/module_model_constants.F   (CP, G, R_D, R_V, EP_1 the driver passes)
#   phys/physics_mmm/bl_gwdo.F90     bl_gwdo_run, the scheme body
#   phys/module_bl_gwdo.F            gwdo, WRF's wrapper
#
# Writes into BUILD_DIR, per vertical grid G in (L50, L90):
#   gwdo-kim-inputs-G.bin / .npz     the columns (make_columns.py)
#   gwdo-kim-wrf-O0-G.bin            reference of record: -O0, scalar libm,
#                                    no FMA
#   gwdo-kim-wrf-O2vec-G.bin         WRF's stock GNU physics flags
# and libmvec-report.txt, oracle-sha256sums.txt, compiler.txt, wrf-commit.txt,
# and the CPU half of the wind-sector sweep (libsector_sweep.so, see
# sector_sweep.py).
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python=${PYTHON:-python3}

pinned_commit="d66e442fccc04111067e29274c9f9eaccc3cef28"
head_commit=$(git -C "${source_root}" rev-parse HEAD)
if [[ "${head_commit}" != "${pinned_commit}" ]]; then
    echo "WRF tree is at ${head_commit}, not pinned ${pinned_commit}" >&2
    exit 3
fi
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}"
cd "${build_dir}"
rm -f ./*.o ./*.mod ./*.so run_gwdo_kim_* gwdo-kim-*.bin gwdo-kim-*.npz

"${python}" "${script_dir}/make_columns.py" "${build_dir}"

ff=(-cpp -ffree-form -ffree-line-length-none)
# Reference of record.  -O0 keeps every libm call scalar (atan2f, powf), and
# baseline x86-64 has no FMA instruction, so nothing is contracted.
o0=(-O0 -ffp-contract=off "${ff[@]}")
# WRF's stock GNU physics flags (arch/configure.defaults), which the shipped
# wrf.exe uses; -fconvert/-frecord-marker only touch Fortran record I/O.
o2=(-O2 -ftree-vectorize -funroll-loops -w "${ff[@]}" -fconvert=big-endian
    -frecord-marker=4)

for v in O0 O2vec; do
    mkdir -p "mod_${v}"
    if [[ ${v} == O0 ]]; then flags=("${o0[@]}"); else flags=("${o2[@]}"); fi
    gfortran -c "${flags[@]}" -J "mod_${v}" -DRWORDSIZE=4 \
        -o "ccpp_kind_types_${v}.o" "${source_root}/phys/ccpp_kind_types.F"
    gfortran -c "${flags[@]}" -J "mod_${v}" -DEM_CORE=1 \
        -o "module_model_constants_${v}.o" \
        "${source_root}/share/module_model_constants.F"
    gfortran -c "${flags[@]}" -J "mod_${v}" -o "bl_gwdo_${v}.o" \
        "${source_root}/phys/physics_mmm/bl_gwdo.F90"
    gfortran -c "${flags[@]}" -J "mod_${v}" -o "module_bl_gwdo_${v}.o" \
        "${source_root}/phys/module_bl_gwdo.F"
    # The driver itself is always -O0: it only copies words.
    gfortran -c "${o0[@]}" -J "mod_${v}" -o "run_gwdo_kim_${v}.o" \
        "${script_dir}/run_gwdo_kim.F90"
    gfortran -o "run_gwdo_kim_${v}" "run_gwdo_kim_${v}.o" \
        "module_bl_gwdo_${v}.o" "bl_gwdo_${v}.o" \
        "module_model_constants_${v}.o" "ccpp_kind_types_${v}.o"
done

# --- libmvec and FMA receipts --------------------------------------------------
{
    echo "# gfortran: $(gfortran --version | sed -n 1p)"
    echo "# glibc:    $(ldd --version | sed -n 1p)"
    for v in O0 O2vec; do
        for obj in bl_gwdo module_bl_gwdo; do
            echo "# ${obj}_${v}.o libm imports:"
            nm -u "${obj}_${v}.o" | sed 's/^ *//' \
                | grep -E 'atan2f|powf|expf|logf|sqrtf|_ZGV' || true
            echo "# ${obj}_${v}.o FMA instructions: $(objdump -d \
                "${obj}_${v}.o" | grep -Ec 'vfn?m(add|sub)' || true)"
        done
    done
} > libmvec-report.txt
for obj in bl_gwdo_O0.o module_bl_gwdo_O0.o; do
    if nm -u "${obj}" | grep -q '_ZGV'; then
        echo "-O0 object ${obj} pulled in a libmvec vector symbol" >&2
        exit 4
    fi
    if objdump -d "${obj}" | grep -Eq 'vfn?m(add|sub)'; then
        echo "-O0 object ${obj} contains an FMA instruction" >&2
        exit 4
    fi
done
# positive control: the _ZGV grep can see a vector symbol on this toolchain
cat > libmvec_positive_control.f90 <<'EOF'
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
    libmvec_positive_control.f90
if ! nm -u libmvec_positive_control.o | grep -q '_ZGV'; then
    echo "libmvec positive control produced no _ZGV symbol; the guard is" \
         "vacuous on this toolchain" >&2
    exit 5
fi
echo "# positive control (plain expf loop at -Ofast): $(nm -u \
    libmvec_positive_control.o | grep -c '_ZGV') _ZGV import(s)" \
    >> libmvec-report.txt

# --- run -------------------------------------------------------------------------
for g in L50 L90; do
    for v in O0 O2vec; do
        "./run_gwdo_kim_${v}" "gwdo-kim-inputs-${g}.bin" \
            "gwdo-kim-wrf-${v}-${g}.bin"
    done
    if cmp -s "gwdo-kim-wrf-O0-${g}.bin" "gwdo-kim-wrf-O2vec-${g}.bin"; then
        echo "${g}: stock-flags build byte-identical to the -O0 reference" \
            | tee -a libmvec-report.txt
    else
        echo "${g}: stock-flags build DIFFERS from the -O0 reference" \
            | tee -a libmvec-report.txt
    fi
done

# --- CPU half of the wind-sector sweep ---------------------------------------------
gcc -O0 -ffp-contract=off -fopenmp -fPIC -shared -o libsector_sweep.so \
    "${script_dir}/sector_sweep.c" -lm
if nm -D -u libsector_sweep.so | grep -q '_ZGV'; then
    echo "libsector_sweep.so pulled in a libmvec vector symbol" >&2
    exit 4
fi
echo "# libsector_sweep.so libm imports: $(nm -D -u libsector_sweep.so \
    | grep -Eo 'atan2?f[@A-Z_.0-9]*' | sort -u | paste -sd ' ')" \
    >> libmvec-report.txt

# --- receipts --------------------------------------------------------------------
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | awk '{print $2}' | xargs sha256sum ) > oracle-sha256sums.txt
( cd "${script_dir}" && sha256sum build.sh make_columns.py run_gwdo_kim.F90 \
    sector_sweep.c sector_sweep.py compare_gpu.py branch_mutations.py ) \
    >> oracle-sha256sums.txt
sha256sum gwdo-kim-inputs-*.bin gwdo-kim-inputs-*.npz gwdo-kim-wrf-*.bin \
    >> oracle-sha256sums.txt
gfortran --version | sed -n 1p > compiler.txt
ldd --version | sed -n 1p >> compiler.txt
echo "${pinned_commit}" > wrf-commit.txt
echo "gwdo-kim oracle built in ${build_dir}"
