#!/bin/sh
# Build the WRF v4.6.1 Noah-MP VEGE_FLUX bitwise oracle and emit its fixture.
#
# Usage:  build_vegeflux.sh <wrf-tree> <workdir> [optlevel] [leaf]
#
#   <wrf-tree>  pinned WRF checkout, e.g. /home/drew/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#               sha256(phys/module_sf_noahmplsm.F) =
#                 bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "wrf"   (default) reproduces WRF's own gfortran FCOPTIM
#               "noopt" builds at -O0.  Two roles: it is the level at which the
#                       visibility patch is provably code-neutral (nothing is
#                       inlined, so all 85 module procedures exist in both
#                       objects), and it is the negative control that the
#                       fixture does not depend on the optimiser.
#   [leaf]      esat | ragrb | sfcdif1 | stomata | vegeflux | all (default all)
#
# The only change made to phys/module_sf_noahmplsm.F is the accessibility lift
# performed by visibility_patch_leaves.py, which self-checks the pristine
# sha256 and proves the diff is nothing but private:: -> public::.
set -eu

WRF_TREE=$1
WORK=$2
OPTLEVEL=${3:-wrf}
LEAF=${4:-all}

HERE=$(cd "$(dirname "$0")" && pwd)

case "$OPTLEVEL" in
  wrf)   FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
  noopt) FCOPTIM="-O0" ;;
  *)     echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac

# WRF arch/configure.defaults, "Linux x86_64, gfortran" serial block:
#   FCOPTIM         = -O2 -ftree-vectorize -funroll-loops
#   FORMAT_FREE     = -ffree-form -ffree-line-length-none
#   BYTESWAPIO      = -fconvert=big-endian -frecord-marker=4
#   FCBASEOPTS_NO_G = -w $(FORMAT_FREE) $(BYTESWAPIO) $(FCCOMPAT)
# No -ffast-math and no -march: gfortran may neither reassociate nor contract,
# so the FP operation sequence is fixed by the source.
FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4"

mkdir -p "$WORK"
cd "$WORK"

cp "$WRF_TREE/phys/module_sf_noahmplsm.F" ./pristine.F
cp "$WRF_TREE/phys/module_sf_gecros.F"    ./module_sf_gecros.F

python3 "$HERE/visibility_patch_leaves.py" ./pristine.F \
        --out ./module_sf_noahmplsm_public.F \
        --check \
        --require-symbol VEGE_FLUX \
        --require-symbol SFCDIF1 \
        --require-symbol RAGRB \
        --require-symbol STOMATA \
        --require-symbol ESAT

cat > wrf_stubs.f90 <<'STUB'
! Minimal stand-ins for the WRF frame routines the physics module calls.
! VEGE_FLUX reaches wrf_message/wrf_error_fatal only on the HCAN <= ZPD abort,
! which no fixture case takes; they exist so the module links.
subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') trim(msg)
end subroutine wrf_message

subroutine wrf_error_fatal(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'wrf_error_fatal: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal

subroutine wrf_error_fatal3(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') 'wrf_error_fatal3: '//trim(msg)
  stop 1
end subroutine wrf_error_fatal3

subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
STUB

# ---------------------------------------------------------------------------
# 1. object-code evidence: pristine vs patched at the requested flags
# ---------------------------------------------------------------------------
# Both sides are compiled from a file with the *same* name in a directory with
# the same relative path.  gfortran embeds the source file name in .rodata for
# runtime diagnostics, so compiling "pristine.F" against "patched.F" would make
# .rodata differ by the filename alone and mask what the comparison is for.
rm -rf objcmp && mkdir -p objcmp/pristine objcmp/patched
cp ./pristine.F                     objcmp/pristine/module_sf_noahmplsm.F
cp ./module_sf_noahmplsm_public.F   objcmp/patched/module_sf_noahmplsm.F
for side in pristine patched; do
  cp module_sf_gecros.F "objcmp/$side/"
  ( cd "objcmp/$side" \
    && gfortran $FCOPTIM $FCBASE -c module_sf_gecros.F -o gecros.o \
    && gfortran $FCOPTIM $FCBASE -c module_sf_noahmplsm.F -o module.o )
done

echo "=== object-code comparison ($OPTLEVEL) ==="
if [ "$OPTLEVEL" = noopt ]; then
  python3 "$HERE/compare_object_code_vegeflux.py" \
          objcmp/pristine/module.o objcmp/patched/module.o --expect-total
else
  python3 "$HERE/compare_object_code_vegeflux.py" \
          objcmp/pristine/module.o objcmp/patched/module.o --allow-inline-diff || true
fi

# ---------------------------------------------------------------------------
# 2. build the fixture generator against the patched module
# ---------------------------------------------------------------------------
gfortran $FCOPTIM $FCBASE -c module_sf_gecros.F            -o gecros.o
gfortran $FCOPTIM $FCBASE -c module_sf_noahmplsm_public.F  -o noahmp.o
gfortran $FCOPTIM $FCBASE -c wrf_stubs.f90                 -o stubs.o
gfortran $FCOPTIM $FCBASE -c "$HERE/run_vegeflux.F90"      -o run_vegeflux.o
gfortran -o run_vegeflux run_vegeflux.o noahmp.o gecros.o stubs.o

./run_vegeflux "$LEAF" > fixture.csv
echo "=== fixture rows: $(( $(wc -l < fixture.csv) - 1 )) ==="
