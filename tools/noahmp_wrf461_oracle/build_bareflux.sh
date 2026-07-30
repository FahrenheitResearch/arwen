#!/bin/sh
# Build the WRF v4.6.1 Noah-MP BARE_FLUX bitwise oracle.
#
# Usage:  build_bareflux.sh <wrf-tree> <workdir> [optlevel]
#
#   <wrf-tree>  pinned WRF checkout, e.g. $HOME/wrf-stock-v461-gate-20260721
#               commit d66e442fccc04111067e29274c9f9eaccc3cef28
#   <workdir>   scratch directory (created if absent)
#   [optlevel]  "wrf" (default) reproduces WRF's own gfortran FCOPTIM
#               "noopt" builds at -O0; used as a negative control that the
#                       fixture does not depend on the optimiser.
#   [mode]      "patched" (default) links the visibility-patched object.
#               "pristine" links the PRISTINE object instead, reaching
#                       BARE_FLUX by globalising its symbol with objcopy.
#                       gfortran gives a `private` module procedure LOCAL
#                       linkage ("t" in nm), which is the only reason the
#                       patch is needed at all.  objcopy --globalize-symbol
#                       flips that one symbol-table binding and touches no
#                       instruction, so the resulting binary executes the
#                       unmodified module's own machine code.  This is the
#                       behavioural proof that the visibility patch does not
#                       change the answer -- stronger than diffing object
#                       code, because at any optimisation level accessibility
#                       legitimately changes what GCC may assume about a
#                       procedure with no external callers.
#
# The only change made to phys/module_sf_noahmplsm.F is the accessibility
# lift performed by visibility_patch_leaves.py, which self-checks the
# pristine sha256 and proves the diff is nothing but private:: -> public::.
set -eu

WRF_TREE=$1
WORK=$2
OPTLEVEL=${3:-wrf}
MODE=${4:-patched}

HERE=$(cd "$(dirname "$0")" && pwd)

case "$OPTLEVEL" in
  wrf)   FCOPTIM="-O2 -ftree-vectorize -funroll-loops" ;;
  noopt) FCOPTIM="-O0" ;;
  *)     echo "unknown optlevel: $OPTLEVEL" >&2; exit 2 ;;
esac

# WRF arch/configure.defaults, "Linux x86_64, gfortran" block:
#   FORMAT_FREE = -ffree-form -ffree-line-length-none
#   BYTESWAPIO  = -fconvert=big-endian -frecord-marker=4
#   FCBASEOPTS_NO_G = -w $(FORMAT_FREE) $(BYTESWAPIO) $(FCCOMPAT)
FCBASE="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4"

mkdir -p "$WORK"
cd "$WORK"

cp "$WRF_TREE/phys/module_sf_noahmplsm.F" ./pristine.F
cp "$WRF_TREE/phys/module_sf_gecros.F"    ./module_sf_gecros.F

python3 "$HERE/visibility_patch_leaves.py" ./pristine.F \
        --out ./module_sf_noahmplsm_public.F \
        --check \
        --require-symbol BARE_FLUX \
        --require-symbol SFCDIF1 \
        --require-symbol ESAT

cat > wrf_stubs.f90 <<'STUB'
! Minimal replacements for the WRF frame routines the physics module calls.
! BARE_FLUX reaches none of them on any path exercised by the oracle; they
! exist only so the module links.
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

subroutine wrf_message(msg)
  character(len=*), intent(in) :: msg
  write(0,'(A)') trim(msg)
end subroutine wrf_message

subroutine wrf_debug(lvl, msg)
  integer, intent(in) :: lvl
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
STUB

gfortran -c $FCBASE $FCOPTIM module_sf_gecros.F           -o module_sf_gecros.o
# The patched object is always built: its .mod is what the driver compiles
# against, whichever object is finally linked.
gfortran -c $FCBASE $FCOPTIM module_sf_noahmplsm_public.F -o module_sf_noahmplsm.o
gfortran -c -w -O2 wrf_stubs.f90 -o wrf_stubs.o
gfortran -c $FCBASE $FCOPTIM "$HERE/run_bareflux.F90" -o run_bareflux.o

LINK_OBJ=module_sf_noahmplsm.o
if [ "$MODE" = "pristine" ]; then
  gfortran -c $FCBASE $FCOPTIM pristine.F -o module_sf_noahmplsm_pristine.o
  # -O2 clones private procedures (__..._MOD_esat.isra.0), so the plain
  # symbol only exists at -O0.  That is why pristine mode is -O0 only; the
  # optimiser gap is closed separately by 'noopt patched' == 'wrf patched'.
  if [ "$OPTLEVEL" != "noopt" ]; then
    echo "pristine mode requires optlevel 'noopt' (see comment above)" >&2
    exit 4
  fi
  objcopy --globalize-symbol=__module_sf_noahmplsm_MOD_bare_flux           --globalize-symbol=__module_sf_noahmplsm_MOD_esat           module_sf_noahmplsm_pristine.o module_sf_noahmplsm_pristine_glob.o
  # Prove objcopy changed the symbol table and not one instruction byte.
  objdump -s -j .text module_sf_noahmplsm_pristine.o      | tail -n +3 > .text_before
  objdump -s -j .text module_sf_noahmplsm_pristine_glob.o | tail -n +3 > .text_after
  if ! cmp -s .text_before .text_after; then
    echo "objcopy changed .text -- refusing to proceed" >&2
    exit 3
  fi
  echo "objcopy globalised BARE_FLUX; .text is byte-identical"
  LINK_OBJ=module_sf_noahmplsm_pristine_glob.o
fi

gfortran -o run_bareflux run_bareflux.o "$LINK_OBJ" module_sf_gecros.o wrf_stubs.o -lm

echo "built $WORK/run_bareflux  (optlevel=$OPTLEVEL, mode=$MODE, FCOPTIM='$FCOPTIM')"
