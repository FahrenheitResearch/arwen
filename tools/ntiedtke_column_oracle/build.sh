#!/usr/bin/env bash
# Build the WRF side of the New Tiedtke column oracle.
#
#   build.sh WRF_SOURCE_ROOT BUILD_DIR
#
# Three executables over the WRF v4.6.1 scheme files (the first two
# byte-unmodified; the third differs by one declaration, see below):
#
#   run_nt_oracle        the reference: scheme objects at -O0, no
#                        vectoriser, so every exp/log/pow is a scalar glibc
#                        call (the same convention as the stage oracle in
#                        tools/ntiedtke_wrf461_oracle).
#   run_nt_oracle_stock  the scheme objects compiled with WRF's own GNU
#                        flags from arch/configure.defaults (-O2
#                        -ftree-vectorize -funroll-loops ...), i.e. what a
#                        stock wrf.exe on this box runs.  Reported beside
#                        the reference so a difference between the two is
#                        visible rather than assumed away.
#
# The pins are tools/ntiedtke_wrf461_oracle/build.sh's, checked again here.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2; exit 2; }
src=$(realpath "$1"); out=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

declare -A pinned=(
  ["phys/physics_mmm/cu_ntiedtke.F90"]="e762101f04d4acd2d19047a92d0b7cd4e244df930f9f9ef7aabae54bfe9a9fd1"
  ["phys/module_cu_ntiedtke.F"]="447406d1550d4095e4e6f129ee74a7ec0ccdebd21383ba0cc6fe3d282ac58d2f"
  ["phys/ccpp_kind_types.F"]="a76e1e5b52fc7cd40be9ccb506fde28b1ef2486b812d518f7f1c882766d484db"
  ["phys/module_cumulus_driver.F"]="d73eb4b670599e0bbf98a3661b5dbf17413aed01139d18fa1a84161578dc31e3"
)
for rel in "${!pinned[@]}"; do
  got=$(sha256sum "$src/$rel" | awk '{print $1}')
  [[ "$got" == "${pinned[$rel]}" ]] || { echo "FATAL: $rel is $got, not pinned ${pinned[$rel]}" >&2; exit 3; }
done

stock="-O2 -ftree-vectorize -funroll-loops -w -ffree-form -ffree-line-length-none -frecord-marker=4 -fallow-argument-mismatch -fallow-invalid-boz"
ref="-O0 -ffree-form -ffree-line-length-none"

# THE THIRD BUILD, run_nt_oracle_zcbase_real: the -O0 reference with ONE
# declaration corrected.  cutypen declares its cloud-base pressure
#
#     integer,dimension(klon):: zcbase,itoppacel        (cu_ntiedtke.F90:1300)
#
# so `zcbase(jl) = paph(jl,ik) + zdp` truncates a pressure in Pa toward zero
# before the nearest-half-level test that picks the cloud base.  The name
# says real (the scheme's z-prefix convention), itoppacel beside it is a
# genuine level index, and nothing in the routine wants a whole-Pa pressure:
# this is a WRF declaration bug.  WOOF keeps zcbase real and does not copy
# it.  This build is the same pinned source with only that declaration
# moved to real(kind=kind_phys), so the comparison can separate the columns
# the bug decides from everything else.  The edit is asserted to be exactly
# that one line.
zfix="$out/zcbase-real-src"; mkdir -p "$zfix"
sed 's/^      integer,dimension(klon):: zcbase,itoppacel$/      integer,dimension(klon):: itoppacel\n      real(kind=kind_phys),dimension(klon):: zcbase/' \
    "$src/phys/physics_mmm/cu_ntiedtke.F90" > "$zfix/cu_ntiedtke.F90"
changed=$(diff "$src/phys/physics_mmm/cu_ntiedtke.F90" "$zfix/cu_ntiedtke.F90" | grep -c '^[<>]' || true)
[[ "$changed" -eq 3 ]] || { echo "FATAL: the zcbase correction did not apply as one line (diff lines: $changed)" >&2; exit 6; }
diff "$src/phys/physics_mmm/cu_ntiedtke.F90" "$zfix/cu_ntiedtke.F90" > "$out/zcbase-real.diff" || true

for flavour in ref stock zfixed; do
  d="$out/$flavour"; mkdir -p "$d"; cd "$d"
  scheme="$src/phys/physics_mmm/cu_ntiedtke.F90"
  if [[ $flavour == zfixed ]]; then flags=$ref; scheme="$zfix/cu_ntiedtke.F90"; else flags=${!flavour}; fi
  # -DRWORDSIZE=4 is not optional: without it ccpp_kind_types.F silently
  # takes its double-precision branch (see the stage oracle's README).
  gfortran -c $flags -cpp -DRWORDSIZE=4 -o ccpp_kind_types.o "$src/phys/ccpp_kind_types.F"
  gfortran -c $flags -cpp -I "$d" -o cu_ntiedtke.o "$scheme"
  gfortran -c $flags -cpp -I "$d" -o module_cu_ntiedtke.o "$src/phys/module_cu_ntiedtke.F"
  # The harness itself is always -O0: it does I/O and the fold, nothing
  # that a flag should be allowed to move.
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$d" -o run_nt_oracle.o "$here/run_nt_oracle.F90"
  exe=run_nt_oracle; [[ $flavour == stock ]] && exe=run_nt_oracle_stock
  [[ $flavour == zfixed ]] && exe=run_nt_oracle_zcbase_real
  gfortran -o "$out/$exe" ccpp_kind_types.o cu_ntiedtke.o module_cu_ntiedtke.o run_nt_oracle.o
done

# The reference must not reach libmvec: at -O0 nothing is vectorised, so a
# _ZGV* symbol would mean the reference is not the scalar-libm build it
# claims to be.  The stock build may, and the receipt says whether it does.
nvec=$(nm -u "$out/ref/cu_ntiedtke.o" "$out/ref/module_cu_ntiedtke.o" | grep -c _ZGV || true)
if [[ "$nvec" -ne 0 ]]; then
  echo "FATAL: the -O0 reference objects reference libmvec" >&2; exit 5
fi
{
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  echo "ref flags:   $ref"
  echo "stock flags: $stock"
  echo "stock libmvec symbols: $(nm -u "$out/stock/cu_ntiedtke.o" "$out/stock/module_cu_ntiedtke.o" | grep -c _ZGV || true)"
  for rel in "${!pinned[@]}"; do echo "$rel ${pinned[$rel]}"; done | sort
} > "$out/build-receipt.txt"
cat "$out/build-receipt.txt"
