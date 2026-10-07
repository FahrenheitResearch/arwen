#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build.sh BUILD_DIR TABLE_DIR'; exit 2; }
script=$(cd "$(dirname "$0")" && pwd)
build=$(realpath -m "$1")
tables=$(realpath "$2")
(cd "$script/source" && sha256sum -c ../SOURCES.sha256)
(cd "$script" && sha256sum -c HARNESS.sha256)
(cd "$tables" && sha256sum -c "$script/TABLES.sha256")
[[ $(gfortran -dumpfullversion) == 15.2.0 ]] || { echo 'requires gfortran 15.2.0'; exit 3; }
gfortran --version | head -1
ldd --version | head -1
mkdir -p "$build"
cat > "$build/control.f90" <<'EOF'
subroutine probe(n,a,b)
integer :: n,i
real :: a(n),b(n)
do i=1,n
 b(i)=exp(a(i))
enddo
end subroutine
EOF
gfortran -c -O3 -ffast-math -ftree-vectorize -march=x86-64-v3 "$build/control.f90" -o "$build/control.o"
nm -u "$build/control.o" | grep '_ZGV' || { echo 'libmvec positive control failed'; exit 4; }
for precision in sp dp; do
 mkdir -p "$build/$precision"
 cd "$build/$precision"
 flags=(-O0 -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -ffp-contract=off -fcheck=bounds)
 [[ $precision == sp ]] && flags+=(-DRTE_USE_SP)
 for name in mo_rte_kind mo_rte_config mo_rte_util_array_validation mo_optical_props_kernels mo_optical_props mo_aerosol_optics_rrtmgp_merra; do
  gfortran "${flags[@]}" -c "$script/source/$name.F90"
 done
 gfortran "${flags[@]}" -c "$script/../gocart/oracle_io.F90"
 gfortran "${flags[@]}" $(nf-config --fflags) -c "$script/run_rrtmgp_aerosol.F90"
 if nm -u ./*.o | grep '_ZGV'; then echo '-O0 object calls libmvec'; exit 4; fi
 gfortran -O0 ./*.o $(nf-config --flibs) -o run_rrtmgp_aerosol
 mkdir -p fixtures
 ./run_rrtmgp_aerosol "$build/$precision/fixtures" "$tables"
done
