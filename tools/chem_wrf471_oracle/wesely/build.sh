#!/usr/bin/env bash
# Pinned unmodified WRF v4.7.1 chem/module_dep_simple.F.
set -euo pipefail
[[ $# == 2 ]] || { echo 'usage: build.sh WRF_SOURCE_ROOT BUILD_DIR' >&2; exit 2; }
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$build_dir/share" "$build_dir/frame" "$build_dir/chem"
cp "$source_root/chem/module_dep_simple.F" "$build_dir/chem/"
for rel in share/module_model_constants.F frame/module_wrf_error.F; do
 curl --fail --silent --show-error "https://raw.githubusercontent.com/wrf-model/WRF/v4.7.1/$rel" -o "$build_dir/$rel"
done
cd "$build_dir"
sha256sum -c "$script_dir/SOURCES.sha256"
flags='-O0 -fno-fast-math -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4'
fc=/usr/bin/gfortran
$fc $flags -c "$script_dir/stub_wesely.F90"
$fc $flags -c share/module_model_constants.F
$fc $flags -c frame/module_wrf_error.F
$fc $flags -c chem/module_dep_simple.F
$fc $flags -c "$script_dir/oracle_io.F90" "$script_dir/run_wesely.F90"
for obj in stub_wesely module_model_constants module_wrf_error module_dep_simple oracle_io run_wesely; do
 nm -u "$obj.o" > "undefined-$obj.txt"
 if grep -q '_ZGV' "undefined-$obj.txt"; then echo "libmvec forbidden in $obj" >&2; exit 4; fi
done
$fc -Ofast -ftree-vectorize -c "$script_dir/libmvec_positive_control.F90"
nm -u libmvec_positive_control.o > undefined-control.txt
grep -q '_ZGV' undefined-control.txt || { echo 'positive control did not emit _ZGV' >&2; exit 5; }
$fc -o run_wesely stub_wesely.o module_model_constants.o module_wrf_error.o module_dep_simple.o oracle_io.o run_wesely.o
./run_wesely > run-wesely.log
{
 echo '# WRF v4.7.1 Wesely column oracle'
 $fc --version
 ldd --version | head -1
 echo "Flags: $flags"
 echo "Command: nice -n 15 bash $script_dir/build.sh $source_root $build_dir"
 cat "$script_dir/SOURCES.sha256"
 echo 'No NETCDF; non-MOZART path only. No GPU used.'
 echo 'Scalar objects have no _ZGV symbol. Positive control:'
 cat undefined-control.txt
} > fixtures/PROVENANCE.md
for directory in fixtures/*/; do cp fixtures/PROVENANCE.md "$directory/PROVENANCE.md"; done
find fixtures -type f -printf '%P %s bytes\n' > fixture-sizes.txt
