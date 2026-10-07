#!/usr/bin/env bash
# Isolated UPP statement oracle. No WRF/GSL modules or device use.
# Downloads only the two pinned public UPP files; compiles the copied
# MDLFLD.f:2442 statement in the driver and writes with oracle_io.F90.
set -euo pipefail
script_dir=$(cd "$(dirname "$0")" && pwd)
build_dir=$(realpath -m "$1")
mkdir -p "$build_dir/upp"
cd "$build_dir"
revision=1296eebb295251d0fe1cc697f47058c62d887977
for name in MDLFLD.f params.F; do
    curl -fsSL "https://raw.githubusercontent.com/NOAA-EMC/UPP/$revision/sorc/ncep_post.fd/$name" -o "upp/$name"
done
sha256sum -c "$script_dir/SOURCES-upp.sha256"
# Assert that the source statement was copied exactly, including spaces.
python3 - "$script_dir/run_source_conversion.F90" <<'PY'
import pathlib, sys
source = pathlib.Path('upp/MDLFLD.f').read_bytes().splitlines()[2441]
assert source in pathlib.Path(sys.argv[1]).read_bytes().splitlines()
assert b'real, parameter :: RD=287.04' in pathlib.Path('upp/params.F').read_bytes()
PY
gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffp-contract=off "$script_dir/oracle_io.F90"
gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffp-contract=off -I . "$script_dir/run_source_conversion.F90"
nm -u oracle_io.o run_source_conversion.o > undefined.txt
if grep -E '_ZGV' undefined.txt; then
    echo 'vector libm would replace scalar oracle arithmetic' >&2
    exit 1
fi
gfortran -c -Ofast -ftree-vectorize "$script_dir/libmvec_positive_control.F90" -o positive.o
nm -u positive.o > positive.txt
grep -E '_ZGV' positive.txt
gfortran oracle_io.o run_source_conversion.o -o run_source_conversion
./run_source_conversion fixtures
gfortran -c -O0 -cpp -DPOISON -DRWORDSIZE=4 -ffp-contract=off -I . "$script_dir/run_source_conversion.F90" -o poison.o
gfortran oracle_io.o poison.o -o poison_source_conversion
./poison_source_conversion poison
diff -r fixtures poison
gfortran --version | head -1 > compiler.txt
ldd --version | head -1 > glibc.txt
sha256sum "$script_dir/run_source_conversion.F90" "$script_dir/oracle_io.F90" > driver-sha256sums.txt
(cd fixtures; find . -type f -print0 | sort -z | xargs -0 sha256sum) > oracle-sha256sums.txt
