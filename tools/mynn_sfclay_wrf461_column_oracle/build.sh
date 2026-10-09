#!/usr/bin/env bash
# Build and run the WRF v4.6.1 MYNN surface-layer column oracle.
#
#   build.sh /path/to/WRF-4.6.1 /new/output-directory
#
# Compiles the pinned, unmodified phys/module_sf_mynn.F four ways and runs
# the same column driver against each:
#   o2               -O2 (the flags of tools/mynn_wrf461_oracle/build.sh):
#                    the reference of record.  Scalar libm only: the build
#                    refuses an object that binds a libmvec (_ZGV*) entry.
#   o0               -O0, to show the answer does not depend on the optimiser.
#   wrfstock         WRF's own configure.defaults GNU flags for the module
#                    (-O2 -ftree-vectorize -funroll-loops ...), the driver
#                    plain.  Ubuntu's gfortran pre-includes the vector math
#                    header, so vectorized loops call libmvec.
#   wrfstock_scalar  the same stock flags plus -fno-tree-vectorize: shows
#                    whether vectorization (libmvec) is the whole difference.
# It records whether the outputs are byte-identical to o2, the libmvec entry
# points each object binds, and (stock_position.py) whether a column's answer
# changes when the column only moves to another loop index.  No GPU is used.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/WRF-4.6.1 /new/output-directory" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
wrf_root=$(realpath "$1")
out=$(realpath -m "$2")
source_file="$wrf_root/phys/module_sf_mynn.F"
expected_wrf_commit=d66e442fccc04111067e29274c9f9eaccc3cef28
# module_sf_mynn.F as the LF checkout stores it, and the same bytes with
# CRLF line ends (the form tools/mynn_wrf461_oracle/README.md pins).
expected_lf=$(grep -m1 '^lf ' "$script_dir/SOURCE_SHA256" | cut -d' ' -f2)
expected_crlf=$(grep -m1 '^crlf ' "$script_dir/SOURCE_SHA256" | cut -d' ' -f2)

if [ "$(git -C "$wrf_root" rev-parse HEAD)" != "$expected_wrf_commit" ]; then
  echo "WRF checkout is not the pinned v4.6.1 commit" >&2
  exit 2
fi
got_lf=$(sha256sum "$source_file" | cut -d' ' -f1)
got_crlf=$(sed 's/$/\r/' "$source_file" | sha256sum | cut -d' ' -f1)
if [ "$got_lf" != "$expected_lf" ] && [ "$got_crlf" != "$expected_crlf" ]; then
  echo "module_sf_mynn.F bytes differ from the pinned source" >&2
  exit 2
fi
if [ -e "$out" ]; then
  echo "refusing to reuse output directory: $out" >&2
  exit 2
fi
mkdir -p "$out"
cd "$out"

{
  # sed -n 1p reads to the end: head would SIGPIPE the writer, and under
  # pipefail that aborts the build.
  gfortran --version | sed -n 1p
  ldd --version | sed -n 1p
  grep -m1 'model name' /proc/cpuinfo || true
  grep -m1 -o -w 'fma' /proc/cpuinfo || echo 'no fma flag'
} > TOOLCHAIN.txt

python3 "$script_dir/columns.py" "$out/inputs" > inputs.json

common=(-ffree-form -ffree-line-length-none)
checks=(-fcheck=all -ffpe-trap=invalid,zero,overflow)
stock=(-O2 -ftree-vectorize -funroll-loops -w -ffree-form \
       -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 \
       -fallow-argument-mismatch -fallow-invalid-boz)

build_one() {  # name, module flags..., -- driver flags...
  local name=$1; shift
  local mflags=() dflags=()
  while [ "$1" != "--" ]; do mflags+=("$1"); shift; done; shift
  dflags=("$@")
  mkdir -p "$name"
  (
    cd "$name"
    gfortran "${mflags[@]}" -c "$script_dir/../mynn_wrf461_oracle/stub_wrf.F90"
    gfortran "${mflags[@]}" -c "$source_file"
    gfortran "${dflags[@]}" -c "$script_dir/run_columns.F90"
    gfortran -o run_columns stub_wrf.o module_sf_mynn.o run_columns.o
    nm -u module_sf_mynn.o | awk '{print $2}' | sort > module_undefined.txt
    ./run_columns ../inputs/columns.bin ../inputs/configs.txt \
        ../oracle-"$name".bin
  )
  echo "$name libmvec: $(grep -c '^_ZGV' "$name/module_undefined.txt" || true)"
}

build_one o2 -O2 "${common[@]}" "${checks[@]}" -- \
    -O2 "${common[@]}" "${checks[@]}" | tee -a BUILD-SUMMARY.txt
if grep -q '^_ZGV' o2/module_undefined.txt; then
  echo "the -O2 reference binds libmvec; refusing it" >&2
  exit 3
fi
build_one o0 -O0 "${common[@]}" "${checks[@]}" -- \
    -O0 "${common[@]}" "${checks[@]}" | tee -a BUILD-SUMMARY.txt
build_one wrfstock "${stock[@]}" -- \
    -O2 "${common[@]}" | tee -a BUILD-SUMMARY.txt
build_one wrfstock_scalar "${stock[@]}" -fno-tree-vectorize -- \
    -O2 "${common[@]}" | tee -a BUILD-SUMMARY.txt

for name in o2 o0 wrfstock wrfstock_scalar; do
  echo "$name binds: $(grep '^_ZGV' "$name/module_undefined.txt" | tr '\n' ' ')" \
    | tee -a BUILD-SUMMARY.txt
done

for other in o0 wrfstock wrfstock_scalar; do
  if cmp -s oracle-o2.bin "oracle-$other.bin"; then
    echo "o2 vs $other: byte-identical" | tee -a BUILD-SUMMARY.txt
  else
    echo "o2 vs $other: DIFFERENT" | tee -a BUILD-SUMMARY.txt
  fi
done

sha256sum "$source_file" "$script_dir/run_columns.F90" \
    "$script_dir/columns.py" inputs/columns.bin inputs/configs.txt \
    oracle-o2.bin oracle-o0.bin oracle-wrfstock.bin \
    oracle-wrfstock_scalar.bin | tee SHA256SUMS

# Position dependence: the same columns moved 1-3 loop indices.  A nonzero
# count means that build's answer for a column is not a function of the
# column alone.
for name in o2 wrfstock wrfstock_scalar; do
  echo "position test, $name:" | tee -a POSITION.txt
  PYTHONPATH="$script_dir/../..${PYTHONPATH:+:$PYTHONPATH}" python3 -m \
      tools.mynn_sfclay_wrf461_column_oracle.stock_position \
      "$name/run_columns" inputs "position-$name" | tee -a POSITION.txt \
      || true
done
