#!/usr/bin/env bash
# Build the gsl_wrf39 MYNN surface-layer column oracle and run every deck.
#   ./build.sh /path/to/fork/module_sf_mynn.F /new/build-directory
# The source is NOAA-EMC/HRRR tag v4.1.21,
# sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_sf_mynn.F, pinned by SHA-256:
#   curl -sSfLO https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21/sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_sf_mynn.F
# Two builds, -O0 and -O2, no -march (x86-64 SSE2: no FMA contraction);
# their outputs must be byte-identical or the script fails.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 /path/to/module_sf_mynn.F /new/build-directory" >&2; exit 2
fi
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
src=$(realpath "$1")
out=$(realpath -m "$2")
expected=47bc9943f505c853e7dbb09b24d61889b4fc85ccc72a53b60903f6f46458a214
[ "$(sha256sum "$src" | cut -d ' ' -f 1)" = "$expected" ] || {
  echo "module_sf_mynn.F bytes differ from the pinned fork source" >&2; exit 2; }
[ -e "$out" ] && { echo "refusing to reuse build directory $out" >&2; exit 2; }
mkdir -p "$out/decks"
cd "$out"
{ gfortran --version | sed -n 1p; ldd --version | sed -n 1p;
  grep -m1 'model name' /proc/cpuinfo;
  grep -qw fma /proc/cpuinfo && echo "cpu fma: yes" || echo "cpu fma: no"; } \
  > TOOLCHAIN.txt
python3 "$here/gsl_wrf39_columns.py" decks
for opt in O0 O2; do
  mkdir -p "$opt"
  ( cd "$opt"
    flags=(-$opt -ffree-form -ffree-line-length-none)
    gfortran "${flags[@]}" -c "$here/../mynn_gsl_wrf39_oracle/stub_wrf.F90"
    cp "$src" module_sf_mynn.F
    gfortran "${flags[@]}" -c module_sf_mynn.F
    gfortran "${flags[@]}" -c "$here/run_columns.F90"
    gfortran "${flags[@]}" -o run_columns stub_wrf.o module_sf_mynn.o \
      run_columns.o
    # Which libm entry points the fork object calls (the GPU side must use
    # the same functions; sqrt is an instruction, not a call).
    nm -u module_sf_mynn.o | grep -Ei ' (logf|expf|powf|atanf|sqrtf|log|exp|pow|atan)$' \
      | sort > LIBM_CALLS.txt || true
    for deck in "$out"/decks/*.txt; do
      ./run_columns "$deck" "$(basename "$deck" .txt).csv"
    done )
done
for f in O0/*.csv; do
  cmp "$f" "O2/$(basename "$f")" || { echo "O0 and O2 differ: $f" >&2; exit 1; }
done
echo "O0 and O2 outputs byte-identical" | tee -a TOOLCHAIN.txt
cat O2/LIBM_CALLS.txt >> TOOLCHAIN.txt
sha256sum "$src" "$here"/gsl_wrf39_columns.py "$here"/run_columns.F90 decks/*.txt O2/*.csv \
  | sed -E 's@  .*/@  @' > SHA256SUMS
cat TOOLCHAIN.txt SHA256SUMS
