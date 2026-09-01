#!/bin/sh
# Build WRF's own p3_init and dump the lookup-table arrays it fills.
# See README.md.  Needs gfortran; no GPU, no CUDA, no MPI.
set -eu

AUTHORITY_SHA256=716950a3081ec4e338c9a918d26ec80f7ee0e40b3e284283f070423237f6a3c6

if [ $# -lt 3 ]; then
    echo "usage: $0 <path/to/phys/module_mp_p3.F> <table dir> <out dir>" >&2
    exit 2
fi
src=$1
tabledir=$2
outdir=$3
here=$(cd "$(dirname "$0")" && pwd)

got=$(sha256sum "$src" | cut -d' ' -f1)
if [ "$got" != "$AUTHORITY_SHA256" ]; then
    echo "REFUSED: $src has sha256 $got" >&2
    echo "expected $AUTHORITY_SHA256 (WRF phys/module_mp_p3.F, P3 v4.5.2)" >&2
    echo "Measuring a different file would pin a digest to nothing." >&2
    exit 1
fi

work=${outdir}/build
mkdir -p "$work" "$outdir/O0" "$outdir/O2"
cp "$src" "$work/module_mp_p3_pristine.F"

python3 - "$work" "$here" <<'PY'
import sys, pathlib
work, here = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
src = (work / "module_mp_p3_pristine.F").read_text(newline="")
add = (here / "dump_append.f90").read_text(newline="")
marker = " END MODULE microphy_p3"
public = " public :: p3_main, polysvp1, p3_init"
assert src.count(marker) == 1 and src.count(public) == 1
out = src.replace(public, public + "\n public :: p3_oracle_dump", 1)
out = out.replace(marker, add + marker, 1)
(work / "module_mp_p3_oracle.F").write_text(out, newline="")
PY

echo "--- the ONLY difference from upstream (additions only) ---"
diff "$work/module_mp_p3_pristine.F" "$work/module_mp_p3_oracle.F" || true
echo "----------------------------------------------------------"

FFLAGS="-cpp -ffree-form -ffree-line-length-none"
for opt in O0 O2; do
    gfortran $FFLAGS "-$opt" -J"$work" -c "$work/module_mp_p3_oracle.F" \
        -o "$work/mod_$opt.o"
    gfortran $FFLAGS "-$opt" -I"$work" "$here/drv.f90" "$work/mod_$opt.o" \
        -o "$work/drv_$opt"
    "$work/drv_$opt" "$tabledir" "$outdir/$opt" 1
done

echo "--- -O0 vs -O2 ---"
for f in itab itabcoll vn_table vm_table revap_table mu_r_table; do
    if cmp -s "$outdir/O0/$f.f32" "$outdir/O2/$f.f32"; then
        echo "  $f: identical"
    else
        echo "  $f: DIFFER"
    fi
done
echo "dumps in $outdir/O0 and $outdir/O2"
