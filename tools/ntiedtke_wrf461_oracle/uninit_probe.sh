#!/usr/bin/env bash
# Does New Tiedtke's answer depend on uninitialised memory?
#
# Build the pinned scheme twice, identically except for gfortran's
# -finit-real, and compare the driver's output bitwise.  -finit-real=nan
# seeds every uninitialised local with a NaN, which propagates loudly
# through anything that reads one; -finit-real=zero seeds them with 0.
# If any uninitialised value reaches the output on this fixture, the two
# runs disagree.
#
# This is a DIAGNOSTIC build, not the oracle: -finit-real changes code
# generation, so nothing it produces is pinned.
set -euo pipefail
src=/tmp/wrf461src
hh="$(cd "$(dirname "$0")" && pwd)"

for mode in zero nan; do
    d=/tmp/ntinit_${mode}
    rm -rf "$d"; mkdir -p "$d"; cd "$d"
    gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none \
        -finit-real=${mode} -o k.o "$src/phys/ccpp_kind_types.F"
    gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -I "$d" \
        -finit-real=${mode} -o s.o "$src/phys/physics_mmm/cu_ntiedtke.F90"
    gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -I "$d" \
        -finit-real=${mode} -o m.o "$src/phys/module_cu_ntiedtke.F"
    gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$d" \
        -finit-real=${mode} "$hh/nt_cases.F90"
    gfortran -c -O0 -ffree-form -ffree-line-length-none -I "$d" \
        -finit-real=${mode} "$hh/run_cu_ntiedtke.F90"
    gfortran -o run k.o s.o m.o nt_cases.o run_cu_ntiedtke.o
    ./run > /dev/null
    echo "built and ran -finit-real=${mode}"
done

echo
echo "=== bitwise diff of the driver's output across the two seedings ==="
for f in nt-levels.csv nt-surface.csv; do
    if cmp -s /tmp/ntinit_zero/$f /tmp/ntinit_nan/$f; then
        echo "  $f  IDENTICAL"
    else
        n=$(diff /tmp/ntinit_zero/$f /tmp/ntinit_nan/$f | grep -c '^<' || true)
        echo "  $f  *** $n ROWS DIFFER ***"
    fi
done

echo
echo "=== and against the pinned oracle build (-finit-real absent) ==="
for f in nt-levels.csv nt-surface.csv; do
    if cmp -s /tmp/ntbuild/$f /tmp/ntinit_zero/$f; then
        echo "  $f  oracle == zero-seeded"
    else
        echo "  $f  *** oracle differs from zero-seeded ***"
    fi
done
