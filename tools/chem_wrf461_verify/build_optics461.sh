#!/usr/bin/env bash
# Build the GOCART optics column oracle from WRF v4.6.1's own Fortran.
#
#   bash tools/chem_wrf461_verify/build_optics461.sh WRF461_CHECKOUT WRF471_CHECKOUT OUT
#
# WRF v4.7.1 changed chem/module_optical_averaging.F and chem/optical_driver.F
# for GOCART (upstream commit note "A. Ukhov 09/17/24, bug fix"): the
# GOCART-to-MOSAIC mass redistribution in 4.6.1 uses 8 sections spanning
# 0.039-10 um and adds only sea-salt bins 1-3 and dust bins 1-4, so the
# largest sea-salt bin and the largest dust bin never reach the optics; 4.7.1
# uses 9 sections to 20 um and all bins.  The published optics oracle is
# 4.7.1's.  This script rebuilds it from 4.6.1:
#
#   * every line-range extract of module_optical_averaging.F is re-located in
#     the 4.6.1 file by aligning it with 4.7.1 (map_ranges.py); a range whose
#     text differs is reported, and only the two ranges the bug fix touches
#     (prep.inc, section_fractions.inc) may differ;
#   * the harness's section count follows 4.6.1's optical_driver.F for
#     GOCART_SIMPLE (nbin_o = 8) and its table dumps use 10 um, in a scratch
#     copy of the harness under OUT (the repository copy is untouched);
#   * every other WRF file is compiled from the 4.6.1 checkout after a cmp
#     against 4.7.1 proves it is the same file.
# Compile flags are build_optics.sh's (-O0, no libmvec; the _ZGV check runs).
set -euo pipefail
src=$(realpath "${1:?usage: build_optics461.sh WRF461 WRF471 OUT}")
ref=$(realpath "${2:?}")
out=$(realpath -m "${3:?}")/optics461
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
gocart=$(cd "$here/../chem_wrf471_oracle/gocart" && pwd)
rm -rf "$out"
mkdir -p "$out/harness" "$out/build" "$out/fixtures" "$out/tables"
cp -a "$gocart/." "$out/harness/"
h="$out/harness"

for rel in chem/module_data_rrtmgaeropt.F chem/module_data_sorgam.F chem/module_data_gocartchem.F \
           chem/module_data_gocart_seas.F phys/module_data_gocart_dust.F chem/module_data_mosaic_asect.F \
           chem/module_aer_opt_out.F phys/module_ra_rrtmg_sw.F Registry/registry.chem \
           share/module_model_constants.F; do
    cmp "$src/$rel" "$ref/$rel" || { echo "$rel differs between 4.6.1 and 4.7.1" >&2; exit 3; }
done
grep -q "nbin_o = 8" <(sed -n '/GOCART_SIMPLE/,/nbin_o/p' "$src/chem/optical_driver.F") ||
    { echo "4.6.1 optical_driver.F does not give GOCART_SIMPLE nbin_o = 8" >&2; exit 3; }

python3 "$here/map_ranges.py" "$ref/chem/module_optical_averaging.F" "$src/chem/module_optical_averaging.F" \
    "$gocart/extract-ranges.txt" "$out/extract-ranges-461.txt" > "$out/range-map.txt"
cat "$out/range-map.txt"

# Section count 9 -> 8 and upper size 20 -> 10 um, only where the harness sizes sections.
sed -i -e 's/nbin_o=9/nbin_o=8/g' -e 's/(1,nz,1,9)/(1,nz,1,8)/g' -e 's/(1,nz,1,9,/(1,nz,1,8,/g' \
       -e 's/(9,nz)/(8,nz)/g' -e 's/(9,nz,/(8,nz,/g' "$h/optics_module.F90"
sed -i -e 's/(1,nz,1,9)/(1,nz,1,8)/g' -e 's/(1,nz,1,9,/(1,nz,1,8,/g' -e 's/(nc,nz,1,9)/(nc,nz,1,8)/g' \
       -e 's/(9,nz)/(8,nz)/g' -e 's/(9,nz,/(8,nz,/g' \
       -e 's/mass_i(9),mass_j(9),mass_c(9),nn(9),diam(9)/mass_i(8),mass_j(8),mass_c(8),nn(8),diam(8)/' \
       -e 's/1\.,9,0\.0390625,20\./1.,8,0.0390625,10./g' -e 's/^ do n=1,9$/ do n=1,8/' \
       -e 's/nbin_o=9/nbin_o=8/' -e 's/dlo_sectm(9),dhi_sectm(9)/dlo_sectm(8),dhi_sectm(8)/' \
       -e 's/seasfrc_goc9bin(4,9),dustfrc_goc9bin(5,9)/seasfrc_goc8bin(4,8),dustfrc_goc8bin(5,8)/' \
       -e "s/oracle_put('seasfrc_goc9bin',seasfrc_goc9bin)/oracle_put('seasfrc_goc8bin',seasfrc_goc8bin)/" \
       -e "s/oracle_put('dustfrc_goc9bin',dustfrc_goc9bin)/oracle_put('dustfrc_goc8bin',dustfrc_goc8bin)/" \
       -e 's/dhi_um=20\./dhi_um=10./' "$h/run_gocart_optics.F90"
diff "$gocart/optics_module.F90" "$h/optics_module.F90" > "$out/harness-optics_module.diff" || true
diff "$gocart/run_gocart_optics.F90" "$h/run_gocart_optics.F90" > "$out/harness-run_gocart_optics.diff" || true
if grep -nE '\b9\b' "$h/optics_module.F90" | grep -v '^#'; then echo "a 9 is left in optics_module.F90" >&2; exit 6; fi

cd "$out/build"
while read -r file first last name; do
    sed -n "${first},${last}p" "$src/$file" > "$name"
done < "$out/extract-ranges-461.txt"
flags=(-O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4
       -ffree-form -ffree-line-length-none -fallow-argument-mismatch -I .)
gfortran "${flags[@]}" -c "$h/stub_optics.F90" "$src/share/module_model_constants.F" \
    "$src/chem/module_data_gocart_seas.F" "$src/phys/module_data_gocart_dust.F" \
    "$src/chem/module_data_rrtmgaeropt.F" "$h/optics_module.F90" "$src/chem/module_aer_opt_out.F" \
    "$h/rrtmg_module.F90" "$h/oracle_io.F90" "$h/run_gocart_optics.F90"
objs=(stub_optics.o module_model_constants.o module_data_gocart_seas.o module_data_gocart_dust.o
      module_data_rrtmgaeropt.o optics_module.o module_aer_opt_out.o rrtmg_module.o oracle_io.o run_gocart_optics.o)
nm -u "${objs[@]}" > undefined-O0.txt
if grep -q _ZGV undefined-O0.txt; then echo 'FAIL: vector libm in oracle' >&2; exit 4; fi
gfortran -Ofast -ftree-vectorize -c "$h/libmvec_positive_control.F90" -o control.o
nm -u control.o > undefined-control.txt
grep -q _ZGV undefined-control.txt || { echo 'positive control produced no _ZGV' >&2; exit 5; }
gfortran -o run_optics "${objs[@]}"
{ gfortran --version | sed -n 1p; ldd --version | sed -n 1p; } > "$out/toolchain.txt"
./run_optics "$out/fixtures" "$out/tables" > run_optics.log
echo "optics 4.6.1 oracle: $(find "$out/fixtures" -name MANIFEST.txt | wc -l) case folders"
