#!/usr/bin/env bash
set -euo pipefail
src=$(realpath "$1")
build=$(realpath -m "$2")
fixtures=$(realpath -m "$3")
tables=$(realpath -m "$4")
here=$(cd "$(dirname "$0")" && pwd)

# The recipe and hashes are literals in this build script.
diff -u "$here/source-sha256sums.txt" <(cat <<'SOURCE_PINS'
073113232be6e8f0d800825e0af5b0c5434cd5db1a3ef428569e79a68d8c92d4  chem/module_optical_averaging.F
5afcfa5d26e3d40c42be20253d7f1d3b88cf0b279e5f87d164912bff8b792b5e  chem/module_data_rrtmgaeropt.F
67ffc5b574ad6994e572890aefa3b9069843af5e4b08fd68583437baf4300709  chem/module_data_sorgam.F
efc13a98a4899babee8cf420b23ffb9bc4de8049426bc78d85a7fb2ed3b2b3c2  chem/module_data_gocartchem.F
d7288531253596075ab0703448b164a083fb943802655adf924cc62feb078cf0  chem/module_data_gocart_seas.F
8694100875645b54a58bcff38d462ad3c9fd115261fb39941358466c84b62329  phys/module_data_gocart_dust.F
55850415d1d357f350bf975fb9a860744e19a2fb3330ded310987279c1d4169f  chem/module_data_mosaic_asect.F
46ea2932a1c1c1cfdc41c6ef23a4d0405bc768110050f9a4a3ba1dca4f7365d0  chem/module_aer_opt_out.F
447345d2658cd370e6bc97ff2ab582a5d12b84adffc58f72a938b353e017987e  phys/module_ra_rrtmg_sw.F
416a92e91bf5475b5f94f3bdf0f18fdcbb1c7bc9c88131cb5d7457e38b542f6e  Registry/registry.chem
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
SOURCE_PINS
)
diff -u "$here/extract-ranges.txt" <(cat <<'EXTRACT_RANGES'
chem/module_optical_averaging.F 3545 4186 prep.inc
chem/module_optical_averaging.F 296 303 prep_call.inc
chem/module_optical_averaging.F 426 436 mie_call.inc
chem/module_optical_averaging.F 476 482 sw_clamps.inc
chem/module_optical_averaging.F 493 496 lw_clamps.inc
chem/module_optical_averaging.F 4453 4454 fit_limits.inc
chem/module_optical_averaging.F 4466 4467 fit_sizes.inc
chem/module_optical_averaging.F 3689 3727 section_fractions.inc
chem/module_optical_averaging.F 4221 6978 mie.inc
chem/module_data_sorgam.F 728 749 modal_parameters.inc
chem/module_data_sorgam.F 1191 1200 hygro_parameters.inc
chem/module_data_mosaic_asect.F 581 581 msa_parameter.inc
chem/module_data_gocartchem.F 18 20 mass_parameters.inc
phys/module_ra_rrtmg_sw.F 10241 10249 rrtmg_parameters.inc
phys/module_ra_rrtmg_sw.F 11355 11430 rrtmg_conversion.inc
EXTRACT_RANGES
)
(cd "$src"; sha256sum -c "$here/source-sha256sums.txt")
mkdir -p "$build" "$fixtures" "$tables"
cd "$build"
while read -r file first last name; do
  sed -n "${first},${last}p" "$src/$file" > "$name"
done < "$here/extract-ranges.txt"
flags=(-O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form -ffree-line-length-none -fallow-argument-mismatch -I .)
gfortran "${flags[@]}" -c "$here/stub_optics.F90" "$src/share/module_model_constants.F" "$src/chem/module_data_gocart_seas.F" "$src/phys/module_data_gocart_dust.F" "$src/chem/module_data_rrtmgaeropt.F" "$here/optics_module.F90" "$src/chem/module_aer_opt_out.F" "$here/rrtmg_module.F90" "$here/oracle_io.F90" "$here/run_gocart_optics.F90"
nm -u stub_optics.o module_model_constants.o module_data_gocart_seas.o module_data_gocart_dust.o module_data_rrtmgaeropt.o optics_module.o module_aer_opt_out.o rrtmg_module.o oracle_io.o run_gocart_optics.o > undefined-O0.txt
if grep -q _ZGV undefined-O0.txt; then echo 'FAIL: vector libm in oracle'; exit 4; fi
gfortran -Ofast -ftree-vectorize -c "$here/libmvec_positive_control.F90" -o control.o
nm -u control.o > undefined-control.txt
grep _ZGV undefined-control.txt
gfortran -o run_optics stub_optics.o module_model_constants.o module_data_gocart_seas.o module_data_gocart_dust.o module_data_rrtmgaeropt.o optics_module.o module_aer_opt_out.o rrtmg_module.o oracle_io.o run_gocart_optics.o
gfortran --version | head -1
ldd --version | head -1
./run_optics "$fixtures" "$tables"
if ./run_optics --negative > negative-tau.log 2>&1; then
  echo 'FAIL: negative RRTMG tau was accepted'; exit 7
fi
grep 'ERROR: Negative total optical depth' negative-tau.log
if ./run_optics --badindex > invalid-refindex.log 2>&1; then
  echo 'FAIL: invalid Mie refractive index was accepted'; exit 8
fi
grep 'mieaer /refr/ outside range' invalid-refindex.log
mv "$fixtures/tables/"* "$tables/"
rmdir "$fixtures/tables"
(cd "$fixtures"; find . -type f ! -name oracle-sha256sums.txt ! -name PROVENANCE.md -print0 | sort -z | xargs -0 sha256sum > oracle-sha256sums.txt)
(cd "$tables"; find . -name '*.bin' -o -name MANIFEST.txt | sort | xargs sha256sum > table-sha256sums.txt)
echo 'optics oracle completed'
