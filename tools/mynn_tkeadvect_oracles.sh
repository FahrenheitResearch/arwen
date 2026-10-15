#!/usr/bin/env bash
set -euo pipefail
: "${L:?lane folder required}" "${T:?lane checkout required}"
if [[ -z "${WRF_SOURCE_ROOT:-}" ]]; then source /work/wrf/env.sh; fi
mkdir -p "$L/oracle-stock" "$L/oracle-fork" "$L/oracle-advect"
printf '%s\n' "$L/oracle-advect" >> "$L/MANIFEST.txt"
stock="$T/tools/mynn_pbl_wrf461_oracle"
fork="$T/tools/mynn_pbl_gsd41_oracle"
cd "$L/oracle-stock"
gfortran -O0 -cpp -ffree-form -ffree-line-length-none -ffp-contract=off -c \
  "$stock/stub_wrf.F90" "$WRF_SOURCE_ROOT/phys/module_bl_mynn_common.F" \
  "$WRF_SOURCE_ROOT/phys/module_bl_mynn.F" "$stock/run_driver_families.F90"
gfortran -o run stub_wrf.o module_bl_mynn_common.o module_bl_mynn.o run_driver_families.o
for length in 1 2; do test -f "active-$length.csv" || ./run "active-$length.csv" "$length" 12 "" true; done
cd "$L/oracle-fork"
curl -fL --retry 3 'https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21/sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F' -o source.F
printf '%s\n' '42178a22693ad14b1a3adcf3e44be3047f051abf80f3a64c0ce9ed9ac2ec6d8a  source.F' | sha256sum -c -
sed 's/qtke(k) = 0.5\*qkw(k)  ! qkw -> TKE/qtke(k) = 0.5*(qkw(k)**2)  ! qkw -> TKE/' source.F > module_bl_mynn.F
gfortran -O0 -cpp -ffree-form -ffree-line-length-none -ffp-contract=off -c \
  "$fork/stub_wrf39.F90" module_bl_mynn.F "$fork/run_driver_families_gsd41.F90"
gfortran -o run stub_wrf39.o module_bl_mynn.o run_driver_families_gsd41.o
for length in 1 2; do
  printf '&review tkeadvect=.true., bl_mynn_mixlength=%s /\n' "$length" > "controls-$length.nml"
  test -f "active-$length.csv" || ./run "active-$length.csv" 12 "controls-$length.nml"
done
cd "$L/oracle-advect"
adv="$T/tools/advect_wrf471_oracle"
flags=(-O0 -cpp -ffree-form -ffree-line-length-none -ffp-contract=off -fallow-argument-mismatch -fcheck=all)
defs=(-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4)
gfortran "${flags[@]}" -c "$adv/stub_wrf.F90"
gfortran "${flags[@]}" "${defs[@]}" -c "$WRF_SOURCE_ROOT/share/module_model_constants.F" \
  "$WRF_SOURCE_ROOT/frame/module_wrf_error.F" "$WRF_SOURCE_ROOT/dyn_em/module_advect_em.F"
gfortran "${flags[@]}" -c "$adv/run_advect.F90"
gfortran -o run stub_wrf.o module_model_constants.o module_wrf_error.o module_advect_em.o run_advect.o
sha256sum "$WRF_SOURCE_ROOT/dyn_em/module_advect_em.F" run > "$L/receipts/advect-source.txt"
gfortran --version | head -1 > "$L/receipts/compiler.txt"
