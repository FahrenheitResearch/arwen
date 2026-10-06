#!/usr/bin/env bash
# Builds the GSD MYNN v4.1 oracle from the unmodified NOAA-EMC WRF 3.9 branch
# source (tag v4.1.21, sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F) and
# records option-2 mixing lengths:
#   mixlength2-gsd41.csv      the source as written (0.5*qkw at :995)
#   mixlength2-gsd41-sq.csv   that one line squared, 0.5*(qkw**2): the
#                             bl_mynn_gsd41_unsquared_qtke = false form
#   condensation-gsd41.csv    mym_condensation, bl_mynn_cloudpdf = 2
#   turbulence2-gsd41-sq.csv  mym_turbulence, level 2.5, mixing length 2,
#                             the squared :995 form (the default)
#   plume-condensation-gsd41.csv  condensation_edmf, 24 plume states
# usage: build.sh FORK_module_bl_mynn.F BUILD_DIR
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: build.sh module_bl_mynn.F BUILD_DIR" >&2; exit 2; }
src=$(realpath "$1"); build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$build/asis" "$build/sq"
cp "$src" "$build/asis/module_bl_mynn.F"
# The one patched line; the build refuses unless it matches exactly once.
line='           qtke(k) = 0.5\*qkw(k)  ! qkw -> TKE'
test "$(grep -c "$line" "$src")" = 1
sed "s/$line/           qtke(k) = 0.5*(qkw(k)**2)  ! qkw -> TKE/" \
    "$src" > "$build/sq/module_bl_mynn.F"
test "$(grep -c 'qtke(k) = 0.5\*(qkw(k)\*\*2)  ! qkw -> TKE' "$build/sq/module_bl_mynn.F")" = 1
for v in asis sq; do
  cd "$build/$v"
  gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none "$here/stub_wrf39.F90"
  gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none -I. module_bl_mynn.F
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_mixlength_gsd41.F90"
  gfortran -o run_mixlength_gsd41 stub_wrf39.o module_bl_mynn.o run_mixlength_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_condensation_gsd41.F90"
  gfortran -o run_condensation_gsd41 stub_wrf39.o module_bl_mynn.o run_condensation_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_turbulence_gsd41.F90"
  gfortran -o run_turbulence_gsd41 stub_wrf39.o module_bl_mynn.o run_turbulence_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_plume_condensation_gsd41.F90"
  gfortran -o run_plume_condensation_gsd41 stub_wrf39.o module_bl_mynn.o run_plume_condensation_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_dmp_mf_gsd41.F90"
  gfortran -o run_dmp_mf_gsd41 stub_wrf39.o module_bl_mynn.o run_dmp_mf_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_transport_gsd41.F90"
  gfortran -o run_transport_gsd41 stub_wrf39.o module_bl_mynn.o run_transport_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_transport_cloud_gsd41.F90"
  gfortran -o run_transport_cloud_gsd41 stub_wrf39.o module_bl_mynn.o run_transport_cloud_gsd41.o
  gfortran -c -O0 -ffree-form -ffree-line-length-none -I. "$here/run_predict_gsd41.F90"
  gfortran -o run_predict_gsd41 stub_wrf39.o module_bl_mynn.o run_predict_gsd41.o
done
rm -f "$build/mixlength2-gsd41.csv" "$build/mixlength2-gsd41-sq.csv" \
    "$build/condensation-gsd41.csv"
"$build/asis/run_mixlength_gsd41" "$build/mixlength2-gsd41.csv"
"$build/sq/run_mixlength_gsd41" "$build/mixlength2-gsd41-sq.csv"
"$build/asis/run_condensation_gsd41" "$build/condensation-gsd41.csv"
rm -f "$build/turbulence2-gsd41-sq.csv"
"$build/sq/run_turbulence_gsd41" "$build/turbulence2-gsd41-sq.csv"
"$build/asis/run_plume_condensation_gsd41" "$build/plume-condensation-gsd41.csv"
rm -f "$build/dmp-mf-gsd41.csv"
"$build/asis/run_dmp_mf_gsd41" "$build/dmp-mf-gsd41.csv"
"$build/asis/run_transport_gsd41" "$build/transport-gsd41.csv"
"$build/asis/run_transport_cloud_gsd41" "$build/transport-cloud-gsd41.csv"
"$build/asis/run_predict_gsd41" "$build/predict-gsd41.csv"
cd "$build"
sha256sum "$src" "$here/stub_wrf39.F90" "$here/run_mixlength_gsd41.F90" \
    "$here/run_condensation_gsd41.F90" "$here/run_turbulence_gsd41.F90" \
    "$here/run_plume_condensation_gsd41.F90" \
    "$here/run_dmp_mf_gsd41.F90" dmp-mf-gsd41.csv \
    "$here/run_transport_gsd41.F90" transport-gsd41.csv \
    "$here/run_transport_cloud_gsd41.F90" transport-cloud-gsd41.csv \
    "$here/run_predict_gsd41.F90" predict-gsd41.csv \
    mixlength2-gsd41.csv mixlength2-gsd41-sq.csv condensation-gsd41.csv \
    turbulence2-gsd41-sq.csv plume-condensation-gsd41.csv > oracle-sha256sums.txt
gfortran --version | head -1 > compiler.txt
