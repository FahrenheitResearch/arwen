#!/bin/sh
# Run this only on an authorized CPU host. OUT must be a lane-owned path.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
out=${1:?pass a lane-owned output directory}
mkdir -p "$out"
out=$(CDPATH= cd -- "$out" && pwd)
cd "$out"
curl --fail --location --silent --show-error https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/phys/module_cu_tiedtke.F -o module_cu_tiedtke.F
curl --fail --location --silent --show-error https://raw.githubusercontent.com/wrf-model/WRF/v4.6.1/share/module_model_constants.F -o module_model_constants.F
echo '3514aaaaf6084edc0f6a6b2097e34607562f0b6f12934cc3c675a9e651566999  module_cu_tiedtke.F' | sha256sum -c -
sha256sum module_cu_tiedtke.F module_model_constants.F "$here/oracle.F90" "$here/host.cpp" > sources.sha256
gfortran --version > compiler.txt
ldd --version >> compiler.txt
# The pinned target is the ordinary x86 branch at F:2938-2942.
gfortran -cpp -U AARCH64_X86_CORRECTNESS_FIX -ffree-form -ffree-line-length-none -O0 -fno-fast-math -ffp-contract=off -fPIC -shared module_model_constants.F module_cu_tiedtke.F "$here/oracle.F90" -o libcu6_oracle.so
# A second oracle diagnoses out-of-bounds accesses and uninitialized scalars.
gfortran -cpp -U AARCH64_X86_CORRECTNESS_FIX -ffree-form -ffree-line-length-none -O0 -fno-fast-math -ffp-contract=off -fcheck=all -finit-real=snan -fPIC -shared module_model_constants.F module_cu_tiedtke.F "$here/oracle.F90" -o libcu6_checked.so
g++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off -fPIC -shared "$here/host.cpp" -o libcu6_host.so
g++ -std=c++17 -O2 -fno-fast-math -ffp-contract=off -DTK_NATIVE_TAN_CONTROL -fPIC -shared "$here/host.cpp" -o libcu6_native_tan_control.so
printf '%s\n' 'gfortran: -cpp -U AARCH64_X86_CORRECTNESS_FIX -ffree-form -ffree-line-length-none -O0 -fno-fast-math -ffp-contract=off -fPIC -shared' 'checked: add -fcheck=all -finit-real=snan' 'g++: -std=c++17 -O2 -fno-fast-math -ffp-contract=off -fPIC -shared' > flags.txt
