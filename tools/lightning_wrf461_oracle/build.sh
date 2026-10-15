#!/usr/bin/env bash
set -euo pipefail
source_root=${1:?WRF source root}
out=${2:?build directory}
here=$(cd "$(dirname "$0")" && pwd)
test "$(git -C "$source_root" rev-parse HEAD)" = d66e442fccc04111067e29274c9f9eaccc3cef28 || {
  git -C "$source_root" rev-parse HEAD
  echo "Use the pinned WRF v4.6.1 commit recorded in source.sha256"; exit 2;
}
git -C "$source_root" diff --quiet HEAD -- phys/module_lightning_driver.F phys/module_ltng_crmpr92.F phys/module_ltng_cpmpr92z.F phys/module_ltng_lpi.F phys/module_ltng_iccg.F
mkdir -p "$out"
cd "$out"
sha256sum "$source_root"/phys/module_{lightning_driver,ltng_crmpr92,ltng_cpmpr92z,ltng_lpi,ltng_iccg}.F > source.sha256
gfortran -O0 -ffree-form -ffp-contract=off -fno-tree-vectorize -fcheck=bounds -ffree-line-length-none -cpp -DEM_CORE=1 -DWRF_CHEM=0 \
 "$here/stubs.F90" "$source_root/phys/module_ltng_crmpr92.F" "$source_root/phys/module_ltng_cpmpr92z.F" \
 "$source_root/phys/module_ltng_lpi.F" "$source_root/phys/module_ltng_iccg.F" \
 "$source_root/phys/module_lightning_driver.F" "$here/run_lightning.F90" -o run_lightning
