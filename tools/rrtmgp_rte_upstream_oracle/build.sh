#!/usr/bin/env bash
# Build the upstream RTE+RRTMGP column oracle (rte_column_oracle.F90).
#
#   build.sh UPSTREAM_CHECKOUT BUILD_DIR [--boundary-fixes]
#
# With --boundary-fixes the build compiles a copy of the checkout with
# upstream-boundary-fixes.patch applied: three one-line changes that measure
# an interpolation fraction from the table node actually read (temperature
# at 355 K, eta at 1, Planck temperature at 355 K).  WOOF carries those
# three fixes; the patched build is the referee that shows every remaining
# difference from the stock build is one of them (README.md).
#
# UPSTREAM_CHECKOUT is a clone of github.com/earth-system-radiation/rte-rrtmgp
# at fa107a16120051c4124305c6b3d4c87059119f58, the commit
# gpuwm/core/rrtmgp.py cites; the script refuses any other commit.
#
# Single precision (-DRTE_USE_SP: wp = c_float), -O0 and -ffp-contract=off:
# every float operation is one binary32 rounding in source order, no
# multiply-add is fused, and EXP/LOG are the C library's expf/logf.  The
# "default" CPU kernels are compiled (rte/kernels, rrtmgp/kernels), not the
# accel or extern variants.  Needs gfortran and netcdf-fortran (nf-config).
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh UPSTREAM_CHECKOUT BUILD_DIR [--boundary-fixes]" >&2
    exit 2
fi
up=$(realpath "$1")
build=$(realpath -m "$2")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
want=fa107a16120051c4124305c6b3d4c87059119f58
got=$(git -C "$up" rev-parse HEAD)
if [[ "$got" != "$want" ]]; then
    echo "upstream checkout is at $got, not $want" >&2
    exit 1
fi
mkdir -p "$build"
cd "$build"
if [[ "${3:-}" == "--boundary-fixes" ]]; then
    rm -rf upstream-fixed
    mkdir upstream-fixed
    cp -r "$up/rte" "$up/rrtmgp" "$up/examples" upstream-fixed/
    patch -s -p1 -d upstream-fixed < "$here/upstream-boundary-fixes.patch"
    up=$build/upstream-fixed
elif [[ -n "${3:-}" ]]; then
    echo "unknown option $3" >&2
    exit 2
fi
FFLAGS="-O0 -ffp-contract=off -cpp -DRTE_USE_SP -ffree-line-length-none \
  -fno-fast-math $(nf-config --fflags)"
sources=(
  rte/kernels/mo_rte_kind.F90
  rte/frontend/mo_rte_config.F90
  rte/kernels/mo_rte_util_array.F90
  rte/frontend/mo_rte_util_array_validation.F90
  rte/kernels/mo_optical_props_kernels.F90
  rte/frontend/mo_optical_props.F90
  rte/frontend/mo_source_functions.F90
  rte/kernels/mo_fluxes_broadband_kernels.F90
  rte/frontend/mo_fluxes.F90
  rte/kernels/mo_rte_solver_kernels.F90
  rte/frontend/mo_rte_lw.F90
  rte/frontend/mo_rte_sw.F90
  rte/kernels/mo_gas_optics_constants.F90
  rte/kernels/mo_gas_optics_utils.F90
  rte/frontend/gas-optics-template/mo_gas_optics_util_string.F90
  rte/frontend/gas-optics-template/mo_gas_concentrations.F90
  rte/frontend/gas-optics-template/mo_gas_optics.F90
  rrtmgp/kernels/mo_gas_optics_rrtmgp_kernels.F90
  rrtmgp/kernels/mo_cloud_optics_rrtmgp_kernels.F90
  rrtmgp/frontend/mo_gas_optics_rrtmgp.F90
  rrtmgp/frontend/mo_cloud_optics_rrtmgp.F90
  rrtmgp/frontend/mo_aerosol_optics_rrtmgp_merra.F90
  examples/shared-utils/mo_testing_utils.F90
  examples/shared-utils/mo_simple_netcdf.F90
  rrtmgp/data-loading-examples/mo_optics_utils_rrtmgp.F90
)
objs=()
for s in "${sources[@]}"; do
    gfortran -c $FFLAGS "$up/$s"
    objs+=("$(basename "${s%.F90}").o")
done
gfortran -c $FFLAGS "$here/rte_column_oracle.F90"
gfortran -o rte_column_oracle "${objs[@]}" rte_column_oracle.o \
    $(nf-config --flibs)
{
    echo "upstream $want${3:+ with $3}"
    gfortran --version | sed -n 1p
    ldd --version | sed -n 1p
    echo "FFLAGS $FFLAGS"
    sha256sum "$here/rte_column_oracle.F90" "$here/upstream-boundary-fixes.patch"
} > build-provenance.txt
echo "built $build/rte_column_oracle"
