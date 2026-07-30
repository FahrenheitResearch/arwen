#!/usr/bin/env bash
# Self-contained runtime profile for the pinned WPS/WRF binaries in WSL.

N5S_WRF_LIBRARIES_ROOT=${N5S_WRF_LIBRARIES_ROOT:-/home/drew/WRF_BUILD/LIBRARIES}
N5S_NETCDF_ROOT="$N5S_WRF_LIBRARIES_ROOT/netcdf"
N5S_GRIB2_ROOT="$N5S_WRF_LIBRARIES_ROOT/grib2"
N5S_MPICH_ROOT="$N5S_WRF_LIBRARIES_ROOT/mpich"

for n5s_required_dir in \
  "$N5S_NETCDF_ROOT/lib" \
  "$N5S_GRIB2_ROOT/lib" \
  "$N5S_GRIB2_ROOT/include" \
  "$N5S_MPICH_ROOT/lib"; do
  if [[ ! -d "$n5s_required_dir" ]]; then
    echo "N5S runtime environment is missing required directory: $n5s_required_dir" >&2
    return 2 2>/dev/null || exit 2
  fi
done

export NETCDF="$N5S_NETCDF_ROOT"
export JASPERLIB="$N5S_GRIB2_ROOT/lib"
export JASPERINC="$N5S_GRIB2_ROOT/include"
N5S_RUNTIME_LIBRARY_PATH="$N5S_NETCDF_ROOT/lib:$N5S_GRIB2_ROOT/lib:$N5S_MPICH_ROOT/lib"
case "${LD_LIBRARY_PATH:-}" in
  "$N5S_RUNTIME_LIBRARY_PATH"|"$N5S_RUNTIME_LIBRARY_PATH":*) ;;
  *) export LD_LIBRARY_PATH="$N5S_RUNTIME_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac

unset n5s_required_dir N5S_RUNTIME_LIBRARY_PATH
