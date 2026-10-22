#!/usr/bin/env bash
# Build the SCM host with one saved-column scheme at the operational pin.
# usage: build_scm.sh SCM_SOURCE NCEPLIBS_MPI_PREFIX BUILD_DIR
# The prefix supplies MPICH 4.2.3, bacio 2.4.1, sp 2.5.0 and w3emc 2.12.0.
# Numerical physics sources remain byte-unmodified. The host's thread-count
# assignment and build tables are adapted to the pinned framework interface.
set -euo pipefail
source_root=$(realpath "$1")
prefix=$(realpath "$2")
build=$(realpath -m "$3")
mkdir -p "$build/tmp"
export TMPDIR=$build/tmp CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 RAYON_NUM_THREADS=4
export PATH=$prefix/bin:$PATH NETCDF=/usr CMAKE_PREFIX_PATH=$prefix
export bacio_ROOT=$prefix sp_ROOT=$prefix w3emc_ROOT=$prefix
host=d939f51fe8875192015fd0a0cfcf3c503cbb69e2
physics=edd85a55502179611fdb7b902f0639c305cd1c3b
framework=b945b07f18f69a1abec99b54f9e92c23f6103d8d
[[ $(git -C "$source_root" rev-parse HEAD) == "$host" ]]
[[ $(git -C "$source_root/ccpp/physics" rev-parse HEAD) == "$physics" ]]
[[ $(git -C "$source_root/ccpp/framework" rev-parse HEAD) == "$framework" ]]
# The optional simulator's host base type is absent from the operational
# physics tree. These host-only hooks come from the physics paired with SCM 7.
while IFS= read -r path; do
    git -C "$source_root/ccpp/physics" cat-file -e "$physics:$path" 2>/dev/null && continue
    git -C "$source_root/ccpp/physics" show "da75531bf90a799e4c0dcb988cc591d82bd024de:$path" > "$source_root/ccpp/physics/$path"
done < <(git -C "$source_root/ccpp/physics" ls-tree -r --name-only da75531bf90a799e4c0dcb988cc591d82bd024de physics/Interstitials/UFS_SCM_NEPTUNE | grep ccpp_suite_sim)
python3 - "$source_root" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / 'ccpp/suites/suite_SCM_saved_drag.xml').write_text('''<?xml version="1.0" encoding="UTF-8"?>
<suite name="SCM_saved_drag" version="1"><group name="physics"><subcycle loop="1"><scheme>drag_suite</scheme></subcycle></group></suite>
''')
config = root / 'ccpp/config/ccpp_prebuild_config.py'
with config.open('a') as out:
    out.write("\nSCHEME_FILES = ['ccpp/physics/physics/GWD/drag_suite.F90', 'ccpp/physics/physics/Interstitials/UFS_SCM_NEPTUNE/GFS_suite_interstitial_rad_reset.F90']\nfrom parse_tools import register_fortran_ddt_name\nregister_fortran_ddt_name('MPI_Comm')\n")
host = root / 'scm/src/scm.F90'
host.write_text(host.read_text().replace('  cdata%thrd_cnt = 1',
                '  ! The pinned single-thread interface carries thrd_no only.', 1))
PY
cat > "$build/compat.cmake" <<EOF
find_package(bacio REQUIRED)
if(TARGET bacio::bacio_4 AND NOT TARGET bacio::bacio)
  add_library(bacio::bacio ALIAS bacio::bacio_4)
endif()
cmake_language(DEFER CALL target_sources ccpp_physics PRIVATE "$source_root/ccpp/physics/physics/tools/funcphys.f90")
cmake_language(DEFER CALL target_link_libraries scm MPI::MPI_Fortran)
EOF
cmake -S "$source_root/scm/src" -B "$build" \
    -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$build/compat.cmake" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCCPP_SUITES=SCM_saved_drag \
    -DMPI_Fortran_COMPILER="$prefix/bin/mpifort" -DMPI_C_COMPILER="$prefix/bin/mpicc" \
    -DCMAKE_PREFIX_PATH="$prefix"
nice -n 10 cmake --build "$build" -j4
(cd "$source_root/ccpp/physics" && git diff --exit-code)
sha256sum "$build/scm" "$build/ccpp/physics/libccpp_physics.a"
