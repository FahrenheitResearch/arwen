#!/usr/bin/env bash
set -euo pipefail
L=/work/pool-sfclay-noftz-mynn-mm5-eta
T=$L/src
source /work/env.sh
source /work/wrf/env.sh
export PYTHONPATH=$T:$T/tests CUPY_CACHE_DIR=$L/cupy-cache
mkdir -p "$L/receipts" "$L/oracles"
printf '%s\n' "$L/receipts" "$L/subnormal" "$L/extra" "$L/extras" >> "$L/MANIFEST.txt"
bash "$T/tools/mynn_sfclay_wrf461_column_oracle/build.sh" "$WRF_SOURCE_ROOT" "$L/oracles/mynn"
bash "$T/tools/sfclay_classic_wrf461_oracle/build.sh" "$WRF_SOURCE_ROOT" "$L/oracles/classic"
bash "$T/tools/sfclayrev_wrf461_oracle/build.sh" "$WRF_SOURCE_ROOT" "$L/oracles/revised"
bash "$T/tools/myjsfc_wrf461_oracle/build.sh" "$WRF_SOURCE_ROOT" "$L/oracles/eta"
curl --fail --location --retry 2 https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21/sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_bl_mynn.F -o "$L/oracles/fork_mynn.F"
bash "$T/tools/mynn_pbl_gsd41_oracle/build_columns.sh" "$L/oracles/fork_mynn.F" "$L/oracles/pbl"
python "$T/tools/sfclay_noftz_check/prepare.py"
python "$T/tools/sfclay_noftz_check/mynn_surface.py" subnormal prepare
python "$T/tools/sfclay_noftz_check/pbl_prepare.py"
python "$T/tools/sfclay_noftz_check/eta_prepare.py"
