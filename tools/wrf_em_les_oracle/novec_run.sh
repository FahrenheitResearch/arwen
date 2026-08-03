#!/bin/bash
# Toolchain-robustness arm for the em_les oracle.
#
# Framing matters here. gfortran seeds random_number from OS entropy when
# RANDOM_SEED is never called, and WRF's ideal case never calls it, so two
# em_les runs of one configuration on one node with one binary are already
# different realisations. That is proven, not assumed: on node-2 two
# identical-config runs differ by 4.27 percent in the entrainment minimum.
# It is therefore IMPOSSIBLE to isolate a compiler-flag effect from
# realisation noise with a single pair of runs.
#
# The answerable question is whether the -fno-tree-vectorize build lands
# inside the realisation spread already measured for the vectorised build.
# If it does, the flag effect on the reported statistics is smaller than the
# noise those statistics already carry, which is the honest and sufficient
# result for a chaotic LES.
B=${LES_ORACLE_ROOT:-$HOME/weather/les-oracle-wpl7}

# wait for the 50 m run to release the box, and for the novec build to exist
while ! grep -q "SUCCESS COMPLETE WRF" "$B/runs/match_km2_50m/rsl.error.0000" 2>/dev/null; do
  sleep 120
done
while [ ! -x "$B/src/WRFV4.6.1-novec/main/wrf.exe" ]; do
  sleep 120
done

sha256sum "$B/src/WRFV4.6.1-novec/main/ideal.exe" \
          "$B/src/WRFV4.6.1-novec/main/wrf.exe" \
          > "$B/provenance/novec_binaries.sha256"
cat "$B/provenance/novec_binaries.sha256"

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

# two realisations with the novec binary so it carries its own spread too
for N in 1 2; do
  R="novec_km3_100m_r$N"
  rm -rf "$B/runs/$R"
  mkdir -p "$B/runs/$R"
  cd "$B/runs/$R" || exit 1
  cp -L "$B"/src/WRFV4.6.1/run/* . 2>/dev/null
  cp "$B/src/WRFV4.6.1-novec/main/ideal.exe" .
  cp "$B/src/WRFV4.6.1-novec/main/wrf.exe" .
  cp "$B/assets/namelist.match_km3_100m" namelist.input
  cp "$B/assets/input_sounding.arwen_cbl" input_sounding
  cp "$B/assets/iofields_les.txt" .
  mpirun -n 1 ./ideal.exe > ideal.stdout 2>&1
  mpirun -n 24 ./wrf.exe > wrf.stdout 2>&1
  echo "novec run $N wrf_rc=$?"
  python3 "$B/assets/score_wrf_les.py" "$B/runs/$R" "$B/scores/$R" \
      --window-min 30 > "$B/scores/$R.txt" 2>&1
  tail -13 "$B/scores/$R.txt"
  echo "########## $R DONE $(date -u +%H:%M:%S) ##########"
done
echo "NOVEC_RUNS_DONE $(date -u +%H:%M:%S)"
