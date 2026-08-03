#!/bin/bash
# Stage and run one WRF em_les case.
#   run_les.sh <run_id> <namelist_file> [nranks] [sounding]
# The pinned campaign build is never referenced by this script.
B=${LES_ORACLE_ROOT:-$HOME/weather/les-oracle-wpl7}
SRC=$B/src/WRFV4.6.1
RID=$1
NML=$(readlink -f "$2")
NR=${3:-24}
SND=$(readlink -f "${4:-$B/assets/input_sounding.arwen_cbl}")
RD=$B/runs/$RID
if [ ! -f "$NML" ]; then echo "MISSING NAMELIST: $2"; exit 2; fi
if [ ! -f "$SND" ]; then echo "MISSING SOUNDING: ${4:-default}"; exit 2; fi

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1

rm -rf "$RD"
mkdir -p "$RD"
cd "$RD" || exit 1

# Stage the run directory. cp -L dereferences relative symlinks (MPTABLE.TBL
# is one; a plain cp leaves a dangling link and Noah-MP dies in
# read_mp_veg_parameters with an error that looks like physics and is not).
cp -L "$SRC"/run/* "$RD"/ 2>/dev/null
cp "$SRC"/main/ideal.exe "$SRC"/main/wrf.exe "$RD"/
cp "$NML" "$RD"/namelist.input
cp "$SND" "$RD"/input_sounding
cp "$B"/assets/iofields_les.txt "$RD"/

# provenance for this run
{
  echo "run_id: $RID"
  echo "host: $(hostname)"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nranks: $NR"
  echo "ideal.exe sha256: $(sha256sum ideal.exe | cut -d" " -f1)"
  echo "wrf.exe   sha256: $(sha256sum wrf.exe   | cut -d" " -f1)"
  echo "namelist  sha256: $(sha256sum namelist.input | cut -d" " -f1)"
  echo "sounding  sha256: $(sha256sum input_sounding | cut -d" " -f1)"
  echo "iofields  sha256: $(sha256sum iofields_les.txt | cut -d" " -f1)"
} > run_provenance.txt

echo "=== ideal.exe (1 rank) ==="
t0=$(date +%s)
mpirun -n 1 ./ideal.exe > ideal.stdout 2>&1
IRC=$?
t1=$(date +%s)
echo "ideal_rc=$IRC wall=$((t1-t0))s"
tail -3 rsl.error.0000 2>/dev/null
if [ ! -f wrfinput_d01 ]; then
  echo "IDEAL_FAILED: no wrfinput_d01"
  exit 3
fi
mv rsl.out.0000 ideal.rsl.out.0000 2>/dev/null
mv rsl.error.0000 ideal.rsl.error.0000 2>/dev/null

echo "=== wrf.exe ($NR ranks) ==="
t2=$(date +%s)
mpirun -n "$NR" ./wrf.exe > wrf.stdout 2>&1
WRC=$?
t3=$(date +%s)
echo "wrf_rc=$WRC wall=$((t3-t2))s"
tail -4 rsl.error.0000 2>/dev/null

{
  echo "ideal_rc: $IRC"
  echo "ideal_wall_s: $((t1-t0))"
  echo "wrf_rc: $WRC"
  echo "wrf_wall_s: $((t3-t2))"
  echo "wrfout_bytes: $(du -cb wrfout_d01_* 2>/dev/null | tail -1 | cut -f1)"
  echo "wrfout_files: $(ls wrfout_d01_* 2>/dev/null | wc -l)"
} >> run_provenance.txt

# keep the logs, drop the per-rank chatter
rm -f rsl.out.00[0-9][1-9] rsl.error.00[0-9][1-9] 2>/dev/null
ls -la wrfout_d01_* 2>/dev/null | head -3
echo "RUN_DONE $RID rc=$WRC"
