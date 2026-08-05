#!/bin/bash
# Stage and run one WRF em_les case.
#   run_les.sh <run_id> <namelist_file> [nranks] [sounding] [iofields]
# The pinned campaign build is never referenced by this script.
#
# The iofields file is an argument as of P1: a moist run needs the moist rows
# in stream 0 and the dry file must stay byte-identical, so the two travel
# separately instead of one being edited.  Defaulting it preserves every
# existing call site exactly.
B=${LES_ORACLE_ROOT:-$HOME/weather/les-oracle-wpl7}
# LES_ORACLE_SRC overrides the build tree.  P1's secondary arm is
# em_quarter_ss, which needs the wrf.exe built against the em_quarter_ss
# Registry; its ideal.exe is byte-identical to the em_les one because v4.6.1
# selects the ideal case from &ideal/ideal_case at runtime
# (dyn_em/module_initialize_ideal.F:181 `SELECT CASE (ideal_case)`), not at
# compile time.
SRC=${LES_ORACLE_SRC:-$B/src/WRFV4.6.1}
RID=$1
NML=$(readlink -f "$2")
NR=${3:-24}
SND=$(readlink -f "${4:-$B/assets/input_sounding.arwen_cbl}")
IOF=$(readlink -f "${5:-$B/assets/iofields_les.txt}")
RD=$B/runs/$RID
if [ ! -f "$NML" ]; then echo "MISSING NAMELIST: $2"; exit 2; fi
if [ ! -f "$SND" ]; then echo "MISSING SOUNDING: ${4:-default}"; exit 2; fi
if [ ! -f "$IOF" ]; then echo "MISSING IOFIELDS: ${5:-default}"; exit 2; fi
# The namelist names the iofields file it expects; staging a different one
# under a different name is how a run silently loses its SGS rows.
IOF_WANT=$(sed -n 's/.*iofields_filename *= *"\([^"]*\)".*/\1/p' "$NML" | head -1)
if [ -n "$IOF_WANT" ] && [ "$IOF_WANT" != "$(basename "$IOF")" ]; then
  echo "IOFIELDS MISMATCH: namelist wants $IOF_WANT, staging $(basename "$IOF")"
  exit 2
fi

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
cp "$IOF" "$RD"/

# provenance for this run
{
  echo "run_id: $RID"
  echo "host: $(hostname)"
  echo "src_tree: $SRC"
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nranks: $NR"
  echo "ideal.exe sha256: $(sha256sum ideal.exe | cut -d" " -f1)"
  echo "wrf.exe   sha256: $(sha256sum wrf.exe   | cut -d" " -f1)"
  echo "namelist  sha256: $(sha256sum namelist.input | cut -d" " -f1)"
  echo "sounding  sha256: $(sha256sum input_sounding | cut -d" " -f1)"
  echo "sounding  source: $SND"
  echo "iofields  file:   $(basename "$IOF")"
  echo "iofields  sha256: $(sha256sum "$(basename "$IOF")" | cut -d" " -f1)"
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
