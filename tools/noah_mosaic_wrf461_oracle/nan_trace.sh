#!/usr/bin/env bash
# Diagnostic only, never an oracle: trace WRF's subnormal-CHS probe column.
# Copies a finished build.sh output, recompiles a COPY of module_sf_noahlsm.F
# with PRINT statements that fire only when 0 < CH < 1E-37 (the ftz family's
# CHS probe), relinks run_mosaic and runs it. The pristine tree is read only.
# Usage: nan_trace.sh WRF_SOURCE_ROOT BUILD_DIR OUT_DIR   -> OUT_DIR/trace.txt
set -euo pipefail
src=$(realpath "$1"); build=$(realpath "$2"); out=$(realpath -m "$3")
rm -rf "${out}"; mkdir -p "${out}"; cp -a "${build}/." "${out}/"; rm -rf "${out}/fixtures"
cd "${out}"
python3 - "${src}/phys/module_sf_noahlsm.F" module_sf_noahlsm_trace.F <<'PY'
import sys
src = open(sys.argv[1]).read().split("\n")
def find(pat, start=0):
    for i in range(start, len(src)):
        if pat in src[i]:
            return i
    raise SystemExit("missing " + pat)
ins = {}
def after(i, line):
    ins.setdefault(i, []).append(line)
after(find("CONTAINS") - 1, "      LOGICAL :: NMDBG = .FALSE.")
p = find("CALL PENMAN (SFCTMP,SFCPRS,CH,T2V")
after(p - 1, "         NMDBG = (CH > 0. .AND. CH < 1.E-37)")
after(p + 3, '         IF (NMDBG) PRINT *, "PENMAN ETP RCH RR EPSCA", ETP, RCH, RR, EPSCA')
after(find("RCS,RCT,RCQ,RCSOIL,EMISSI)", p) + 3, '         IF (NMDBG) PRINT *, "CANRES PC RC", PC, RC')
e = find("ETA_KINEMATIC =  ESNOW + ETNS - 1000.0*DEW", p)
after(e + 1, '         IF (NMDBG) PRINT *, "POST ETA ETA_KINEMATIC T1", ETA, ETA_KINEMATIC, T1')
after(e + 1, '         IF (NMDBG) PRINT *, "POST SMC", SMC, " STC", STC')
s = find("SUBROUTINE SMFLX")
after(find("CALL FAC2MIT(SMCMAX,FLIMIT)", s), '      IF (NMDBG) PRINT *, "SMFLX PCPDRP SMC", PCPDRP, SMC')
ss = find("SUBROUTINE SSTEP")
after(find("SH2OOUT (K) = MAX ( (SMC (K) - SICE (K)),0.0)", ss),
      '         IF (NMDBG) PRINT *, "SSTEP K STOT SMC", K, STOT, SMC(K)')
out = []
for i, line in enumerate(src):
    out.append(line)
    out.extend(ins.get(i, []))
open(sys.argv[2], "w").write("\n".join(out))
PY
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
fc="gfortran -c -O0 -cpp ${defines} -ffree-form -ffree-line-length-none -fallow-argument-mismatch -I ${out}"
${fc} -o module_sf_noahlsm.o module_sf_noahlsm_trace.F
${fc} -o module_sf_noahlsm_glacial_only.o "${src}/phys/module_sf_noahlsm_glacial_only.F"
${fc} -o module_sf_noahdrv.o "${src}/phys/module_sf_noahdrv.F" 2>/dev/null
objs=(stub_wrf.o module_model_constants.o module_wrf_error.o module_sf_urban.o
      module_bep_bem_helper.o module_sf_bep.o module_sf_bem.o module_sf_bep_bem.o
      module_sf_noahlsm.o module_sf_noahlsm_glacial_only.o module_ra_gfdleta_cal_mon_day.o
      module_sf_noahdrv.o oracle_io.o)
[[ -f cr_libm_control.o ]] && objs+=(cr_libm_control.o)
gfortran -o run_mosaic_trace "${objs[@]}" run_mosaic.o
mkdir -p fx
./run_mosaic_trace fx | grep -E '^ *(PENMAN|CANRES|POST|SMFLX|SSTEP)' > trace.txt
echo "trace in ${out}/trace.txt"
