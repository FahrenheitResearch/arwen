#!/bin/sh
# Source-level branch coverage for the BARE_FLUX oracle case deck.
#
# Usage:  branch_coverage_bareflux.sh <wrf-tree> <workdir> <case-deck>
#
# Builds the visibility-patched module with gcov instrumentation, runs the
# committed case deck through it, and reports every source line and every
# branch leg of BARE_FLUX / SFCDIF1 / ESAT that the deck does NOT reach.  This
# is what turns "these cases bind those branches" from a claim into evidence.
#
# Result on the 27-case deck (gfortran 13.3.0, -O0 --coverage):
#
#   BARE_FLUX  77 lines executed, 6 not, 8 untaken branch legs
#       every one of them is an option-identity leg:
#       4359 OPT_SFC==1 false leg          4444 OPT_STC==1.OR.==3 false leg
#       4367 OPT_SFC==2 true leg           4446 OPT_STC==1 false leg
#       4371-4378 the SFCDIF2 block        4447 OPT_STC==3 true leg
#       (call tail, CH/CM rescale,         4462 OPT_SFC==1.OR.==2 false leg
#        SNOWH>0 clamp and its legs)
#       -> no data-dependent branch of BARE_FLUX is unbound.
#
#   SFCDIF1    67 lines executed, 2 not, 3 untaken branch legs
#       4647-4649  the ZLVL<=ZPD abort.  Deliberately unbound: WRF calls
#                  wrf_error_fatal there.  The transcription raises instead.
#       4732 ABS(CM2FM2)<=MPE, 4733 ABS(CH2FH2)<=MPE.  Unbindable *through
#                  BARE_FLUX*, not merely unbound: CM2FM2 feeds only the
#                  commented-out CH2 formula on line 4736, and CH2FH2 feeds
#                  only CH2, which BARE_FLUX declares, passes as INTENT(OUT)
#                  and never reads.  Both are still transcribed.
#
#   ESAT       6 lines executed, 0 not, 0 untaken branch legs (fully covered)
set -eu

WRF_TREE=$1
WORK=$2
DECK=$3
HERE=$(cd "$(dirname "$0")" && pwd)

rm -rf "$WORK"
mkdir -p "$WORK"
cd "$WORK"

cp "$WRF_TREE/phys/module_sf_gecros.F" .
python3 "$HERE/visibility_patch_leaves.py" \
        "$WRF_TREE/phys/module_sf_noahmplsm.F" \
        --out module_sf_noahmplsm_public.F --check >/dev/null

cat > stubs.f90 <<'STUB'
subroutine wrf_error_fatal(m)
  character(len=*), intent(in) :: m
  write(0,*) m
  stop 1
end subroutine wrf_error_fatal
subroutine wrf_message(m)
  character(len=*), intent(in) :: m
end subroutine wrf_message
subroutine wrf_debug(l, m)
  integer, intent(in) :: l
  character(len=*), intent(in) :: m
end subroutine wrf_debug
STUB

F="-w -ffree-form -ffree-line-length-none -O0 --coverage"
gfortran -c $F module_sf_gecros.F -o gecros.o
gfortran -c $F module_sf_noahmplsm_public.F -o mod.o
gfortran -c -w -O0 stubs.f90 -o stubs.o
gfortran -c $F "$HERE/run_bareflux.F90" -o drv.o
gfortran --coverage -o run drv.o mod.o gecros.o stubs.o -lm

./run < "$DECK" > /dev/null
gcov -b -o . mod.gcno >/dev/null 2>&1

python3 - <<'PY'
import re
recs = []
branches = {}
cur = None
with open("module_sf_noahmplsm_public.F.gcov", encoding="latin-1") as fh:
    for line in fh:
        s = line.strip()
        if s.startswith("branch ") and cur is not None:
            branches.setdefault(cur, []).append(s)
            continue
        m = re.match(r"^\s*([^:]+):\s*(\d+):(.*)$", line.rstrip("\n"))
        if m:
            cur = int(m.group(2))
            recs.append((m.group(1).strip(), cur, m.group(3)))

def report(name, lo, hi):
    unexec, taken, untaken = [], 0, []
    for count, n, src in recs:
        if not (lo <= n <= hi):
            continue
        if count == "#####":
            unexec.append((n, src.strip()))
        elif count != "-":
            taken += 1
        for b in branches.get(n, []):
            if "never executed" in b or "taken 0%" in b:
                untaken.append((n, src.strip(), b))
    print(f"== {name}: lines {lo}-{hi}: executed {taken}, "
          f"not executed {len(unexec)}, untaken branch legs {len(untaken)}")
    for n, s in unexec:
        print(f"   unexecuted {n}: {s[:96]}")
    for n, s, b in untaken:
        print(f"   untaken    {n}: {s[:76]}  [{b}]")

report("BARE_FLUX", 4174, 4479)
report("SFCDIF1", 4583, 4743)
report("ESAT", 4952, 5001)
PY
