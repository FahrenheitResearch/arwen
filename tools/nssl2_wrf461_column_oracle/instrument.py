"""Make a stage-dumping copy of WRF's preprocessed module_mp_nssl_2mom.f90.

Diagnostic only, never the reference.  Three CALL statements are inserted in
nssl_2mom_driver; no executable WRF statement is changed:

  stage 1  just before ``call nssl_2mom_gs``   (after gather, sedimentation
                                                and the precipitation sums)
  stage 2  just before ``CALL NUCOND``          (after the GS process step)
  stage 3  just before the ``ENDDO`` closing the loopcnt loop, i.e. the
           first bare ``ENDDO`` after the NUCOND call (WRF's preprocessing
           strips the ``! loopcnt`` comment that marks it in the .F)

Each dump appends the driver's concentration-space slab ``an`` and its
temperature ``t0`` and primary-ice target ``t7`` (see stage_dump.F90).  run_oracle.sh checks that the
instrumented build's final output is byte-identical to the scalar build,
which proves the dumps did not perturb the arithmetic.
"""
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().split("\n")
idx = ("(/ lt,lv,lc,lr,li,ls,lh,lhl,lccn,lnc,lnr,lni,lns,lnh,lnhl,lvh,lvhl /)")
marks = [
    (re.compile(r"^\s*call nssl_2mom_gs\s*&\s*$"), 1),
    (re.compile(r"^\s*CALL NUCOND\s*&\s*$"), 2),
    (re.compile(r"^\s*ENDDO\s*(! loopcnt=1,loopmax)?\s*$"), 3),
]
out, seen = [], set()
for line in lines:
    for pattern, stage in marks:
        if stage == 3 and (2 not in seen or 3 in seen):
            continue
        if pattern.match(line):
            if stage in seen:
                sys.exit(f"stage {stage} anchor found twice")
            seen.add(stage)
            out.append(f"      call n2m_stage_dump({stage}, ite-its+1, kte-kts+1, na, an, t0, t7, {idx})")
    out.append(line)
if seen != {1, 2, 3}:
    sys.exit(f"anchors found: {sorted(seen)}")
open(dst, "w").write("\n".join(out))
print("instrumented", dst)
