"""Diagnostic build only: dump WRF's GS variables at the update point.

  instrument_gs.py WRF.f90 FUSED_GS.cu OUT.f90 OUT_NAMES.json

Inserts, in nssl_2mom_gs, just before the loop that adds
dtp*(pqwvi+pqwvd) to qwvp (module_mp_nssl_2mom.F:23049-23056: after every
rate, limiter, source/sink sum and the latent-heating update), one call per
gather point:

  call n2m_gs_dump(igs(mgs), kgs(mgs), N, (/ real(v1(mgs)), ... /))

for every gather array that WOOF's nssl2_gs_point also holds under the same
name at the same point (gs_names.shared_names).  No executable WRF statement
changes.  Also inserts the three stage-dump calls (instrument.py) so the
column counter in gs_dump.F90 advances with stage 1.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from gs_names import shared_names

src, cu, dst, names_out = sys.argv[1:5]
here = Path(__file__).resolve().parent
tmp = Path(dst).with_suffix(".stages.f90")
subprocess.check_call([sys.executable, str(here / "instrument.py"), src, str(tmp)])
lines = tmp.read_text().split("\n")
start = next(i for i, l in enumerate(lines) if re.match(r"\s*subroutine nssl_2mom_gs\b", l, re.I))
end = next(i for i in range(start, len(lines))
           if re.match(r"\s*end subroutine nssl_2mom_gs", lines[i], re.I))
body = "\n".join(lines[start:end]).lower()
names = shared_names(Path(cu), body)
anchor = next(i for i in range(start, end)
              if re.match(r"\s*qwvp\(mgs\) = qwvp\(mgs\) \+\s*&\s*$", lines[i]))
loop = max(i for i in range(start, anchor)
           if re.match(r"\s*do mgs = 1,ngscnt\s*$", lines[i], re.I))
vals = ", ".join(f"real({n.lower()}(mgs))" for n in names)
lines[loop:loop] = [
    "      do mgs = 1,ngscnt",
    f"        call n2m_gs_dump(igs(mgs), kgs(mgs), {len(names)}, (/ {vals} /))",
    "      end do",
]
Path(dst).write_text("\n".join(lines))
Path(names_out).write_text(json.dumps({"names": names}, indent=1))
print(f"instrumented {dst}: {len(names)} variables at line {loop + 1}")
