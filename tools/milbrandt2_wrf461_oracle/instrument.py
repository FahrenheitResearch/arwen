"""Make a DIAGNOSTIC copy of module_mp_milbrandt2mom.F that dumps its state.

    python instrument.py PRISTINE.F OUT.f90

Never the oracle: the oracle is the byte-unmodified object.  This copy adds
only CALL statements to two external subroutines (my2dump.F90) at the
scheme's own ``check_values`` checkpoints -- the part boundaries and the
per-species sedimentation calls -- plus one call that dumps the SAVE
constants and one that dumps the reflectivity factors, so a mismatch can be
pinned to the part that first produced it.  run_oracle.sh checks that the
instrumented build's final output is bit-identical to the oracle's, so the
calls are shown not to move any arithmetic.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gpuwm.core.milbrandt2_constants import CK_ORDER  # noqa: E402

LATE = ("cxr", "cxi", "Gzr", "Gzi", "Gzs", "Gzg", "Gzh")
STATE = "T,Q,QC,QR,QI,QN,QG,QH,NC,NR,NY,NN,NG,NH"
AUX = "DE,QSW,QSI,pres,gamfact,DZ,iDZ"


def _ctor(names):
    body = ", ".join(names)
    # gfortran's free form has no line-length limit under
    # -ffree-line-length-none, so one constructor line is fine.
    return f"(/ {body} /)"


def main(src, dst):
    text = Path(src).read_text()
    lines = text.splitlines()
    out = []
    early = [n for n in CK_ORDER if n not in LATE]
    for line in lines:
        m = re.match(r"\s*!?\s*if \(DEBUG_ON\) call check_values\(Q,T,QC.*,(\d+)\)\s*$",
                     line)
        if m and not line.lstrip().startswith("!") and int(m.group(1)) >= 300:
            tag = int(m.group(1))
            out.append(f"  call my2dump_state({tag}, size(QC,dim=1), size(QC,dim=2), "
                       f"{STATE}, {AUX})")
        out.append(line)
        if line.strip() == "endif  !if (KOUNT=0)":
            out.append(f"  call my2dump_const(1, {len(early)}, {_ctor(early)})")
        if line.strip().startswith("Gzh= (6.+alpha_h)"):
            out.append(f"     call my2dump_const(2, {len(LATE)}, {_ctor(LATE)})")
    Path(dst).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
