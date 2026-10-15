"""Write a measurement arm that splits section 5 of the defined arm in two.

usage: python3 split_arm.py PRISTINE_module_mp_wdm6.F OUTPUT.F [MODE]

MODE ``transfer`` (default) writes the arm described below; ``fineN`` (for
example ``fine8``) writes section 5 exactly as the defined arm has it but
with N times WOOF's sub-step count, a converged reference against which
both counts can be judged.

Section 5 of docs/wdm6_oracle_known_deltas.md changes two things in WRF's
rain sedimentation loop at once:

  (a) the transfer: one donor-limited amount per interface, which closes
      WRF's loss of rain (the arrival cap by the already-updated donor);
  (b) the sub-step count: WOOF's geometric count from the 1e-3 m inverse
      slope cap, instead of WRF's count from the initial fall speeds.

This arm keeps (a) and restores WRF's own count for (b), so comparing it
with the stock arm measures the transfer alone, and comparing it with the
``only5`` arm measures the count alone.  It is evidence only: WOOF's kernel
is graded against the defined arm (defined_arm.py), never against this one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defined_arm as da  # noqa: E402

_WOOF_COUNT = """\
      do i = its, ite
        mstep(i) = 1
        do k = kts, kte
          wdm6d_df = sqrt(1.28/den(i,k))
          wdm6d_x = 1.e-3
          wdm6d_vtb = wdm6d_x**0.8
          wdm6d_x = 2.5*2.99849272e3*wdm6d_vtb*wdm6d_df/delz(i,k)*dtcld + 1.
          wdm6d_steps = max(int(wdm6d_x), 1)
          if(wdm6d_steps.gt.mstep(i)) mstep(i) = wdm6d_steps
        enddo
      enddo
      do k = kte, kts, -1
        do i = its, ite
          work1(i,k,1) = work1(i,k,1)/delz(i,k)
          workn(i,k) = workn(i,k)/delz(i,k)
        enddo
      enddo
      mstepmax = 1
"""

# WRF v4.6.1 module_mp_wdm6.F:795-803, verbatim.
_WRF_COUNT = """\
      mstepmax = 1
      numdt = 1
      do k = kte, kts, -1
        do i = its, ite
          work1(i,k,1) = work1(i,k,1)/delz(i,k)
          workn(i,k) = workn(i,k)/delz(i,k)
          numdt(i) = max(nint(max(work1(i,k,1),workn(i,k))*dtcld+.5),1)
          if(numdt(i).ge.mstep(i)) mstep(i) = numdt(i)
        enddo
      enddo
"""


def transfer_only_source(text: str) -> str:
    out = da.defined_source(text, (5,))
    if out.count(_WOOF_COUNT) != 1:
        raise SystemExit("section 5 count block not found exactly once")
    out = out.replace(_WOOF_COUNT, _WRF_COUNT)
    return out.replace(
        "! WDM6-DEFINED (docs/wdm6_oracle_known_deltas.md section 5): geometric\n"
        "! sub-step count from the 1e-3 m inverse-slope cap, and one donor-limited\n"
        "! transfer per interface, as WOOF's wdm6_rain_steps / wdm6_rain_substep.\n",
        "! SPLIT ARM: section 5's donor-limited transfer with WRF's own count.\n")


_STEPS = "          wdm6d_steps = max(int(wdm6d_x), 1)\n"


def fine_source(text: str, factor: int) -> str:
    out = da.defined_source(text, (5,))
    if out.count(_STEPS) != 1 or factor < 1:
        raise SystemExit("section 5 step statement not found exactly once")
    return out.replace(
        _STEPS, f"          wdm6d_steps = {factor}*max(int(wdm6d_x), 1)\n")


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    mode = argv[3] if len(argv) == 4 else "transfer"
    source = Path(argv[1]).read_text(encoding="ascii")
    if mode == "transfer":
        out = transfer_only_source(source)
    elif mode.startswith("fine") and mode[4:].isdigit():
        out = fine_source(source, int(mode[4:]))
    else:
        print(__doc__, file=sys.stderr)
        return 2
    Path(argv[2]).write_text(out, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
