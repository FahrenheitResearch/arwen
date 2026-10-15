"""Write the oracle's DEFINED arm of WRF v4.6.1 phys/module_mp_wdm6.F.

usage: python3 defined_arm.py PRISTINE_module_mp_wdm6.F OUTPUT.F [SECTIONS]

SECTIONS is a comma list drawn from 2,5,6 (default all three).  The
single-section arms attribute the stock comparison's differences to the
divergence that produced them; the arm of record is all three.

The defined arm is the byte-frozen WRF source with exactly the three places
where WOOF deliberately implements defined behaviour instead of WRF's
(docs/wdm6_oracle_known_deltas.md) replaced by that behaviour, written in
WRF's own Fortran beside WRF's own code.  Nothing else changes.  Each
substitution must match its anchor exactly once, so a different WRF file
cannot be patched silently.

  section 5  rain mass/number sedimentation (:795-854): one donor-limited
             transfer per interface and the geometric sub-step schedule,
             instead of WRF's arrival cap by the already-updated donor
             (which discards rain) and its initial-speed sub-step count.
  section 6  the condensation cap at :1255 keeps the rate's sign:
             max(min(prevp, satdt/2), 0) instead of min(prevp, satdt/2),
             which turns numberless rain (rate -0) into unbounded
             evaporation of half the saturation deficit.
  section 2  the PLM remap's `kt = kt - 1` (:2629, :2891) is clamped at
             the first layer instead of falling below kb's floor and
             dropping a layer's mass.

The stock arm (the unmodified file) is the comparison of record; the
defined arm shows what remains once these three documented divergences are
set aside, so a residual there is a port defect and not one of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DECL_ANCHOR = "   real                               :: gfac, sfac\n"
_DECL_ADD = (
    "   real :: wdm6d_df, wdm6d_air, wdm6d_mass, wdm6d_num, wdm6d_outm,     &\n"
    "           wdm6d_outn, wdm6d_inm, wdm6d_inn, wdm6d_dts, wdm6d_x,       &\n"
    "           wdm6d_vtb\n"
    "   integer :: wdm6d_steps\n")

_RAIN_START = "      mstepmax = 1\n      numdt = 1\n"
_RAIN_END = "!\n! for semi\n"

_RAIN_DEFINED = """\
! WDM6-DEFINED (docs/wdm6_oracle_known_deltas.md section 5): geometric
! sub-step count from the 1e-3 m inverse-slope cap, and one donor-limited
! transfer per interface, as WOOF's wdm6_rain_steps / wdm6_rain_substep.
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
      do i = its, ite
        if(mstepmax.le.mstep(i)) mstepmax = mstep(i)
      enddo
!
      do n = 1, mstepmax
        do i = its, ite
          if(n.le.mstep(i)) then
            wdm6d_dts = dtcld/real(mstep(i))
            wdm6d_inm = 0.
            wdm6d_inn = 0.
            do k = kte, kts, -1
              wdm6d_air = den(i,k)*delz(i,k)
              wdm6d_mass = qrs(i,k,1)*wdm6d_air
              wdm6d_num = ncr(i,k,3)*delz(i,k)
              wdm6d_outm = min(wdm6d_mass, wdm6d_mass*work1(i,k,1)*wdm6d_dts)
              wdm6d_outn = min(wdm6d_num, wdm6d_num*workn(i,k)*wdm6d_dts)
              qrs(i,k,1) = (wdm6d_mass - wdm6d_outm + wdm6d_inm)/wdm6d_air
              ncr(i,k,3) = (wdm6d_num - wdm6d_outn + wdm6d_inn)/delz(i,k)
              wdm6d_inm = wdm6d_outm
              wdm6d_inn = wdm6d_outn
            enddo
            fall(i,kts,1) = fall(i,kts,1) + wdm6d_inm/delz(i,kts)/dtcld
          endif
        enddo
        do k = kts, kte
          do i = its, ite
            qrs_tmp(i,k,1) = qrs(i,k,1)
            ncr_tmp(i,k) = ncr(i,k,3)
          enddo
        enddo
        call slope_rain(qrs_tmp,ncr_tmp,den_tmp,denfac,t,rslope,rslopeb,rslope2, &
                     rslope3,work1,workn,its,ite,kts,kte)
        do k = kte, kts, -1
          do i = its, ite
            work1(i,k,1) = work1(i,k,1)/delz(i,k)
            workn(i,k) = workn(i,k)/delz(i,k)
          enddo
        enddo
      enddo
"""

_PREVP_WRF = "              prevp(i,k) = min(prevp(i,k),satdt/2)\n"
_PREVP_DEFINED = (
    "! WDM6-DEFINED (docs/wdm6_oracle_known_deltas.md section 6)\n"
    "              prevp(i,k) = max(min(prevp(i,k),satdt/2),0.)\n")

_KT_WRF = "               kt = kt - 1\n"
_KT_DEFINED = (
    "! WDM6-DEFINED (docs/wdm6_oracle_known_deltas.md section 2)\n"
    "               kt = max(kt - 1, 1)\n")


def defined_source(text: str, sections=(2, 5, 6)) -> str:
    def once(haystack: str, needle: str, expected: int = 1) -> None:
        found = haystack.count(needle)
        if found != expected:
            raise SystemExit(f"anchor found {found} times, expected "
                             f"{expected}: {needle!r}")

    sections = set(sections)
    if not sections <= {2, 5, 6}:
        raise SystemExit(f"unknown section in {sorted(sections)}")
    once(text, _DECL_ANCHOR)
    once(text, _RAIN_START)
    once(text, _PREVP_WRF)
    once(text, _KT_WRF, expected=2)
    if 5 in sections:
        text = text.replace(_DECL_ANCHOR, _DECL_ANCHOR + _DECL_ADD)
        start = text.index(_RAIN_START)
        end = text.index(_RAIN_END, start)
        text = text[:start] + _RAIN_DEFINED + text[end:]
    if 6 in sections:
        text = text.replace(_PREVP_WRF, _PREVP_DEFINED)
    if 2 in sections:
        text = text.replace(_KT_WRF, _KT_DEFINED)
    return text


def main(argv: list[str]) -> int:
    if len(argv) not in (3, 4):
        print(__doc__, file=sys.stderr)
        return 2
    sections = (2, 5, 6) if len(argv) == 3 else tuple(
        int(v) for v in argv[3].split(","))
    source = Path(argv[1]).read_text(encoding="ascii")
    Path(argv[2]).write_text(defined_source(source, sections),
                             encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
