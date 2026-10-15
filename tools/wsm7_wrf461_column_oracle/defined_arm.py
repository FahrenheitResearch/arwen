"""Write the DEFINED and PROBE arms of WRF v4.6.1 phys/module_mp_wsm7.F.

usage: python3 defined_arm.py PRISTINE_module_mp_wsm7.F OUTPUT.F ARM

ARM is one of

  defined   the byte-frozen WRF source with exactly the two places where
            WOOF implements defined behaviour instead of WRF's
            (docs/wsm7-mp24.md) replaced by that behaviour, written in WRF's
            own Fortran beside WRF's own code.  Nothing else changes.
              D1  nislfv_rain_plm/plm6 find_kt reaches the last arrival
                  interface za(km+1) and, past it, the column top, and kt is
                  never left below kb.  WRF's loop stops at za(km); a target
                  layer whose top lies above every interface it tests falls
                  through with kt = kb - 1 or worse and its mass is dropped
                  (queue 073, the same search in WSM6 and WDM6).
              D2  refl10cm_wsm7 adds hail's dry-ice Rayleigh term (WSM7's own
                  hail PSD: n0h, denh, bm = 3, mu = 0, so graupel's radar_init
                  exponent words), so a hail core shows in REFL_10CM.
  d1        D1 alone (attribution: the stock comparison's state differences)
  probe     the UNMODIFIED arithmetic plus a per-column counter of find_kt
            fall-throughs that leave arrival mass above kb: the
            probe-identified set D1 can change.  build.sh checks that this
            arm's outputs equal the stock arm's word for word.

Each substitution must match its anchor the stated number of times, so a
different WRF file cannot be patched silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

_KT_WRF = """\
               find_kt : do kk=kt,km
                         if( zi(k+1).le.za(kk) ) then
                           kt = kk
                           exit find_kt
                         else
                           cycle find_kt
                         endif
               enddo find_kt
               kt = kt - 1
"""
_KT_DEFINED = """\
! WSM7-DEFINED D1 (docs/wsm7-mp24.md): the search reaches za(km+1) and,
! past it, the column top zi(km+1), so kt never falls below kb.
               ktfound = km+2
               find_kt : do kk=kt,km+1
                         if( zi(k+1).le.za(kk) ) then
                           ktfound = kk
                           exit find_kt
                         endif
               enddo find_kt
               kt = max(ktfound - 1, kb)
"""
_KT_PROBE = _KT_WRF.replace(
    "               enddo find_kt\n",
    "               enddo find_kt\n"
    "! WSM7-PROBE: a fall-through with arrival mass from kb up\n"
    "               if( kk.gt.km ) then\n"
    "                 if( sum(qa(kb:km)).gt.0.0 ) then\n"
    "                   wsm7probe_ft(i) = wsm7probe_ft(i) + 1\n"
    "                 endif\n"
    "               endif\n")

_DECL_PLM = "      integer  i,k,n,m,kk,kb,kt,iter\n"
_DECL_PLM6 = "      integer  i,k,n,m,kk,kb,kt,iter,ist\n"
_DECL_ADD = "      integer  ktfound\n"

_MODULE_CONTAINS = "             rsloper3max,rslopes3max,rslopeg3max,rslopeh3max\nCONTAINS\n"
_PROBE_DECL = ("             rsloper3max,rslopes3max,rslopeg3max,rslopeh3max\n"
               "   INTEGER, PUBLIC, SAVE :: wsm7probe_ft(65536) = 0\n"
               "CONTAINS\n")

# D2: the wrapper hands refl10cm_wsm7 the hail column too.
_WRAP_DECL = "      REAL, DIMENSION(kts:kte):: qv1d, t1d, p1d, qr1d, qs1d, qg1d, dBZ\n"
_WRAP_DECL_D2 = ("      REAL, DIMENSION(kts:kte):: qv1d, t1d, p1d, qr1d, qs1d, qg1d, dBZ\n"
                 "      REAL, DIMENSION(kts:kte):: qh1d\n")
_WRAP_FILL = "                  qg1d(k)=qg(i,k,j)\n"
_WRAP_FILL_D2 = ("                  qg1d(k)=qg(i,k,j)\n"
                 "                  qh1d(k)=qh(i,k,j)\n")
_WRAP_CALL = """\
               call refl10cm_wsm7 (qv1d, qr1d, qs1d, qg1d,              &
                       t1d, p1d, dBZ, kts, kte, i, j)
"""
_WRAP_CALL_D2 = """\
               call refl10cm_wsm7 (qv1d, qr1d, qs1d, qg1d, qh1d,        &
                       t1d, p1d, dBZ, kts, kte, i, j)
"""
_REFL_HEAD = """\
      subroutine refl10cm_wsm7 (qv1d, qr1d, qs1d, qg1d,                 &
                       t1d, p1d, dBZ, kts, kte, ii, jj)
"""
_REFL_HEAD_D2 = """\
      subroutine refl10cm_wsm7 (qv1d, qr1d, qs1d, qg1d, qh1d,           &
                       t1d, p1d, dBZ, kts, kte, ii, jj)
"""
_REFL_ARGS = """\
      REAL, DIMENSION(kts:kte), INTENT(IN)::                            &
                      qv1d, qr1d, qs1d, qg1d, t1d, p1d
"""
_REFL_ARGS_D2 = """\
      REAL, DIMENSION(kts:kte), INTENT(IN)::                            &
                      qv1d, qr1d, qs1d, qg1d, t1d, p1d
      REAL, DIMENSION(kts:kte), INTENT(IN):: qh1d
      REAL, DIMENSION(kts:kte):: ze_hail
      REAL:: xam_h, rh_k
      DOUBLE PRECISION:: lamh, ilamh_k
"""
_REFL_GRAUPEL = """\
         if (qg1d(k) .gt. 1.E-9) then
            rg(k) = qg1d(k)*rho(k)
            N0_g(k) = n0g
            lamg = (xam_g*xcgg(3)*N0_g(k)/rg(k))**(1./xcge(1))
            ilamg(k) = 1./lamg
            L_qg(k) = .true.
         else
            rg(k) = 1.E-12
            L_qg(k) = .false.
         endif
"""
_REFL_GRAUPEL_D2 = _REFL_GRAUPEL + """\
! WSM7-DEFINED D2 (docs/wsm7-mp24.md): hail's dry-ice Rayleigh term, WSM7's
! hail PSD (n0h, denh; bm = 3, mu = 0, the exponent words of graupel's).
         ze_hail(k) = 0.
         if (qh1d(k) .gt. 1.E-9) then
            xam_h = PI*denh/6.
            rh_k = qh1d(k)*rho(k)
            lamh = (xam_h*xcgg(3)*DBLE(n0h)/rh_k)**(1./xcge(1))
            ilamh_k = 1./lamh
            ze_hail(k) = (0.176/0.93) * (6.0/PI)*(6.0/PI)               &
                       * (xam_h/900.0)*(xam_h/900.0)                    &
                       * DBLE(n0h)*xcgg(4)*ilamh_k**xcge(4)
         endif
"""
_REFL_SUM = ("         dBZ(k) = 10.*log10((ze_rain(k)+ze_snow(k)+ze_graupel(k))*1.d18)\n")
_REFL_SUM_D2 = ("! WSM7-DEFINED D2: ze_hail is 0. where there is no hail\n"
                "         dBZ(k) = 10.*log10((ze_rain(k)+ze_snow(k)+ze_graupel(k)"
                "+ze_hail(k))*1.d18)\n")


def _replace(text: str, old: str, new: str, expected: int = 1) -> str:
    found = text.count(old)
    if found != expected:
        raise SystemExit(f"anchor found {found} times, expected {expected}: "
                         f"{old[:80]!r}")
    return text.replace(old, new)


def arm_source(text: str, arm: str) -> str:
    if arm not in ("defined", "d1", "probe"):
        raise SystemExit(f"unknown arm {arm!r}")
    if arm == "probe":
        text = _replace(text, _MODULE_CONTAINS, _PROBE_DECL)
        return _replace(text, _KT_WRF, _KT_PROBE, expected=2)
    text = _replace(text, _DECL_PLM, _DECL_PLM + _DECL_ADD)
    text = _replace(text, _DECL_PLM6, _DECL_PLM6 + _DECL_ADD)
    text = _replace(text, _KT_WRF, _KT_DEFINED, expected=2)
    if arm == "defined":
        text = _replace(text, _WRAP_DECL, _WRAP_DECL_D2)
        text = _replace(text, _WRAP_FILL, _WRAP_FILL_D2)
        text = _replace(text, _WRAP_CALL, _WRAP_CALL_D2)
        text = _replace(text, _REFL_HEAD, _REFL_HEAD_D2)
        text = _replace(text, _REFL_ARGS, _REFL_ARGS_D2)
        text = _replace(text, _REFL_GRAUPEL, _REFL_GRAUPEL_D2)
        text = _replace(text, _REFL_SUM, _REFL_SUM_D2)
    return text


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    source = Path(argv[1]).read_text(encoding="ascii")
    Path(argv[2]).write_text(arm_source(source, argv[3]), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
