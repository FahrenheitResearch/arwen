"""Write WRF 4.7.1's module_ieva_em with A179's two corrections applied.

    python wrf_a179.py WRF/dyn_em/module_ieva_em.f90 OUT/module_ieva_em.f90

gpuwm's ``ieva_solve_w`` departs from WRF 4.7.1's ``advect_w_implicit``
in its two boundary terms (A179, gpuwm/core/ieva.py declared difference
(4)), so the word-for-word anchor is WRF's own routine with the same two
corrections, compiled with WRF's own flags (build.sh).  This rewrites
only ``advect_w_implicit``; every other routine of the module is WRF's
bytes.  Each edit names its WRF line and must match exactly once, so a
WRF source that is not 4.7.1's is refused rather than half-patched.  The
input may be the module's ``.F`` or the ``.f90`` WRF's build preprocessed
from it: every anchor is a code line, which the preprocessing keeps.

1. Lower boundary (module_ieva_em.F:1231-1244).  WRF's ``dw`` sums the
   COUPLED ``utend``/``vtend`` (Pa m s-2) where the ``w_old`` terms beside
   it are uncoupled.  Each level becomes ``utend*msfuy/(c1h*muu+c2h)`` and
   ``vtend*msfvx/(c1h*muv+c2h)``, WRF's own uncoupling of u_2 and v_2
   (module_small_step_em.F:392 and :383), so the routine gains ``c1h, c2h,
   muu, muv``: the stage face masses rk_tendency couples the winds with.
2. Upper boundary (:1248-1253).  ``(ph_new - ph_old)/dt_rk`` (m2 s-3) is
   divided by g, as the ``ph_tend`` term beside it already is.
"""

from __future__ import annotations

import sys

BEGIN = "SUBROUTINE advect_w_implicit("
END = "END SUBROUTINE advect_w_implicit"

_LOWER = [
    ("          *(cf1*vtend(i,1,j+1)+cf2*vtend(i,2,j+1)+cf3*vtend(i,3,j+1))    &",
     "          *(cf1*(vtend(i,1,j+1)*msfvx(i,j+1)/(c1h(1)*muv(i,j+1)+c2h(1)))"
     "+cf2*(vtend(i,2,j+1)*msfvx(i,j+1)/(c1h(2)*muv(i,j+1)+c2h(2)))"
     "+cf3*(vtend(i,3,j+1)*msfvx(i,j+1)/(c1h(3)*muv(i,j+1)+c2h(3))))    &"),
    ("          *(cf1*vtend(i,1,j  )+cf2*vtend(i,2,j  )+cf3*vtend(i,3,j  ))  ) &",
     "          *(cf1*(vtend(i,1,j  )*msfvx(i,j  )/(c1h(1)*muv(i,j  )+c2h(1)))"
     "+cf2*(vtend(i,2,j  )*msfvx(i,j  )/(c1h(2)*muv(i,j  )+c2h(2)))"
     "+cf3*(vtend(i,3,j  )*msfvx(i,j  )/(c1h(3)*muv(i,j  )+c2h(3))))  ) &"),
    ("          *(cf1*utend(i+1,1,j)+cf2*utend(i+1,2,j)+cf3*utend(i+1,3,j))    &",
     "          *(cf1*(utend(i+1,1,j)*msfuy(i+1,j)/(c1h(1)*muu(i+1,j)+c2h(1)))"
     "+cf2*(utend(i+1,2,j)*msfuy(i+1,j)/(c1h(2)*muu(i+1,j)+c2h(2)))"
     "+cf3*(utend(i+1,3,j)*msfuy(i+1,j)/(c1h(3)*muu(i+1,j)+c2h(3))))    &"),
    ("          *(cf1*utend(i  ,1,j)+cf2*utend(i  ,2,j)+cf3*utend(i  ,3,j))  )",
     "          *(cf1*(utend(i  ,1,j)*msfuy(i  ,j)/(c1h(1)*muu(i  ,j)+c2h(1)))"
     "+cf2*(utend(i  ,2,j)*msfuy(i  ,j)/(c1h(2)*muu(i  ,j)+c2h(2)))"
     "+cf3*(utend(i  ,3,j)*msfuy(i  ,j)/(c1h(3)*muu(i  ,j)+c2h(3))))  )"),
]

_UPPER = [
    ("          dw = msfty(i,j)*(  (ph_new(i,k+1,j)-ph_old(i,k+1,j))/dt_rk     &",
     "          dw = msfty(i,j)*(  (ph_new(i,k+1,j)-ph_old(i,k+1,j))/dt_rk/g   &"),
]

_SIGNATURE = [
    ("                              cf1, cf2, cf3,                 &\n",
     "                              cf1, cf2, cf3,                 &\n"
     "                              c1h, c2h, muu, muv,            &\n"),
    ("   REAL , DIMENSION( ims:ime , jms:jme ) , INTENT(IN) :: mut, mut_old, mut_new\n",
     "   REAL , DIMENSION( ims:ime , jms:jme ) , INTENT(IN) :: mut, mut_old, mut_new\n"
     "   REAL , DIMENSION( ims:ime , jms:jme ) , INTENT(IN) :: muu, muv\n"
     "   REAL , DIMENSION( kms:kme ) , INTENT(IN) :: c1h, c2h\n"),
]


def correct(text: str) -> str:
    start = text.find(BEGIN)
    stop = text.find(END, start)
    if start < 0 or stop < 0 or text.find(BEGIN, start + 1) >= 0:
        raise SystemExit("advect_w_implicit not found exactly once")
    body = text[start:stop]
    for old, new in _SIGNATURE + _LOWER + _UPPER:
        if body.count(old) != 1:
            raise SystemExit(
                f"anchor found {body.count(old)} times, not once: {old!r}")
        body = body.replace(old, new)
    return text[:start] + body + text[stop:]


def main() -> int:
    source, target = sys.argv[1], sys.argv[2]
    with open(source, encoding="ascii", newline="") as fh:
        text = fh.read()
    with open(target, "w", encoding="ascii", newline="") as fh:
        fh.write(correct(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
