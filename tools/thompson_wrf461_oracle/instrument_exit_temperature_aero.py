#!/usr/bin/env python3
"""Write an ADDITIVELY instrumented copy of ``module_mp_thompson.F``.

WHY THIS EXISTS
---------------
``mp_gt_driver``'s working temperature is the LOCAL array ``t1d``
(module_mp_thompson.F:1117).  It is filled at :1222 with
``t1d(k) = th(i,k,j)*pii(i,k,j)``, handed to ``mp_thompson`` (:1290),
``calc_refl10cm`` (:1459) and ``calc_effectRad`` (:1472), and then
destroyed: :1358 writes ``th(i,k,j) = t1d(k)/pii(i,k,j)`` and the routine
returns.  A caller of PRISTINE WRF therefore has no way to read the
temperature the kernels actually operated on at exit, and recovering it by
multiplying ``th`` back by ``pii`` is a float32 round trip that is not the
identity: measured on the committed set, 37 of 456 after-rows come back one
ulp away, and for 158 of 912 rows TWO float32 values of ``t1d`` map to the
same ``th``, so the round trip is not even invertible in principle.

This module produces the ONLY thing that closes that gap: a copy of WRF with
two ``write`` statements added, and nothing else.  The fixtures themselves are
NOT generated from it -- ``build_aero.sh`` keeps compiling the pristine file --
and ``check_exit_temperature_aero.py`` proves the instrumentation is inert by
requiring the instrumented build to reproduce all 44 pristine outputs BYTE FOR
BYTE.

WHAT IS INSERTED
----------------
Two guarded blocks, both under ``i == i_start .and. j == j_start`` so they fire
once per scenario for exactly the column the fixtures dump:

* after :1222-1234's transfer loop -- the ENTRY ``t1d`` and ``pii``;
* immediately before :1349's write-back loop -- the EXIT ``t1d``, which is the
  value ``calc_refl10cm`` and ``calc_effectRad`` are handed at :1459/:1472.

Both write hexadecimal float32 bit patterns as well as ES24.16E3 decimals, so
the receipt is bit-exact by construction rather than by round-trip luck.

usage:  instrument_exit_temperature_aero.py SOURCE.F DESTINATION.F
"""
from __future__ import annotations

import sys
from pathlib import Path

#: module_mp_thompson.F:1221-1234.  Matched in full so a version bump that
#: reorders the transfer loop fails loudly instead of instrumenting the wrong
#: place.
ENTRY_ANCHOR = """         do k = kts, kte
            t1d(k) = th(i,k,j)*pii(i,k,j)
            p1d(k) = p(i,k,j)
            w1d(k) = w(i,k,j)
            dz1d(k) = dz(i,k,j)
            qv1d(k) = qv(i,k,j)
            qc1d(k) = qc(i,k,j)
            qi1d(k) = qi(i,k,j)
            qr1d(k) = qr(i,k,j)
            qs1d(k) = qs(i,k,j)
            qg1d(k) = qg(i,k,j)
            ni1d(k) = ni(i,k,j)
            nr1d(k) = nr(i,k,j)
            rho(k) = 0.622*p1d(k)/(R*t1d(k)*(qv1d(k)+0.622))
         enddo
"""

#: module_mp_thompson.F:1349-1358.
EXIT_ANCHOR = """         do k = kts, kte
            qv(i,k,j) = qv1d(k)
            qc(i,k,j) = qc1d(k)
            qi(i,k,j) = qi1d(k)
            qr(i,k,j) = qr1d(k)
            qs(i,k,j) = qs1d(k)
            qg(i,k,j) = qg1d(k)
            ni(i,k,j) = ni1d(k)
            nr(i,k,j) = nr1d(k)
            th(i,k,j) = t1d(k)/pii(i,k,j)
"""

#: Unit 91/92 are opened and closed inside the block, so neither is held while
#: ``table_ccnAct`` (:5123-5133) performs its "lowest free unit in 20..99"
#: search -- that search runs inside ``thompson_init``, long before
#: ``mp_gt_driver`` is entered.
_BLOCK = """
!ARWEN-INSTRUMENT-BEGIN {tag}
         if (i .eq. i_start .and. j .eq. j_start) then
            open(unit={unit}, file='{filename}', status='unknown', &
                 position='append', action='write')
            do k = kts, kte
               write({unit},'(I0,2(",",Z8.8),2(",",ES24.16E3))') k, &
                    transfer(t1d(k),1), transfer(pii(i,k,j),1), &
                    t1d(k), pii(i,k,j)
            enddo
            close({unit})
         endif
!ARWEN-INSTRUMENT-END {tag}
"""

ENTRY_FILE = "T1D_ENTRY_AERO.csv"
EXIT_FILE = "T1D_EXIT_AERO.csv"


def instrument(text: str) -> str:
    if text.count(ENTRY_ANCHOR) != 1:
        raise SystemExit(
            "entry anchor (module_mp_thompson.F:1221-1235) matched "
            f"{text.count(ENTRY_ANCHOR)} times, expected 1")
    if text.count(EXIT_ANCHOR) != 1:
        raise SystemExit(
            "exit anchor (module_mp_thompson.F:1349-1358) matched "
            f"{text.count(EXIT_ANCHOR)} times, expected 1")
    text = text.replace(
        ENTRY_ANCHOR,
        ENTRY_ANCHOR + _BLOCK.format(
            tag="T1D-ENTRY", unit=91, filename=ENTRY_FILE))
    text = text.replace(
        EXIT_ANCHOR,
        _BLOCK.format(tag="T1D-EXIT", unit=92, filename=EXIT_FILE)
        + EXIT_ANCHOR)
    return text


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    source, destination = Path(argv[1]), Path(argv[2])
    text = source.read_text()
    destination.write_text(instrument(text))
    added = len(instrument(text).splitlines()) - len(text.splitlines())
    print(f"instrumented {source} -> {destination} (+{added} lines, "
          "two write blocks, no arithmetic touched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
