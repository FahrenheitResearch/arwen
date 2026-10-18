#!/usr/bin/env python3
"""Write a REPAIRED copy of WRF v4.6.1 module_mp_thompson.F: the control
oracle for the two defects classic Thompson (and mp=28) carry in 4.6.1.

The pinned source is never edited; this reads it, checks its SHA-256, and
writes the repaired copy elsewhere.  Each repair is an exact text
substitution with a required match count, so a different source refuses.

The two defects, both introduced when the hail-aware (mp=38) generation
gave the graupel tables an NRHG dimension (absent in v4.4.2, present
unchanged from v4.5.1 through v4.7.1 and the develop branch):

``rcg``  Rain collecting graupel (:2524-2546) reads tcg_racg, tmr_racg,
         tcr_gacr, tnr_racg and tnr_gacr at ``idx_bg(k)``, which is
         ``idx_bg1 = 5`` whenever the scheme is not hail-aware (:1944,
         :1950).  For mp=8 and mp=28 those tables are allocated with
         ``dimNRHG = NRHG1 = 1`` (:465, :607-615), and qr_acr_qg fills that
         one slot for rho_g(idx_bg1) (:4195-4223).  Index 5 on a dimension
         of extent 1 is out of bounds: column-major addressing lands four
         ntb_g1*ntb_g strides on, which is the entry for drop intercept bin
         ``idx_r1 + 4`` (a different rain distribution), and past the end
         of the allocation, which is undefined behaviour (heap contents),
         when ``idx_r = ntb_r`` and ``idx_r1 >= ntb_r1 - 3``.  v4.4.2 read
         the same table, without the NRHG dimension, at
         ``(idx_g1, idx_g, idx_r1, idx_r)``.  Repair: read slot 1 (the slot
         thompson_init filled) unless hail-aware.

``lqg``  The TAU+1 update (:3283-3304) sets ``L_qg(k) = .true.`` for
         graupel present after the sources only in its hail-aware branch;
         the classic branch can only clear it.  Graupel that the sources
         create at a level that had none at entry is therefore not flagged,
         and when no level of the column had graupel at entry the graupel
         sedimentation (:3744, :3902, both gated on ANY(L_qg)) is skipped
         for that step.  Every other species, the hail-aware branch, and
         v4.4.2 (:2980-2986) set the flag.  Repair: set it.

usage: repair_wrf461.py SRC_module_mp_thompson.F OUT.F [--only rcg|lqg]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

PINNED = "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802"

RCG_OLD = "idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)"
RCG_NEW = "idx_g1,idx_g,MERGE(idx_bg(k),1,is_hail_aware),idx_r1,idx_r)"
#: Seven live reads plus one commented-out read, all in :2524-2546.
RCG_COUNT = 8

LQG_OLD = ("         if ((qg1d(k) + qgten(k)*DT) .gt. R1) then\n"
           "            rg(k) = (qg1d(k) + qgten(k)*DT)*rho(k)\n"
           "               ygra1 = alog10(max(1.E-9, rg(k)))\n")
LQG_NEW = ("         if ((qg1d(k) + qgten(k)*DT) .gt. R1) then\n"
           "            rg(k) = (qg1d(k) + qgten(k)*DT)*rho(k)\n"
           "            L_qg(k) = .true.\n"
           "               ygra1 = alog10(max(1.E-9, rg(k)))\n")

REPAIRS = {"rcg": (RCG_OLD, RCG_NEW, RCG_COUNT),
           "lqg": (LQG_OLD, LQG_NEW, 1)}


def repair(text: str, only: str | None = None) -> str:
    for name, (old, new, count) in REPAIRS.items():
        if only and name != only:
            continue
        found = text.count(old)
        if found != count:
            raise SystemExit(f"repair {name}: {found} matches, want {count}")
        text = text.replace(old, new)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", choices=tuple(REPAIRS))
    a = ap.parse_args(argv)
    raw = a.src.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED:
        raise SystemExit(f"{a.src}: sha256 {digest}, want the pinned "
                         f"v4.6.1 {PINNED}")
    out = repair(raw.decode("utf-8"), a.only).encode("utf-8")
    a.out.write_bytes(out)
    print(f"{a.out}: repaired ({a.only or 'rcg+lqg'}) sha256 "
          f"{hashlib.sha256(out).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
