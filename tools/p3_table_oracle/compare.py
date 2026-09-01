"""Compare gpuwm's P3 table machinery against the Fortran oracle dumps.

Usage: ``python compare.py <outdir>/O2``

Byte-identity is the claim for the PARSED tables, so they are compared
with ``tobytes()``.  The GENERATED rain tables are known not to be
byte-identical (host libm, see README.md), so for those the report gives
absolute, relative and ULP error rather than a pass/fail -- a tolerance
here would be a tolerance invented to obtain a pass.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np

import gpuwm
from gpuwm.core import p3_tables as T


def _report(name, ours, theirs, byte_identity_expected):
    ours = np.ascontiguousarray(ours, dtype=np.float32)
    theirs = np.ascontiguousarray(theirs, dtype=np.float32)
    same = ours.tobytes() == theirs.tobytes()
    ulp = np.abs(ours.view(np.int32).astype(np.int64)
                 - theirs.view(np.int32).astype(np.int64))
    err = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    rel = err / np.maximum(np.abs(theirs.astype(np.float64)), 1e-300)
    verdict = "OK" if same == byte_identity_expected else "UNEXPECTED"
    print(f"{name:14} byte-identical={same!s:5} differing="
          f"{int((ours != theirs).sum())}/{ours.size} "
          f"max_abs={err.max():.6g} max_rel={rel.max():.6g} "
          f"max_ulp={int(ulp.max())}  [{verdict}]")
    print(f"{'':14} sha256(ours)={hashlib.sha256(ours.tobytes()).hexdigest()}")
    return same


def main(outdir: str) -> int:
    print("gpuwm.__file__ =", gpuwm.__file__)
    print("p3_tables      =", T.__file__)
    print("table root     =", T.p3_table_root())

    itab, itabcoll = T.load_lookup_table_1(T.p3_table_root())
    f_itab = np.fromfile(f"{outdir}/itab.f32", dtype="<f4").reshape(
        (T.DENSIZE, T.RIMSIZE, T.ISIZE, T.TABSIZE), order="F")
    f_coll = np.fromfile(f"{outdir}/itabcoll.f32", dtype="<f4").reshape(
        (T.DENSIZE, T.RIMSIZE, T.ISIZE, T.RCOLLSIZE, T.COLLTABSIZE),
        order="F")

    print("\nPARSED (byte-identity is the claim):")
    ok = _report("itab", itab, f_itab, True)
    ok &= _report("itabcoll", itabcoll, f_coll, True)

    print("\nGENERATED (byte-identity is NOT claimed; see README):")
    vn, vm, revap = T.generate_rain_tables()
    for name, ours in (("vn_table", vn), ("vm_table", vm),
                       ("revap_table", revap)):
        theirs = np.fromfile(f"{outdir}/{name}.f32",
                             dtype="<f4").reshape((300, 10), order="F")
        _report(name, ours, theirs, False)

    colli = np.fromfile(f"{outdir}/itabcolli.f32", dtype="<f4")
    print(f"\nitabcolli (nCat>1 only): {colli.size} values, "
          f"{int((colli != 0).sum())} non-zero "
          "-- p3_init never reads table 2 at nCat=1")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
