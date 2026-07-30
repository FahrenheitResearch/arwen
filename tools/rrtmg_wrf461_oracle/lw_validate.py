"""Internal-consistency gates for the RRTMG LW oracle fixtures.

Two bitwise assertions per case, both against the UNMODIFIED module:

1. compose == direct: the decomposed chain (inatm -> cldprmc -> setcoef ->
   taumol -> taut -> rtrnmc) recorded stage by stage must reproduce the
   direct rrtmg_lw call exactly.  This proves the per-routine fixtures
   really are the composition the module executes.

2. transcription == wrapper: outputs derived from the transcribed prep
   (glw/olr/fluxes/heating tendencies) must equal the untouched
   RRTMG_LWRAD's own outputs exactly.  This proves the "in/" records are
   the exact arguments the WRF driver builds.

Usage: python lw_validate.py FIXTURE_DIR
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lw_fixtures import read_fixture  # noqa: E402


def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def assert_bitwise(a, b, what, cid, bad):
    a = np.atleast_1d(np.asarray(a, np.float32))
    b = np.atleast_1d(np.asarray(b, np.float32))
    if a.shape != b.shape:
        bad.append(f"case {cid}: {what}: shape {a.shape} vs {b.shape}")
        return
    neq = bits(a) != bits(b)
    if neq.any():
        i = np.argwhere(neq)[0]
        bad.append(f"case {cid}: {what}: {int(neq.sum())}/{a.size} mismatch, "
                   f"first at {tuple(i)}: {a[tuple(i)]!r} vs {b[tuple(i)]!r}")


def main():
    fixdir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(fixdir, "lw_case_*.bin")))
    if not files:
        raise SystemExit("no fixtures found")
    bad = []
    for path in files:
        fx = read_fixture(path)
        cid = int(fx["meta/caseid"])
        nl = int(fx["meta/nlayers"])
        nz = int(fx["meta/nz"])

        # -- compose == direct ----------------------------------------
        assert_bitwise(fx["out/uflx"][0], fx["rtrnmc/totuflux"],
                       "uflx vs totuflux", cid, bad)
        assert_bitwise(fx["out/dflx"][0], fx["rtrnmc/totdflux"],
                       "dflx vs totdflux", cid, bad)
        assert_bitwise(fx["out/uflxc"][0], fx["rtrnmc/totuclfl"],
                       "uflxc vs totuclfl", cid, bad)
        assert_bitwise(fx["out/dflxc"][0], fx["rtrnmc/totdclfl"],
                       "dflxc vs totdclfl", cid, bad)
        assert_bitwise(fx["out/hr"][0], fx["rtrnmc/htr"][:nl],
                       "hr vs htr", cid, bad)
        assert_bitwise(fx["out/hrc"][0], fx["rtrnmc/htrc"][:nl],
                       "hrc vs htrc", cid, bad)

        # -- transcription == wrapper ----------------------------------
        uflx = fx["out/uflx"][0]
        dflx = fx["out/dflx"][0]
        uflxc = fx["out/uflxc"][0]
        dflxc = fx["out/dflxc"][0]
        assert_bitwise(fx["wrap/glw"], dflx[0], "glw", cid, bad)
        assert_bitwise(fx["wrap/olr"], uflx[nl], "olr", cid, bad)
        assert_bitwise(fx["wrap/lwcf"],
                       np.float32(uflxc[nl]) - np.float32(uflx[nl]),
                       "lwcf", cid, bad)
        assert_bitwise(fx["wrap/lwupt"], uflx[nl], "lwupt", cid, bad)
        assert_bitwise(fx["wrap/lwuptc"], uflxc[nl], "lwuptc", cid, bad)
        assert_bitwise(fx["wrap/lwdnt"], dflx[nl], "lwdnt", cid, bad)
        assert_bitwise(fx["wrap/lwdntc"], dflxc[nl], "lwdntc", cid, bad)
        assert_bitwise(fx["wrap/lwupb"], uflx[0], "lwupb", cid, bad)
        assert_bitwise(fx["wrap/lwupbc"], uflxc[0], "lwupbc", cid, bad)
        assert_bitwise(fx["wrap/lwdnb"], dflx[0], "lwdnb", cid, bad)
        assert_bitwise(fx["wrap/lwdnbc"], dflxc[0], "lwdnbc", cid, bad)

        hr = fx["out/hr"][0]
        hrc = fx["out/hrc"][0]
        pi3d = fx["wrap/pi3d"]
        f = np.float32
        rth = (hr[:nz].astype(np.float32) / f(86400.0)).astype(np.float32)
        rth = (rth / pi3d.astype(np.float32)).astype(np.float32)
        rthc = (hrc[:nz].astype(np.float32) / f(86400.0)).astype(np.float32)
        rthc = (rthc / pi3d.astype(np.float32)).astype(np.float32)
        assert_bitwise(fx["wrap/rthratenlw"], rth, "rthratenlw", cid, bad)
        assert_bitwise(fx["wrap/rthratenlwc"], rthc, "rthratenlwc", cid, bad)

    if bad:
        print(f"FAIL: {len(bad)} mismatches across {len(files)} cases")
        for line in bad[:40]:
            print(" ", line)
        raise SystemExit(1)
    print(f"OK: compose==direct and transcription==wrapper, bitwise, "
          f"{len(files)} cases")


if __name__ == "__main__":
    main()
