"""Two measurements on the regime-column fixture that the GPU gate does not make.

1. ``--cpu-ref``: the float32 CPU authority (gpuwm/verify/gf_driver.py)
   against WRF's GFDRV (crgamma build), word for word.  The authority
   models powf/expf/logf as CORRECTLY ROUNDED by design
   (gpuwm/verify/gf_deep_ref.py), which is not what WRF's libm returns, so
   it is not expected to be bitwise on realistic columns.  ``--syslibm``
   swaps the system libm's float32 routines in, which attributes any
   residual to that one design choice.  MEASURED 2026-10-07 on W1
   (glibc 2.39): 1,783 words on 20 of 280 columns as designed; 0 with
   ``--syslibm``.

2. ``--seam-w``: the engine seam's w convention.  WRF's cumulus driver hands
   GFDRV ``W=grid%w_2`` (module_cumulus_driver.F), so ``omeg(k)`` uses the
   w face BELOW mass level k; the former gpuwm/core/gf.py convention handed it the average
   ``0.5*(w[k] + w[k+1])``.  This runs the authority (with ``--syslibm``,
   i.e. bitwise to WRF and to the kernel on this fixture) both ways on every
   column and reports how much the rain moves.  The top face is taken as
   w = 0.  MEASURED 2026-10-07: rain changes on 103 of 280 columns, 1
   rain/no-rain flip, 1 deep-top move, median 1.26 per cent and worst 32.1
   per cent where both rain, total rain +1.8 per cent.

usage:
    python gf_regime_seam_probe.py --cpu-ref [--syslibm] [--jobs N]
    python gf_regime_seam_probe.py --seam-w [--jobs N]
CPU only; the system-libm swap needs a Linux libm.so.6.
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gf_regime_columns as g                                    # noqa: E402

FIX = _ROOT / "tests" / "data" / "oracles" / "gf" / "regime"
_SYSLIBM = False


def _use_system_libm():
    import ctypes

    import gpuwm.verify.gf_deep_body as body
    import gpuwm.verify.gf_deep_ref as ref
    import gpuwm.verify.gf_shallow_ref as shallow

    lm = ctypes.CDLL("libm.so.6")
    lm.powf.restype = lm.expf.restype = lm.logf.restype = ctypes.c_float
    lm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
    lm.expf.argtypes = [ctypes.c_float]
    lm.logf.argtypes = [ctypes.c_float]
    ref._powf = lambda x, y: np.float32(lm.powf(float(np.float32(x)),
                                                float(np.float32(y))))
    ref._expf = lambda x: np.float32(lm.expf(float(np.float32(x))))
    ref._logf = lambda x: np.float32(lm.logf(float(np.float32(x))))
    for mod in (body, shallow):
        for name in ("_powf", "_expf", "_logf"):
            if hasattr(mod, name):
                setattr(mod, name, getattr(ref, name))


def _column(inp, i, w=None):
    from gpuwm.verify.gf_driver import gfdrv_column

    L, S = g.LEV_FIELDS, g.SCA_FIELDS
    lv = {n: inp["lev"][i, L.index(n)].copy() for n in L}
    if w is not None:
        lv["w"] = w
    sc = {n: np.float32(inp["sc"][i, S.index(n)]) for n in S}
    return gfdrv_column(
        **lv, p8w=inp["p8w"][i, :inp["nz"]].copy(), ht=sc["ht"],
        hfx=sc["hfx"], qfx=sc["qfx"], xland=sc["xland"],
        kpbl=int(inp["ii"][i, 0]), dt=float(sc["dt"]), dx=float(sc["dx"]),
        ishallow=int(inp["ii"][i, 1]), ichoice=int(inp["ii"][i, 2]))


def _init(syslibm):
    if syslibm:
        _use_system_libm()


def _cpu_ref_one(i):
    inp = g.read_inputs(FIX / "gf-regime-inputs.bin")
    out = g.read_outputs(FIX / "gf-regime-wrf-crgamma-k0.bin")
    a = _column(inp, i)
    bad = 0
    for j, n in enumerate(g.OUT_LEV):
        bad += int((np.asarray(a[n], np.float32).view(np.uint32)
                    != out["lev"][i, j].view(np.uint32)).sum())
    for j, n in enumerate(g.OUT_SCA):
        bad += int(np.float32(a[n]).view(np.uint32)
                   != out["sca"][i, j].view(np.uint32))
    for j, n in enumerate(g.OUT_INT):
        bad += int(int(a[n]) != int(out["isc"][i, j]))
    return bad


def _seam_one(i):
    inp = g.read_inputs(FIX / "gf-regime-inputs.bin")
    w = inp["lev"][i, g.LEV_FIELDS.index("w")].astype(np.float32)
    above = np.append(w[1:], np.float32(0.0)).astype(np.float32)
    w_avg = (np.float32(0.5) * (w + above)).astype(np.float32)
    a = _column(inp, i)
    b = _column(inp, i, w=w_avg)
    return (float(a["pratec"]), float(b["pratec"]), int(a["ktop_deep"]),
            int(b["ktop_deep"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cpu-ref", action="store_true")
    ap.add_argument("--seam-w", action="store_true")
    ap.add_argument("--syslibm", action="store_true")
    ap.add_argument("--jobs", type=int, default=32)
    args = ap.parse_args(argv)
    ncol = g.read_inputs(FIX / "gf-regime-inputs.bin")["ncol"]
    if args.cpu_ref:
        with Pool(args.jobs, initializer=_init,
                  initargs=(args.syslibm,)) as pool:
            r = np.array(pool.map(_cpu_ref_one, range(ncol)))
        print(json.dumps(dict(
            measurement="cpu authority vs WRF crgamma",
            libm="system" if args.syslibm else "correctly rounded (design)",
            columns=ncol, words_differ=int(r.sum()),
            columns_differ=int((r > 0).sum()))))
    if args.seam_w:
        with Pool(args.jobs, initializer=_init, initargs=(True,)) as pool:
            r = np.array(pool.map(_seam_one, range(ncol)))
        pa, pb = r[:, 0], r[:, 1]
        both = (pa > 0) & (pb > 0)
        rel = np.abs(pb[both] - pa[both]) / pa[both]
        print(json.dumps(dict(
            measurement="historical seam w: lower face vs former adjacent-face average",
            columns=ncol,
            columns_rain_changed=int((pa != pb).sum()),
            rain_flips=int(((pa > 0) != (pb > 0)).sum()),
            ktop_deep_moves=int((r[:, 2] != r[:, 3]).sum()),
            rel_change_where_both_rain_median=float(np.median(rel)),
            rel_change_where_both_rain_max=float(rel.max()),
            total_rain_ratio=float(pb.sum() / pa.sum()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
