#!/usr/bin/env python3
"""Adversarial columns for the Shin-Hong wide column oracle (refute pass).

Writes records in the SAME binary layout make_columns.py writes, so the
lane's own Fortran driver (run_shinhong_columns.F90) and the lane's own
comparison module (gpuwm.verify.shinhong_column_oracle) read them unchanged.

    python refute_columns.py fuzz  OUT.bin OUT.json [N_BASE]
    python refute_columns.py tanh  IN.bin WRF_OUT.bin OUT.bin OUT.json [--host-libm]

fuzz : columns the lane did not use.  The lane's 17 family builders with a
       new seed, on grids the lane never ran (4, 10, 20, 80, 100, 128 levels,
       first layers 2 to 100 m), with per-level theta/wind noise, random
       cloud layers, surface scalars drawn independently of each other
       (br sign against the heat flux, ust scaled 0.2x to 4x, psim/psih
       scaled), latitudes over the whole sphere, TKE well above the floor,
       and ten grid spacings from 25 m to 100 km with dt from 0.15 s to
       600 s, some with dy != dx.

tanh : csfac = 0.5*(tanh(uwstx)+3) (module_bl_shinhong.F:847).  The WRF
       reference calls the host C library's tanhf; the WOOF kernel spells it
       (float)tanh((double)x).  For every column of IN.bin whose WRF wstar is
       non-zero, ust is re-chosen (wstar does not depend on ust: wstar3 =
       govrth*bfx0*hpbl from the first-guess hpbl, :768) so that
       uwstx = -80*|ust/wstar - 0.5| + 14 lands on a float32 where
       0.5*(tanhf(x)+3) and 0.5*(rn(tanh(x))+3) are different words.  Every
       other input word is the original record's.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

F32 = np.float32
HERE = Path(__file__).resolve().parent


def _load_make_columns():
    for cand in (HERE / "make_columns.py",
                 HERE.parent / "shinhong_column_oracle" / "make_columns.py"):
        if cand.exists():
            spec = importlib.util.spec_from_file_location("mc", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("make_columns.py not found")


SCALARS = ("dt", "dx", "dy", "psfc", "znt", "ust", "hfx", "qfx", "wspd",
           "br", "psim", "psih", "xland", "u10", "v10", "corf")
LEVELS = ("u", "v", "theta", "pi", "t", "qv", "qc", "qi", "p", "dz", "tke")


# --------------------------------------------------------------------------
# raw record reader (input layout of make_columns.py)
# --------------------------------------------------------------------------

def read_inputs(raw: bytes):
    pos = 0
    out = []
    while pos < len(raw):
        start = pos
        nz, tke_diag, fam, base = struct.unpack_from("<4i", raw, pos)
        pos += 16
        sc = np.frombuffer(raw, "<f4", 16, pos).copy()
        pos += 64 + 4 * nz * len(LEVELS) + 4 * (nz + 1)
        out.append(dict(start=start, end=pos, nz=nz, tke_diag=tke_diag,
                        family=fam, base=base, scalars=sc))
    return out


def read_wrf_surface(raw: bytes):
    pos = 0
    out = []
    while pos < len(raw):
        nz, kpbl, fam, base = struct.unpack_from("<4i", raw, pos)
        pos += 16 + 4 * nz * 9
        s = np.frombuffer(raw, "<f4", 9, pos).copy()
        pos += 36
        out.append(dict(nz=nz, kpbl=kpbl, hpbl=s[0], wstar=s[1]))
    return out


# --------------------------------------------------------------------------
# fuzz
# --------------------------------------------------------------------------

FUZZ_SEED = 7_10_2026
#: grids the lane never ran, plus two of the lane's for contrast
FUZZ_GRIDS = (
    (4, 60.0, 1.5, 3000.0),
    (10, 40.0, 1.3, 1500.0),
    (20, 25.0, 1.15, 900.0),
    (80, 8.0, 1.05, 400.0),
    (100, 5.0, 1.045, 350.0),
    (128, 2.5, 1.045, 300.0),
    (50, 20.0, 1.08, 600.0),
    (35, 100.0, 1.05, 1000.0),
)
#: (dx, dt): spacings the lane never ran, dt as a user might set it
FUZZ_DX = ((25.0, 0.15), (50.0, 0.3), (150.0, 1.0), (500.0, 3.0),
           (750.0, 4.0), (2000.0, 12.0), (4000.0, 24.0), (12000.0, 72.0),
           (50000.0, 240.0), (100000.0, 600.0))


def fuzz_records(n_base: int):
    mc = _load_make_columns()
    mc.GRIDS = FUZZ_GRIDS            # the builders draw their grid from here
    rng = np.random.default_rng(FUZZ_SEED)
    builders = [b for b, _ in mc.FAMILY_BUILDERS]
    recs, labels = [], []
    bid = 0
    while bid < n_base:
        builder = builders[rng.integers(len(builders))]
        try:
            c = builder(rng)
            nz = c.nz
            # per-level noise: breaks the smooth profiles so Ri < 0 above
            # the PBL, the dth sign test in mixlen and entfac < 4.6 fire
            if rng.random() < 0.7:
                c.theta = c.theta + rng.normal(0.0, rng.uniform(0.05, 0.8), nz)
            extra = rng.random()
            c.winds(rng, rng.uniform(0.2, 30.0), rng.uniform(-3, 10),
                    rng.uniform(-60, 60),
                    jet=(rng.uniform(100, 1500), rng.uniform(0, 15))
                    if extra < 0.4 else None)
            if rng.random() < 0.6:
                c.u = c.u + rng.normal(0.0, rng.uniform(0.1, 3.0), nz)
                c.v = c.v + rng.normal(0.0, rng.uniform(0.1, 3.0), nz)
            if rng.random() < 0.4:
                zb = rng.uniform(0, 3000)
                c.cloud("qc", zb, zb + rng.uniform(50, 2500),
                        10 ** rng.uniform(-6, -2.7))
            if rng.random() < 0.4:
                zb = rng.uniform(0, 8000)
                c.cloud("qi", zb, zb + rng.uniform(50, 4000),
                        10 ** rng.uniform(-6.5, -3))
            c.lat = rng.uniform(-90, 90) if rng.random() < 0.7 else \
                rng.choice([0.0, 90.0, -90.0, 1e-3])
            if rng.random() < 0.3:
                c.xland = 3.0 - c.xland
            if rng.random() < 0.5:
                c.tke_profile = (rng.choice(["conv", "stable"]),
                                 rng.uniform(50, 4000))
            mc.finished(c, rng)
            if rng.random() < 0.3:
                c.tke = c.tke * rng.uniform(1.0, 20.0)
        except (FloatingPointError, ValueError, ZeroDivisionError):
            continue
        rec = c.finish()
        sc = rec["scalars"]
        # surface scalars drawn apart from one another (WRF's driver hands
        # the scheme whatever the surface layer and land model left)
        if rng.random() < 0.5:
            sc["ust"] = F32(max(float(sc["ust"]) * math.exp(rng.uniform(-1.6, 1.4)), 0.003))
        if rng.random() < 0.35:
            sc["hfx"] = F32(float(sc["hfx"]) * rng.uniform(-1.5, 2.5)
                            + rng.uniform(-30, 30))
        if rng.random() < 0.35:
            sc["qfx"] = F32(float(sc["qfx"]) * rng.uniform(-1.0, 3.0))
        if rng.random() < 0.35:
            sc["br"] = F32(rng.uniform(-6.0, 2.5))
        if rng.random() < 0.3:
            sc["psim"] = F32(float(sc["psim"]) * rng.uniform(0.4, 2.5))
            sc["psih"] = F32(float(sc["psih"]) * rng.uniform(0.4, 2.5))
        if rng.random() < 0.2:
            sc["wspd"] = F32(float(sc["wspd"]) * rng.uniform(0.5, 3.0))
        if rng.random() < 0.2:
            sc["znt"] = F32(10 ** rng.uniform(-5, 0.5))
        aniso = rng.random() < 0.15
        picks = rng.choice(len(FUZZ_DX), size=3, replace=False)
        for j, idx in enumerate(picks):
            dx, dt = FUZZ_DX[idx]
            r = dict(rec)
            r["scalars"] = dict(sc)
            r["scalars"]["dt"] = F32(dt * (rng.choice([1.0, 2.0, 5.0])
                                           if j == 2 else 1.0))
            r["scalars"]["dx"] = F32(dx)
            r["scalars"]["dy"] = F32(dx * (rng.choice([0.25, 3.0])
                                           if aniso else 1.0))
            r["tke_diag"] = 0 if (j == 1 and rng.random() < 0.5) else 1
            r["base"] = bid
            recs.append(r)
            labels.append({"column": len(labels), "base": bid,
                           "family": mc.FAMILIES[c.family], "nz": c.nz,
                           "dx": float(r["scalars"]["dx"]),
                           "dy": float(r["scalars"]["dy"]),
                           "dt": float(r["scalars"]["dt"]),
                           "tke_diag": r["tke_diag"]})
        bid += 1
    return mc, recs, labels


# --------------------------------------------------------------------------
# tanh probes
# --------------------------------------------------------------------------

def _libm_tanhf(host: bool = False):
    """The reference host's float32 tanhf.

    By default WOOF's own float32 tanhf (gpuwm.core.mynn_pbl._tanhf, held to
    the reference host's words by tests/test_mynn_pbl.py), so the generator
    is a pure function on every platform.  host=True calls the running C
    library's tanhf through ctypes instead (Linux), the cross-check the
    fixture was recorded with: both choose the same columns on W1.
    """
    if not host:
        from gpuwm.core.mynn_pbl import _tanhf

        return lambda x: float(_tanhf(F32(x)))
    libm = ctypes.CDLL("libm.so.6")
    libm.tanhf.restype = ctypes.c_float
    libm.tanhf.argtypes = [ctypes.c_float]
    return libm.tanhf


def csfac_pair(x: np.float32, tanhf):
    a = F32(tanhf(float(x)))
    b = F32(math.tanh(float(x)))
    ca = F32(F32(0.5) * F32(a + F32(3.0)))
    cb = F32(F32(0.5) * F32(b + F32(3.0)))
    return ca, cb


def uwstx_of(ust: np.float32, wstar: np.float32) -> np.float32:
    uwst = F32(abs(F32(ust / wstar) - F32(0.5)))
    return F32(F32(F32(-80.0) * uwst) + F32(14.0))


def tanh_records(in_raw: bytes, wrf_raw: bytes, per_col: int = 2,
                 host: bool = False):
    tanhf = _libm_tanhf(host)
    ins = read_inputs(in_raw)
    outs = read_wrf_surface(wrf_raw)
    assert len(ins) == len(outs)
    chunks, labels = [], []
    rng = np.random.default_rng(847)
    for i, (r, o) in enumerate(zip(ins, outs)):
        w = F32(o["wstar"])
        if not (np.isfinite(w) and w > 0):
            continue
        found = []
        for frac in rng.permutation(np.linspace(0.215, 0.785, 6000)):
            ust = F32(w * F32(frac))
            x = uwstx_of(ust, w)
            ca, cb = csfac_pair(x, tanhf)
            if ca != cb:
                found.append((ust, x, ca, cb))
                if len(found) >= per_col:
                    break
        for ust, x, ca, cb in found:
            rec = bytearray(in_raw[r["start"]:r["end"]])
            struct.pack_into("<f", rec, 16 + 4 * SCALARS.index("ust"),
                             float(ust))
            chunks.append(bytes(rec))
            labels.append({"column": len(labels), "source_column": i,
                           "nz": r["nz"], "dx": float(r["scalars"][1]),
                           "tke_diag": r["tke_diag"], "wstar": float(w),
                           "ust": float(ust), "uwstx": float(x),
                           "csfac_libm": float(ca),
                           "csfac_rn_tanh": float(cb)})
    return b"".join(chunks), labels


def main(argv):
    if len(argv) >= 3 and argv[1] == "fuzz":
        n = int(argv[4]) if len(argv) > 4 else 4000
        mc, recs, labels = fuzz_records(n)
        mc.write_records(recs, Path(argv[2]))
        Path(argv[3]).write_text(json.dumps(
            {"seed": FUZZ_SEED, "columns": labels}, indent=0) + "\n")
        print(f"fuzz: wrote {len(recs)} columns from {n} bases")
        return 0
    if len(argv) in (6, 7) and argv[1] == "tanh":
        data, labels = tanh_records(Path(argv[2]).read_bytes(),
                                    Path(argv[3]).read_bytes(),
                                    host=argv[6:] == ["--host-libm"])
        Path(argv[4]).write_bytes(data)
        Path(argv[5]).write_text(json.dumps({"columns": labels}, indent=0)
                                 + "\n")
        print(f"tanh: wrote {len(labels)} probe columns")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main(sys.argv))
