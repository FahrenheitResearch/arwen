#!/usr/bin/env python3
"""Run the six radiation kernels on a CUDA device and dump their outputs.

Written so the GPU leg and the CPU-vs-GPU comparison can happen on different
machines: the rented box has the RTX 5090 but CPython 3.12 (no ``math.fma``,
which gpuwm/core/noahmp_libm.py needs), while the host has 3.13 but must
never touch its GPU.  This script runs on the box and writes
``noahmp-radiation-gpu-<leaf>.csv`` -- the same case column plus every output
as a binary32 bit pattern -- which is then compared on the host.

    python tools/noahmp_wrf461_oracle/dump_radiation_gpu.py <outdir>
"""
from __future__ import annotations

import csv
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

KDIR = REPO / "gpuwm" / "core" / "kernels"
ORACLE = REPO / "gpuwm" / "data" / "noahmp" / "oracle"


def _f(h):
    return np.float32(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])


def _hx(x):
    return "0x%08X" % struct.unpack("<I", struct.pack("<f", np.float32(x)))[0]


def _rows(leaf):
    with (ORACLE / f"noahmp-radiation-{leaf}.csv").open(newline="") as fh:
        return list(csv.DictReader(fh))


def _cols(rows, names):
    return np.array([[_f(r[c]) for c in names] for r in rows], dtype=np.float32)


PAIR = lambda *bs: [f"{b}{i}" for b in bs for i in (1, 2)]  # noqa: E731

SPEC = {
    "snow_age": dict(
        kernel="noahmp_rad_snow_age", nout=2, ints=None,
        fin=["tau0", "grain_growth", "extra_growth", "dirt_soot", "swemx",
             "dt", "tg", "sneqvo", "sneqv", "tauss_in"],
        out=["tauss_out", "fage"]),
    "snowalb_class": dict(
        kernel="noahmp_rad_snowalb_class", nout=5, ints=None,
        fin=["swemx", "qsnow", "dt", "albold"],
        out=["alb", "albsnd1", "albsnd2", "albsni1", "albsni2"]),
    "groundalb": dict(
        kernel="noahmp_rad_groundalb", nout=4, ints=["ist"],
        fin=["albsat1", "albsat2", "albdry1", "albdry2", "alblak1", "alblak2",
             "fsno", "smc1", "albsnd1", "albsnd2", "albsni1", "albsni2",
             "cosz", "tg"],
        out=["albgrd1", "albgrd2", "albgri1", "albgri2"]),
    "surrad": dict(
        kernel="noahmp_rad_surrad", nout=8, ints=None,
        fin=["mpe", "fsun", "fsha", "elai", "vai", "laisun", "laisha"]
            + PAIR("solad", "solai", "fabd", "fabi", "ftdd", "ftid", "ftii",
                   "albgrd", "albgri", "albd", "albi", "frevd", "frevi",
                   "fregd", "fregi"),
        out=["parsun", "parsha", "sav", "sag", "fsa", "fsr", "fsrv", "fsrg"]),
    "twostream": dict(
        kernel="noahmp_rad_twostream", nout=15, ints=["ib", "ic"],
        fin=["xl", "omegas1", "omegas2", "betads", "betais",
             "cosz", "vai", "fwet", "t", "fveg",
             "albgrd1", "albgrd2", "albgri1", "albgri2",
             "rho1", "rho2", "tau1", "tau2",
             "fab_in1", "fab_in2", "fre_in1", "fre_in2",
             "ftd_in1", "ftd_in2", "fti_in1", "fti_in2",
             "gdir_in", "frev_in1", "frev_in2", "freg_in1", "freg_in2",
             "bgap_in", "wgap_in"],
        out=["fab1", "fab2", "fre1", "fre2", "ftd1", "ftd2", "fti1", "fti2",
             "gdir", "frev1", "frev2", "freg1", "freg2", "bgap", "wgap"]),
    "albedo": dict(
        kernel="noahmp_rad_albedo", nout=36, ints=["ist"],
        fin=["tau0", "grain_growth", "extra_growth", "dirt_soot", "swemx",
             "albsat1", "albsat2", "albdry1", "albdry2", "alblak1", "alblak2",
             "rhol1", "rhol2", "rhos1", "rhos2", "taul1", "taul2",
             "taus1", "taus2", "xl", "omegas1", "omegas2", "betads", "betais",
             "dt", "cosz", "elai", "esai", "tg", "tv", "snowh", "fsno", "fwet",
             "sneqvo", "sneqv", "qsnow", "fveg",
             "smc1", "smc2", "smc3", "smc4", "albold_in", "tauss_in",
             "fage_in", "frevd_in1", "frevd_in2", "frevi_in1", "frevi_in2",
             "fregd_in1", "fregd_in2", "fregi_in1", "fregi_in2"],
        out=["fage", "albold", "tauss", "fsun", "bgap", "wgap"]
            + PAIR("albgrd", "albgri", "albd", "albi", "fabd", "fabi",
                   "ftdd", "ftid", "ftii", "frevd", "frevi", "fregd", "fregi",
                   "albsnd", "albsni")),
}


def main(argv):
    import cupy as cp

    outdir = Path(argv[1]) if len(argv) > 1 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)
    src = (KDIR / "noahmp_radiation.cu").read_text()
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false"))
    mod.compile()

    dev = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print(f"device : {dev['name'].decode()} sm_{dev['major']}{dev['minor']}")
    print(f"cupy   : {cp.__version__}  nvrtc {cp.cuda.nvrtc.getVersion()}")

    worst = 0
    for leaf, s in SPEC.items():
        rows = _rows(leaf)
        fin = _cols(rows, s["fin"])
        n = fin.shape[0]
        d_in = cp.asarray(fin.ravel(), dtype=cp.float32)
        d_out = cp.empty(n * s["nout"], dtype=cp.float32)
        args = [d_in]
        if s["ints"]:
            iv = np.array([[int(r[c]) for c in s["ints"]] for r in rows],
                          dtype=np.int32)
            args.append(cp.asarray(iv.ravel()))
        args += [d_out, np.int32(n)]
        threads = 128
        mod.get_function(s["kernel"])(((n + threads - 1) // threads,),
                                      (threads,), tuple(args))
        cp.cuda.runtime.deviceSynchronize()
        got = cp.asnumpy(d_out).reshape(n, s["nout"])
        want = _cols(rows, s["out"])

        gb = got.view(np.uint32)
        wb = want.view(np.uint32)
        nbad = int((gb != wb).sum())
        worst = max(worst, nbad)
        print(f"{leaf:>14}: {n} rows x {s['nout']} outputs, "
              f"{nbad} bit-differing lanes vs the WRF oracle")

        path = outdir / f"noahmp-radiation-gpu-{leaf}.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["case"] + s["out"])
            for r, g in zip(rows, got):
                w.writerow([r["case"]] + [_hx(v) for v in g])
        print(f"                wrote {path}")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
