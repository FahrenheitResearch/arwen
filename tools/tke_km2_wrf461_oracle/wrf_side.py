"""Run the compiled WRF v4.6.1 km_opt=2 chain on every case and arm.

The chain is the one module_first_rk_step_part1/part2 run for diff_opt=2,
km_opt=2: phy_prep (th_phy, t_phy, p8w, t8w, rho, z), compute_diff_metrics
(z, rdz, rdzw, zx, zy) with WRF's physical boundary extensions,
cal_deform_and_div, phy_bc on the tensors, calculate_km_kh (calculate_N2 +
tke_km), phy_bc on the coefficients, then tke_rhs.  tke_shear, tke_buoyancy
and tke_dissip are also called one after another into a second array, so the
cumulative tendency after each term is available for attribution.

Usage: python wrf_side.py CASES_DIR LIBRARY OUT_DIR
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wrf_diffusion_oracle"))
from cases import core, pad2, pad3  # noqa: E402  (wrf_diffusion_oracle/cases.py)
sys.path.insert(1, str(HERE))
from arms import ARMS  # noqa: E402

F32 = np.float32


def _invoke(lib, name, args):
    getattr(lib, name)(*[ctypes.c_void_p(x.ctypes.data) if isinstance(x, np.ndarray)
                        else ctypes.c_int(x) if isinstance(x, int)
                        else ctypes.c_float(x) for x in args])


def wrf_chain(lib, arrays, meta, arm):
    _, iso, isfflx, ck, cd0, heat = arm
    nx, ny, nz, bx, by = [int(meta[k]) for k in ("nx", "ny", "nz", "bx", "by")]
    head = [nx, ny, nz, bx, by]
    shape = (nx + 6, nz + 1, ny + 6)
    a = dict(arrays)
    # WRF's t_2 is theta - t0.  The engine's theta is thb + thp in float32;
    # (theta - 300) + 300 is exact for 150 <= theta <= 600 (Sterbenz), so
    # phy_prep's th_phy = t_2 + t0 is the engine's theta word for word.
    thb = a["thb"] if a["thb"].ndim == 3 else a["thb"][:, None, None]
    theta = (thb + a["thp"]).astype(F32)
    assert theta.min() > 150 and theta.max() < 600
    a["t2"] = (theta - F32(300.0)).astype(F32)
    assert np.array_equal((a["t2"] + F32(300.0)).view(np.uint32),
                          theta.view(np.uint32))
    o = {k: pad3(a[k], nx, ny, nz, bx, by) for k in
         ("u", "v", "w", "php", "phb", "p", "alt", "t2")}
    o.update({k: pad2(a[k], nx, ny, bx, by)
              for k in ("msfu", "msfv", "msft", "mut", "ust", "hfx", "qfx")})
    for k in ("dn", "dnw", "fnm", "fnp", "znw", "c1h", "c2h", "c1f", "c2f"):
        f = np.asarray(a[k], dtype=F32)
        o[k] = np.asfortranarray(f[np.minimum(np.arange(nz + 1), f.size - 1)])
    # phy_prep's interior p8w/t8w interpolation reads fzm/fzp; diffusion only
    # consumes its surface and top extrapolations, which do not.
    o["fzm"], o["fzp"] = o["fnm"].copy(), o["fnp"].copy()
    moist = np.zeros((*shape, 4), dtype=F32, order="F")
    for slot, key in enumerate(("qv", "qc", "qi"), start=1):
        moist[:, :, :, slot] = pad3(a[key], nx, ny, nz, bx, by)
    o["moist"] = moist
    o["tke"] = pad3(a["tke"], nx, ny, nz, bx, by)
    for key in ("z", "rdz", "rdzw", "zx", "zy", "rho", "theta", "temp", "p8w",
                "t8w", "zw", "div", "d11", "d22", "d33", "d12", "d13", "d23",
                "kmh", "kmv", "khh", "khv", "bn2"):
        o[key] = np.zeros(shape, dtype=F32, order="F")
    P = lambda *keys: [o[k] for k in keys]
    rdx, rdy = [float(F32(1.0) / F32(meta[k])) for k in ("dx", "dy")]
    cf = [float(meta[k]) for k in ("cf1", "cf2", "cf3")]
    _invoke(lib, "oracle_metrics", head + P("php", "phb") + [rdx, rdy]
            + P("z", "rdz", "rdzw", "zx", "zy"))
    _invoke(lib, "oracle_phy", head + P("u", "v", "p", "alt", "php", "phb", "t2",
            "moist", "mut", "c1h", "c2h", "c1f", "c2f", "dnw", "fzm", "fzp", "znw")
            + [float(meta["p_top"])]
            + P("rho", "theta", "temp", "p8w", "t8w", "z", "zw"))
    _invoke(lib, "oracle_deform", head + P("u", "v", "w", "msfu", "msfv", "msft",
            "rdz", "rdzw", "zx", "zy", "dn", "dnw", "fnm", "fnp") + [rdx, rdy, *cf]
            + P("div", "d11", "d22", "d33", "d12", "d13", "d23"))
    for key, stag in (("div", 0), ("d11", 0), ("d22", 0), ("d33", 0),
                      ("d12", 4), ("d13", 5), ("d23", 6)):
        _invoke(lib, "oracle_bc", head + [stag, o[key]])
    _invoke(lib, "oracle_km2", head + [int(iso), int(isfflx)]
            + P("theta", "temp", "p", "p8w", "t8w", "moist", "tke", "msft", "rdz",
                "rdzw", "zx", "zy", "dn", "dnw", "div", "d11", "d22", "d33", "d12",
                "d13", "d23")
            + [float(meta["dx"]), float(meta["dy"]), float(meta["dt"]), *cf,
               float(ck), float(meta["mix_upper_bound"]), float(cd0), float(heat)]
            + P("kmh", "kmv", "khh", "khv", "bn2"))
    for key in ("kmh", "kmv", "khh", "khv"):
        _invoke(lib, "oracle_bc", head + [0, o[key]])
    # tke_rhs, called exactly as first_rk_step_part2 calls it.
    inputs = P("u", "v", "w", "d11", "d22", "d33", "d12", "d13", "d23", "div",
               "tke", "bn2", "theta", "p", "p8w", "t8w", "z", "rdz", "rdzw", "zx",
               "zy", "kmh", "kmv", "khv")
    inputs += [np.asfortranarray(o["moist"][:, :, :, 1])]
    inputs += P("rho", "msft", "mut", "ust", "hfx", "qfx", "dn", "dnw", "fnm",
                "fnp", "c1h", "c2h")
    outs = [np.zeros(shape, dtype=F32, order="F") for _ in range(4)]
    fn = lib.oracle_tke_rhs
    fn.argtypes = ([ctypes.c_int] * 7 + [ctypes.c_float] * 9
                   + [ctypes.c_void_p] * (len(inputs) + len(outs)))
    fn.restype = None
    fn(nx, ny, nz, bx, by, int(isfflx), int(iso), float(ck), float(meta["dx"]),
       float(meta["dy"]), float(meta["dt"]), float(cd0), float(heat), *cf,
       *[ctypes.c_void_p(x.ctypes.data) for x in inputs + outs])
    res = {k: core(o[k], nx, ny, nz) for k in
           ("kmh", "kmv", "khh", "khv", "bn2", "d11", "d22", "d33", "div",
            "theta", "temp", "rho", "rdzw")}
    res["d12"] = np.ascontiguousarray(o["d12"][3:3 + nx + 1, :nz, 3:3 + ny + 1].transpose(1, 2, 0))
    res["d13"] = np.ascontiguousarray(o["d13"][3:3 + nx + 1, :nz + 1, 3:3 + ny].transpose(1, 2, 0))
    res["d23"] = np.ascontiguousarray(o["d23"][3:3 + nx, :nz + 1, 3:3 + ny + 1].transpose(1, 2, 0))
    res["p8w_sfc"] = np.ascontiguousarray(o["p8w"][3:3 + nx, 0, 3:3 + ny].T)
    res["t8w_sfc"] = np.ascontiguousarray(o["t8w"][3:3 + nx, 0, 3:3 + ny].T)
    res["p8w_top"] = np.ascontiguousarray(o["p8w"][3:3 + nx, nz, 3:3 + ny].T)
    res["t8w_top"] = np.ascontiguousarray(o["t8w"][3:3 + nx, nz, 3:3 + ny].T)
    for name, arr in zip(("cum_shear", "cum_buoyancy", "cum_dissip", "rtke"), outs):
        res[name] = core(arr, nx, ny, nz)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", type=Path)
    ap.add_argument("library", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lib = ctypes.CDLL(str(args.library.resolve()))
    index = json.loads((args.cases / "cases.json").read_text())
    for case in index["cases"]:
        with np.load(args.cases / case["file"]) as data:
            meta = json.loads(str(data["meta_json"]))
            arrays = {k: data[k] for k in data.files if k != "meta_json"}
        for arm in ARMS:
            res = wrf_chain(lib, arrays, meta, arm)
            np.savez(args.out / f"wrf-{case['name']}-{arm[0]}.npz", **res)
        print(case["name"], flush=True)


if __name__ == "__main__":
    main()
