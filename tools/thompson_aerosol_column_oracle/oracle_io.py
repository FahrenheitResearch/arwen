"""Shared pieces of the mp=28 GPU column oracle: the WRF batch driver's
stream format, the output field list, and the bitwise metrics.

Nothing here imports gpuwm, CuPy or the host backend, so the WRF side and
the comparison run in a plain NumPy process.

The WRF side is ``tools/thompson_real_column_parity/run_columns_aero.F90``,
built by that folder's ``build_wrf.sh`` from unmodified WRF v4.6.1
``phys/module_mp_thompson.F`` (gfortran -O2 -fno-tree-vectorize, scalar
libm, no FMA).  Its stream layout is documented in the driver's header.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

f32 = np.float32

#: The eleven mp=28 moments, in the driver's order.
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
           "nwfa", "nifa")
#: Every 3-D output the driver writes (16) and every 2-D one (7): the 23
#: compared fields.
OUT3 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
        "nifa", "th", "refl", "re_cloud", "re_ice", "re_snow")
OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
        "graupelncv", "sr")
FIELDS = OUT3 + OUT2
WRF_IN = ("th", "pii", "p", "w_lower", "dz", "hgt", *SPECIES)


def write_wrf_input(path, inp, dt):
    """``inp`` holds (ncol, nz) float32 arrays th pii p dz hgt and the
    species, ``w`` (ncol, nz+1), and (ncol,) nwfa2d nifa2d."""
    ncol, nz = inp["p"].shape
    with open(path, "wb") as fh:
        np.array([ncol, nz], dtype="<i4").tofile(fh)
        np.array([dt], dtype="<f4").tofile(fh)
        for name in WRF_IN:
            a = inp["w"][:, :nz] if name == "w_lower" else inp[name]
            np.ascontiguousarray(np.asarray(a, f32).T, dtype="<f4").tofile(fh)
        for name in ("nwfa2d", "nifa2d"):
            np.ascontiguousarray(inp[name], dtype="<f4").tofile(fh)


def read_wrf_output(path, ncol, nz):
    raw = np.fromfile(path, dtype="<f4")
    want = len(OUT3) * ncol * nz + len(OUT2) * ncol
    if raw.size != want:
        raise RuntimeError(f"{path}: {raw.size} words, expected {want}")
    out, off = {}, 0
    for name in OUT3:
        out[name] = raw[off:off + ncol * nz].reshape(nz, ncol).T.copy()
        off += ncol * nz
    for name in OUT2:
        out[name] = raw[off:off + ncol].copy()
        off += ncol
    return out


def run_wrf(binary, run_dir, in_path, out_path):
    """CCN_ACTIVATE.BIN is big-endian and is opened on unit 20."""
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    done = subprocess.run([str(binary), str(in_path), str(out_path)],
                          cwd=str(run_dir), env=env, capture_output=True,
                          text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{binary} failed:\n{done.stdout}\n{done.stderr}")
    return done.stdout


#: WOOF stores the three effective radii in microns, formed as WRF's metre
#: value times 1.0e6 in float32 (thompson_aerosol_state.cu, the
#: mp_gt_driver clamp then ``* 1.0e6f``), which is the conversion WRF's own
#: radiation drivers apply on the way in (``re_cloud*1.E6``).  The oracle
#: compares WOOF's word with WRF's metre value converted the same way.
MICRON_FIELDS = ("re_cloud", "re_ice", "re_snow")


def wrf_view(name, a):
    a = np.asarray(a, f32)
    if name in MICRON_FIELDS:
        return (a * f32(1.0e6)).astype(f32)
    return a


def ordered(a):
    """float32 bit patterns mapped to a monotonic integer line, so the ULP
    distance of two floats is the difference of their images (+0 and -0
    both map to 0)."""
    b = np.asarray(a, f32).view(np.int32).astype(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFF), b)


def ulp_distance(a, b):
    return np.abs(ordered(a) - ordered(b))


def bitwise(port, wrf):
    """Per-field bitwise agreement of two float32 arrays of one shape."""
    a = np.asarray(port, f32)
    b = np.asarray(wrf, f32)
    same_bits = a.view(np.int32) == b.view(np.int32)
    nan_both = np.isnan(a) & np.isnan(b)
    exact = same_bits | nan_both
    d_ulp = ulp_distance(a, b)
    d_ulp = np.where(exact, 0, d_ulp)
    absd = np.where(exact, 0.0, np.abs(a.astype(np.float64)
                                        - b.astype(np.float64)))
    res = {"cells": int(a.size), "bit_identical": int(exact.sum()),
           "differ": int((~exact).sum()),
           "max_ulp": int(d_ulp.max()) if a.size else 0,
           "max_abs_diff": float(np.nanmax(absd)) if a.size else 0.0}
    if res["differ"]:
        idx = np.unravel_index(int(np.argmax(d_ulp)), a.shape)
        res["worst"] = {"index": [int(i) for i in idx],
                        "port": float(a[idx]), "wrf": float(b[idx]),
                        "ulp": int(d_ulp[idx])}
    return res, ~exact


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


HERE = Path(__file__).resolve().parent
