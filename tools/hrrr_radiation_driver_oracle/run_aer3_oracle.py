"""Cut the aer_opt = 3 optics fixture with the compiled fork routines.

Usage: python run_aer3_oracle.py BUILD_DIRECTORY OUTPUT.npz

Columns: physical profiles (surface pressure 600-1040 hPa, surface
temperature 235-320 K with a tropospheric lapse rate and a stratospheric
floor, 50 layers thickening upward) carrying the aerosol a Thompson
aerosol-aware forecast carries, plus edge columns that reach every branch
the routines take: RH below 10.1 and above 98 (gt_aod's clamps), each of
its three RH index branches, temperatures for all four t_idx and the clamp
beyond, QNWFA/QNIFA at zero, negative and above the MIN caps, and QV zero,
negative and supersaturated (the Bolton RH clamps at 0 and 99).  The fixture records inputs, the fork's
outputs and the build receipt; tests/test_rrtmg_aerosol_optics.py replays
the inputs through the NumPy twin and the CUDA kernel.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

F = np.float32


def _qsat(p, t):
    """Saturation mixing ratio for building inputs only (not graded)."""
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    return 0.622 * es / np.maximum(p - es, 1.0)


def inputs(seed=20261003, ncol=320, nz=50):
    rng = np.random.default_rng(seed)
    psfc = rng.uniform(60000.0, 104000.0, ncol)
    tsfc = rng.uniform(235.0, 320.0, ncol)
    eta = np.linspace(0.0, 1.0, nz + 1)
    ptop = 2000.0
    pw = ptop + (psfc[:, None] - ptop) * (1.0 - eta[None, :]) ** 1.3
    p = 0.5 * (pw[:, :-1] + pw[:, 1:])
    z = 7000.0 * np.log(psfc[:, None] / p)
    t = np.maximum(tsfc[:, None] - 0.0065 * z, 205.0 + rng.uniform(-25, 10, (ncol, 1)))
    dz = 7000.0 * np.log(pw[:, :-1] / pw[:, 1:])
    rh_target = rng.uniform(0.0, 1.15, (ncol, nz))
    qv = rh_target * _qsat(p, t)
    nwfa = 10.0 ** rng.uniform(5.0, 11.3, (ncol, nz))
    nifa = 10.0 ** rng.uniform(1.0, 10.3, (ncol, nz))
    # edge columns
    qv[0] = 0.0
    qv[1] = -1.0e-5
    qv[2] = 3.0 * _qsat(p[2], t[2])
    nwfa[3] = 0.0
    nifa[3] = 0.0
    nwfa[4] = -5.0
    nifa[4] = -5.0
    nwfa[5] = 5.0e11
    nifa[5] = 5.0e10
    t[6] = np.linspace(330.0, 180.0, nz)
    for col, frac in ((7, 0.55), (8, 0.62), (9, 0.70), (10, 0.79),
                      (11, 0.85), (12, 0.97), (13, 0.985)):
        qv[col] = frac * _qsat(p[col], t[col])
    out = dict(p=p, t=t, qv=qv, dz8w=dz, nwfa=nwfa, nifa=nifa)
    out = {k: np.asarray(v, F) for k, v in out.items()}
    out["ht"] = rng.uniform(0.0, 3000.0, ncol).astype(F)
    return out


def main() -> None:
    build = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    work = build / "aer3_work"
    work.mkdir(parents=True, exist_ok=True)
    i = inputs()
    ncol, nz = i["p"].shape
    with open(work / "aer3_in.bin", "wb") as fh:
        fh.write(np.array([ncol, nz], np.int32).tobytes())
        for name in ("p", "t", "qv", "dz8w", "nwfa", "nifa"):
            # Fortran (ncol, nz): column fastest -> C order (nz, ncol)
            fh.write(np.ascontiguousarray(i[name].T).tobytes())
        fh.write(i["ht"].tobytes())
    subprocess.run([str(build / "oracle_aer3")], cwd=work, check=True)
    raw = np.fromfile(work / "aer3_out.bin", dtype=F)
    nb = 14
    block = ncol * nz * nb
    assert raw.size == 3 * block + ncol * nz + ncol, raw.size
    arrays = {}
    for n, name in enumerate(("tauaer", "ssaaer", "asyaer")):
        arrays[f"out/{name}"] = raw[n * block:(n + 1) * block].reshape(
            nb, nz, ncol).transpose(2, 1, 0).copy()
    rest = raw[3 * block:]
    arrays["out/taod5503d"] = rest[:ncol * nz].reshape(nz, ncol).T.copy()
    arrays["out/taod5502d"] = rest[ncol * nz:].copy()
    receipt = json.loads((build / "receipt.json").read_text(encoding="utf-8"))
    np.savez(out, **{f"in/{k}": v for k, v in i.items()}, **arrays,
             receipt=json.dumps(receipt))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out} sha256 {digest}")


if __name__ == "__main__":
    main()
