"""Cut a mixed land/water aer_opt = 3 fixture with the NOAA MPAS oracle.

Usage: python run_oracle.py BUILD_DIRECTORY SOURCE_DIRECTORY OUTPUT.npz
       [--profiles M] [--layers NZ]

The deck is M physical profiles (surface pressure 600-1040 hPa, surface
temperature 235-320 K with a tropospheric lapse rate and a stratospheric
floor, layers thickening upward) carrying the aerosol a Thompson or TEMPO
aerosol-aware forecast carries, plus edge columns that reach every branch
the routines take: gt_aod's RH clamps and its three index branches, all
four temperature indices, the number caps, QV zero, negative and
supersaturated (the Bolton RH clamps at 0 and 99).  Each profile appears
twice, once over land (xland 1) and once over water (xland 2), so the
fixture holds the rural and the maritime optics of the same air.  The
fixture records the inputs, the fork's outputs, the rural and maritime
tables read from the pinned source text, and the build receipt;
tests/test_rrtmg_aer3_landmask_type.py replays the inputs through the
NumPy twin and the CUDA kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

F = np.float32
NB = 14


def _qsat(p, t):
    """Saturation mixing ratio for building inputs only (not graded)."""
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    return 0.622 * es / np.maximum(p - es, 1.0)


def profiles(m, nz, seed=20261008):
    rng = np.random.default_rng(seed)
    psfc = rng.uniform(60000.0, 104000.0, m)
    tsfc = rng.uniform(235.0, 320.0, m)
    eta = np.linspace(0.0, 1.0, nz + 1)
    ptop = 2000.0
    pw = ptop + (psfc[:, None] - ptop) * (1.0 - eta[None, :]) ** 1.3
    p = 0.5 * (pw[:, :-1] + pw[:, 1:])
    z = 7000.0 * np.log(psfc[:, None] / p)
    t = np.maximum(tsfc[:, None] - 0.0065 * z,
                   205.0 + rng.uniform(-25, 10, (m, 1)))
    dz = 7000.0 * np.log(pw[:, :-1] / pw[:, 1:])
    qv = rng.uniform(0.0, 1.15, (m, nz)) * _qsat(p, t)
    nwfa = 10.0 ** rng.uniform(5.0, 11.3, (m, nz))
    nifa = 10.0 ** rng.uniform(1.0, 10.3, (m, nz))
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
    out = {k: np.concatenate([v, v]).astype(F) for k, v in out.items()}
    out["xland"] = np.concatenate([np.ones(m), np.full(m, 2.0)]).astype(F)
    ht = rng.uniform(0.0, 3000.0, m)
    out["ht"] = np.concatenate([ht, np.zeros(m)]).astype(F)
    return out


def source_tables(path: Path) -> dict[str, np.ndarray]:
    """raod/ssa/asy rows for types 1 and 3, read from the pinned text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    tables = {}
    for name in ("raod_lut", "ssa_lut", "asy_lut"):
        for kind, label in ((1, "rural"), (3, "maritime")):
            rows = []
            for r in range(1, 9):
                pat = re.compile(
                    rf"data\s*\(\s*{name}\(\s*{kind}\s*,\s*ib\s*,\s*{r}\s*\)"
                    r"\s*,\s*ib\s*=\s*1\s*,\s*N_BANDS\s*\)\s*/([^/]*)/", re.I)
                found = pat.findall(text)
                if len(found) != 1:
                    raise SystemExit(f"{name}({kind},:,{r}): {len(found)} matches")
                vals = [v.strip() for v in found[0].replace("&", " ").split(",")]
                if len(vals) != NB:
                    raise SystemExit(f"{name}({kind},:,{r}): {len(vals)} values")
                rows.append([np.float32(v) for v in vals])
            tables[f"src/{name}/{label}"] = np.array(rows, F)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profiles", type=int, default=40)
    parser.add_argument("--layers", type=int, default=30)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    work = build / f"work-{args.profiles}x{args.layers}"
    work.mkdir(parents=True, exist_ok=True)
    i = profiles(args.profiles, args.layers)
    ncol, nz = i["p"].shape
    with open(work / "aer3_mpas_in.bin", "wb") as fh:
        fh.write(np.array([ncol, nz], np.int32).tobytes())
        for name in ("p", "t", "qv", "dz8w", "nwfa", "nifa"):
            # Fortran (ncol, nz): column fastest -> C order (nz, ncol)
            fh.write(np.ascontiguousarray(i[name].T).tobytes())
        fh.write(i["xland"].tobytes())
        fh.write(i["ht"].tobytes())
    subprocess.run([str(build / "oracle_aer3_mpas")], cwd=work, check=True)
    raw = (work / "aer3_mpas_out.bin").read_bytes()
    block = ncol * nz * NB
    floats = np.frombuffer(raw[:4 * (3 * block + ncol * nz + ncol)], F)
    types = np.frombuffer(raw[4 * (3 * block + ncol * nz + ncol):], np.int32)
    assert types.size == ncol, types.size
    arrays = {}
    for n, name in enumerate(("tauaer", "ssaaer", "asyaer")):
        arrays[f"out/{name}"] = floats[n * block:(n + 1) * block].reshape(
            NB, nz, ncol).transpose(2, 1, 0).copy()
    rest = floats[3 * block:]
    arrays["out/taod5503d"] = rest[:ncol * nz].reshape(nz, ncol).T.copy()
    arrays["out/taod5502d"] = rest[ncol * nz:].copy()
    arrays["out/aer_type"] = types.copy()
    arrays.update(source_tables(args.source_dir
                                / "physics_wrf/module_ra_rrtmg_sw_aerosols.F"))
    receipt = json.loads((build / "receipt.json").read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{f"in/{k}": v for k, v in i.items()},
                        **arrays, receipt=json.dumps(receipt))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{args.output} sha256 {digest} columns {ncol} layers {nz}")


if __name__ == "__main__":
    main()
