"""Cut the swint_opt = 1 fixture with the compiled fork routines.

Usage: python run_swint_oracle.py BUILD_DIRECTORY OUTPUT.npz

The column set is synthetic on purpose: it has to reach every branch of
update_swinterp_parameters and interp_sw_radiation (fresh columns, stored
references, exponents that clamp at -0.5 and 2.5, ratios below, at and
above one, night on either side of a call), which a real hour of columns
does not.  Three radiation calls per column, four between-call suns after
each.  The fixture records inputs, the fork's outputs, and the build
receipt; tests/test_swint_interpolation.py replays the inputs through the
engine twins and the kernels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

F = np.float32


def inputs(seed=20261003, ncol=4096, ncall=3, nloc=4):
    rng = np.random.default_rng(seed)
    cz = rng.uniform(-0.2, 1.0, (ncall, ncol)).astype(F)
    czcall = (cz + rng.uniform(-0.15, 0.15, (ncall, ncol))).astype(F)
    # exact-ratio columns (coszen == coszen_loc) and near-night columns
    cz[:, ::7] = czcall[:, ::7]
    cz[:, 3::11] = F(5.0e-5)
    czcall[:, 5::13] = F(-1.0e-5)
    # fluxes: a clear-sky-like direct beam plus noise, some tiny values so
    # max(1, flux) matters, some wild ratios so the exponent clamps
    swdown = (F(1100.0) * np.maximum(cz, 0.0) ** F(1.2)
              * rng.uniform(0.3, 1.1, (ncall, ncol))).astype(F)
    swddir = (swdown * rng.uniform(0.0, 0.95, (ncall, ncol))).astype(F)
    swdown[:, 1::17] = rng.uniform(0.0, 2.0, swdown[:, 1::17].shape)
    swddir[:, 2::19] = rng.uniform(0.0, 0.5, swddir[:, 2::19].shape)
    swdown[1, 4::23] = F(2000.0)
    swdown[2, 6::29] = F(0.1)
    czloc = rng.uniform(-0.1, 1.0, (ncall, nloc, ncol)).astype(F)
    czloc[:, :, 9::31] = F(0.0)
    albedo = rng.uniform(0.05, 0.9, ncol).astype(F)
    return dict(coszen=cz, czcall=czcall, swddir=swddir, swdown=swdown,
                czloc=czloc, albedo=albedo)


def coszen_inputs(n=2048, seed=7):
    """Points over the globe and calendar sets on both sides of radconst's
    JULIAN = 80 branch, across the day (gmt and xtime), so the per-step
    cosine is graded at sunrise, noon, sunset and night."""
    rng = np.random.default_rng(seed)
    sets = np.array([
        # julian, xtime (min), gmt (h)
        (165.75, 1050.0, 0.5),
        (15.25, 20.0, 12.0),
        (79.999, 719.6667, 18.0),
        (80.0, 1439.6667, 0.0),
        (172.5, 2880.3333, 6.0),
        (300.125, 45.0, 23.5),
        (364.99, 100000.0, 3.25),
    ], dtype=F)
    return dict(xlat=rng.uniform(-80.0, 80.0, n).astype(F),
                xlon=rng.uniform(-180.0, 180.0, n).astype(F),
                sets=sets)


def main() -> None:
    build = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    work = build / "swint_work"
    work.mkdir(parents=True, exist_ok=True)
    i = inputs()
    ncall, ncol = i["coszen"].shape
    nloc = i["czloc"].shape[1]
    with open(work / "swint_in.bin", "wb") as fh:
        fh.write(np.array([ncol, ncall, nloc], np.int32).tobytes())
        for name in ("coszen", "czcall", "swddir", "swdown"):
            # Fortran (ncol, 1, ncall): column fastest -> C order (ncall, ncol)
            fh.write(np.ascontiguousarray(i[name]).tobytes())
        fh.write(np.ascontiguousarray(i["czloc"]).tobytes())
        fh.write(np.ascontiguousarray(i["albedo"]).tobytes())
    c = coszen_inputs()
    with open(work / "coszen_in.bin", "wb") as fh:
        fh.write(np.array([c["xlat"].size, c["sets"].shape[0]],
                          np.int32).tobytes())
        fh.write(c["xlat"].tobytes())
        fh.write(c["xlon"].tobytes())
        fh.write(np.ascontiguousarray(c["sets"]).tobytes())
    subprocess.run([str(build / "oracle_swint")], cwd=work, check=True)
    raw = np.fromfile(work / "swint_out.bin", dtype=F)
    per_call = 7 * ncol + 5 * ncol * (1 + nloc)
    assert raw.size == ncall * per_call, (raw.size, ncall * per_call)
    raw = raw.reshape(ncall, per_call)
    out_arrays = {}
    for name_index, name in enumerate(("bb", "bx", "gg", "gx", "coszen_ref",
                                       "swdown_ref", "swddir_ref")):
        out_arrays[f"out/{name}"] = raw[:, name_index * ncol:(name_index + 1) * ncol]
    rest = raw[:, 7 * ncol:].reshape(ncall, 1 + nloc, 5, ncol)
    for name_index, name in enumerate(("swdown", "swddir", "swddni", "swddif",
                                       "gsw")):
        out_arrays[f"out/{name}"] = rest[:, :, name_index, :]
    n = c["xlat"].size
    nset = c["sets"].shape[0]
    cz_raw = np.fromfile(work / "coszen_out.bin", dtype=F)
    assert cz_raw.size == nset * (2 + 2 * n)
    cz_raw = cz_raw.reshape(nset, 2 + 2 * n)
    receipt = json.loads((build / "receipt.json").read_text(encoding="utf-8"))
    np.savez(out, **{f"in/{k}": v for k, v in i.items()}, **out_arrays,
             **{f"coszen_in/{k}": v for k, v in c.items()},
             **{"coszen_out/declin": cz_raw[:, 0],
                "coszen_out/solcon": cz_raw[:, 1],
                "coszen_out/coszen": cz_raw[:, 2:2 + n],
                "coszen_out/hrang": cz_raw[:, 2 + n:]},
             receipt=json.dumps(receipt))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out} sha256 {digest}")


if __name__ == "__main__":
    main()
