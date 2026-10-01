"""Write the w_damp column set for the WRF 4.7.1 oracle, and pack the result.

    python synth.py DIR [seed]          # inputs + meta.txt
    w_damp_oracle DIR                   # WRF's rw_tend per case (wrf_*.bin)
    python synth.py --pack DIR OUT.npz  # inputs + WRF outputs, flat

A 16 x 12 column, 10-level domain whose vertical Courant numbers
(w_damp's ``|ww/(c1f*mut+c2f)*rdnw*dt|``) run from 0 to 3.5 on every
interior w level, so every case has cells on both sides of Courant 1 and
of Courant 2; ``w`` carries exact and negative zeros for SIGN.  The four
cases are w_crit_cfl 1.0 and 2.0, each with zadvect_implicit 0 and 1,
all at w_damping 1.  tests/test_w_crit_cfl.py grades gpuwm's ``w_damp``
kernel against ``tests/data/w_damp_wrf471.npz`` word for word.
"""

from __future__ import annotations

import os
import sys

import numpy as np

#: (tag, w_crit_cfl, zadvect_implicit).  The tag names the output file.
CASES = (("c1_i0", 1.0, 0), ("c1_i1", 1.0, 1),
         ("c2_i0", 2.0, 0), ("c2_i1", 2.0, 1))
NX, NY, NZ = 16, 12, 10
DT, DX, DY = 20.0, 3000.0, 3000.0


def write(directory: str, seed: int = 165) -> None:
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    f32 = np.float32
    nx, ny, nz = NX, NY, NZ

    def save(name, arr):
        np.ascontiguousarray(arr, dtype=f32).tofile(
            os.path.join(directory, name + ".bin"))

    znw = np.linspace(1.0, 0.0, nz + 1) ** 1.6
    rdnw = (1.0 / np.diff(znw)).astype(f32)                 # < 0
    c1f = np.clip(znw * 1.25 - 0.25, 0.0, 1.0).astype(f32)
    c2f = ((1.0 - c1f.astype(np.float64)) * 95000.0).astype(f32)
    mub = (f32(88000.0) + rng.normal(0, 1500, (ny, nx))).astype(f32)
    mup = rng.normal(0, 300, (ny, nx)).astype(f32)
    mut = mub + mup                                          # float32 sum
    # Omega for a target Courant number in [0, 3.5) on every interior level.
    target = rng.uniform(0.0, 3.5, (nz + 1, ny, nx))
    sign = np.where(rng.uniform(size=target.shape) < 0.5, -1.0, 1.0)
    m = c1f[:, None, None].astype(np.float64) * mut[None] \
        + c2f[:, None, None].astype(np.float64)
    scale = np.ones(nz + 1)
    scale[1:nz] = 1.0 / (np.abs(rdnw[1:nz].astype(np.float64)) * DT)
    ww = (sign * target * m * scale[:, None, None]).astype(f32)
    ww[0] = 0.0
    ww[nz] = 0.0
    w = rng.normal(0.0, 3.0, (nz + 1, ny, nx)).astype(f32)
    flat = w.reshape(-1)
    picks = rng.choice(flat.size, 24, replace=False)
    flat[picks[:12]] = f32(0.0)
    flat[picks[12:]] = f32(-0.0)
    rw_t = rng.normal(0.0, 50.0, (nz + 1, ny, nx)).astype(f32)
    u = rng.normal(0.0, 15.0, (nz, ny, nx + 1)).astype(f32)
    v = rng.normal(0.0, 15.0, (nz, ny + 1, nx)).astype(f32)
    for name, arr in (("rdnw", rdnw), ("c1f", c1f), ("c2f", c2f),
                      ("mub", mub), ("mup", mup), ("mut", mut), ("ww", ww),
                      ("w", w), ("rw_t", rw_t), ("u", u), ("v", v)):
        save(name, arr)
    with open(os.path.join(directory, "meta.txt"), "w") as meta:
        meta.write(f"{nx} {ny} {nz} {len(CASES)}\n")
        meta.write(f"{DT!r} {DX!r} {DY!r}\n")
        for tag, crit, ieva in CASES:
            meta.write(f"{tag} {crit!r} {ieva}\n")


def pack(directory: str, out: str) -> None:
    arrays = {}
    for name in sorted(os.listdir(directory)):
        if name.endswith(".bin"):
            arrays[name[:-4]] = np.fromfile(os.path.join(directory, name),
                                            dtype=np.float32)
    arrays["meta"] = np.array(
        open(os.path.join(directory, "meta.txt")).read().split())
    arrays["wrf_max_cfl"] = np.array(
        open(os.path.join(directory, "wrf_max_cfl.txt")).read().split())
    missing = [f"wrf_rw_t_{tag}" for tag, _, _ in CASES
               if f"wrf_rw_t_{tag}" not in arrays]
    if missing:
        raise SystemExit(f"the oracle has not run: {missing} absent")
    np.savez_compressed(out, **arrays)


def main() -> int:
    if sys.argv[1] == "--pack":
        pack(sys.argv[2], sys.argv[3])
    else:
        write(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 165)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
