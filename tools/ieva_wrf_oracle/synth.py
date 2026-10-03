"""Write a small synthetic IEVA capture for the WRF 4.7.1 oracle.

    python synth.py DIR [seed] [levels] [coordinate.json]

A 9 x 8 column specified domain (7 levels by default) with map factors, terrain and
vertical Courant numbers from 0 to 3.5 on the dynamics flux and the
scalar flux alike, in capture.py's layout.  ``ieva_oracle DIR chain``
then writes WRF's chained outputs (``wrf_*.bin``).  Every ``.bin`` plus
the tokens of ``meta.txt`` (key ``meta``) are packed, flat, into
``tests/data/ieva_wrf471.npz``, which tests/test_zadvect_implicit.py
grades gpuwm's kernels against.
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np


def main() -> int:
    directory = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 158
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    nz = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    if nz < 3:
        raise ValueError("the surface boundary requires at least three mass levels")
    ny, nx = 8, 9
    f32 = np.float32
    dt, dx, dy = f32(20.0), f32(3000.0), f32(3000.0)
    dt_s = f32(20.0)

    def save(name, arr):
        np.ascontiguousarray(arr, dtype=np.float32).tofile(
            os.path.join(directory, name + ".bin"))

    # A stretched hybrid ladder: thin near the ground, eta 1 -> 0.
    znw = np.linspace(1.0, 0.0, nz + 1) ** 1.6
    dnw = np.diff(znw)                                  # < 0
    rdnw = (1.0 / dnw).astype(f32)
    dn = np.zeros(nz)
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    rdn = np.zeros(nz)
    rdn[1:] = 1.0 / dn[1:]
    fnp = np.zeros(nz)
    fnm = np.zeros(nz)
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    c1f = np.clip(znw * 1.25 - 0.25, 0.0, 1.0)
    c2f = (1.0 - c1f) * 95000.0
    znu = 0.5 * (znw[1:] + znw[:-1])
    c1h = np.clip(znu * 1.25 - 0.25, 0.0, 1.0)
    c2h = (1.0 - c1h) * 95000.0
    coordinate = None
    if len(sys.argv) > 4:
        with open(sys.argv[4], encoding="utf-8") as source:
            coordinate = json.load(source)
        if len(coordinate["eta_levels"]) != nz + 1:
            raise ValueError("coordinate level count differs from requested levels")
        rdnw, rdn, fnp, fnm, c1f, c2f, c1h, c2h = (
            np.asarray(coordinate[name], dtype=np.float64)
            for name in ("rdnw", "rdn", "fnp", "fnm", "c1f", "c2f", "c1h", "c2h"))
        with open(os.path.join(directory, "coordinate.json"), "w", encoding="utf-8") as target:
            json.dump(coordinate, target, indent=2)
    for name, arr in (("rdnw", rdnw), ("rdn", rdn), ("fnp", fnp),
                      ("fnm", fnm), ("c1f", c1f), ("c2f", c2f),
                      ("c1h", c1h), ("c2h", c2h)):
        save(name, arr)

    mut = f32(92000.0) + rng.normal(0, 300, (ny, nx)).astype(f32)
    mut_old = mut + rng.normal(0, 20, (ny, nx)).astype(f32)
    save("mut", mut)
    save("mut_old", mut_old)
    mux = np.empty((ny, nx + 1), f32)
    mux[:, 1:nx] = f32(0.5) * (mut[:, 1:] + mut[:, :-1])
    mux[:, 0], mux[:, nx] = mut[:, 0], mut[:, -1]
    muy = np.empty((ny + 1, nx), f32)
    muy[1:ny] = f32(0.5) * (mut[1:] + mut[:-1])
    muy[0], muy[ny] = mut[0], mut[-1]
    save("mux", mux)
    save("muy", muy)
    save("msft", 1.0 + rng.uniform(-0.02, 0.04, (ny, nx)))
    save("msfu", 1.0 + rng.uniform(-0.02, 0.04, (ny, nx + 1)))
    save("msfv", 1.0 + rng.uniform(-0.02, 0.04, (ny + 1, nx)))
    save("ht", rng.uniform(0, 2500, (ny, nx)))

    u = rng.normal(5, 25, (nz, ny, nx + 1))
    v = rng.normal(-3, 25, (nz, ny + 1, nx))
    save("u", u)
    save("v", v)
    save("u0", u + rng.normal(0, 1, u.shape))
    save("v0", v + rng.normal(0, 1, v.shape))
    save("w0", rng.normal(0, 3, (nz + 1, ny, nx)))

    def omega(mass_cols):
        # ww such that |ww*dt*rdnw/(c1f*mut+c2f)| spans 0..3.5.
        cfl = rng.uniform(-3.5, 3.5, (nz + 1, ny, nx))
        mass = c1f[:, None, None] * mass_cols[None] + c2f[:, None, None]
        rd = np.concatenate([rdnw, rdnw[-1:]])[:, None, None]
        ww = cfl * mass / (rd * float(dt))
        ww[0] = 0.0
        ww[nz] = 0.0
        return ww

    save("ww", omega(mut))
    phb = np.cumsum(np.full(nz + 1, 9.81 * 600.0))[:, None, None] \
        + 9.81 * rng.uniform(0, 2500, (1, ny, nx))
    save("phb", np.broadcast_to(phb, (nz + 1, ny, nx)))
    php0 = rng.normal(0, 40, (nz + 1, ny, nx))
    save("php0", php0)
    save("php", php0 + rng.normal(0, 5, php0.shape))
    save("theta_old", 300 + 4 * np.arange(nz)[:, None, None]
         + rng.normal(0, 2, (nz, ny, nx)))
    save("ru_t_explicit", rng.normal(0, 30, (nz, ny, nx + 1)))
    save("rv_t_explicit", rng.normal(0, 30, (nz, ny + 1, nx)))
    save("rth_t_explicit", rng.normal(0, 50, (nz, ny, nx)))
    save("rph_t_explicit", rng.normal(0, 3e5, (nz + 1, ny, nx)))
    save("rw_t_explicit", rng.normal(0, 10, (nz + 1, ny, nx)))

    muts = mut + rng.normal(0, 10, (ny, nx)).astype(f32)
    save("muts", muts)
    save("mu0s", mut_old)
    save("ww_m", omega(muts))
    save("q_old", rng.uniform(1e-4, 2e-2, (nz, ny, nx)))
    save("q_tend_explicit", rng.normal(0, 2e-2, (nz, ny, nx)))

    def r(x):
        return repr(float(np.float32(x)))
    cf1, cf2, cf3 = f32(1.6), f32(-0.8), f32(0.2)
    if coordinate is not None:
        cf1, cf2, cf3 = (f32(coordinate[name]) for name in ("cf1", "cf2", "cf3"))
    with open(os.path.join(directory, "meta.txt"), "w",
              newline="\n") as fh:
        fh.write(f"{nx} {ny} {nz} 1\n")
        fh.write(f"{r(dt)} {r(dx)} {r(dy)} {r(dt_s)}\n")
        fh.write(f"{r(cf1)} {r(cf2)} {r(cf3)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
