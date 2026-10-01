"""Run the WRF v4.7.1 slope/shading oracle over a case set; pack a fixture.

    python make_fixture.py <BUILD_DIR> <OUT.npz> [--real NAME=HGT.npy ...]

``BUILD_DIR`` holds ``oracle`` and ``oracle_crlibm`` from build.sh.  Every
case is run through BOTH: the port (gpuwm/core/topo_radiation.py) is held
bit for bit to ``oracle_crlibm``, and the fixture also records what stock
glibc gives so tests/test_topo_radiation.py can state the libm seam
(how many values differ, by how many ulp) instead of hiding it.

Cases are seeded synthetic planes chosen to reach every branch (flat cells
under the 1e-4 slope floor, negative COSA, every sun quarter, night and
grazing sun, a nest's parent shadow on each edge, SWDOWN at the daytime
threshold, diffuse fraction 0 and 1) plus, with ``--real``, real terrain.
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

F = np.float32


def _run(binary: Path, payload: bytes, nbytes: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.bin").write_bytes(payload)
        subprocess.run([str(binary)], cwd=tmp, check=True)
        out = (tmp / "out.bin").read_bytes()
    if len(out) != nbytes:
        raise SystemExit(f"{binary.name} wrote {len(out)} bytes, "
                         f"expected {nbytes}")
    return out


def _f(a) -> bytes:
    """Fortran order (i fastest) float32 bytes of a (ny, nx) plane."""
    return np.ascontiguousarray(np.asarray(a, F)).tobytes()


def _i(*values) -> bytes:
    return np.asarray(values, np.int32).tobytes()


def _s(*values) -> bytes:
    return np.asarray(values, F).tobytes()


def _planes(raw: bytes, count: int, ny: int, nx: int, dtypes=None):
    out, offset = [], 0
    for index in range(count):
        dt = np.dtype((dtypes or {}).get(index, np.float32))
        size = ny * nx * dt.itemsize
        out.append(np.frombuffer(raw[offset:offset + size], dt)
                   .reshape(ny, nx).copy())
        offset += size
    return out


def _terrain(rng, ny, nx, dx):
    """Mountains, a knife ridge, a flat plain and a sea at 0 m."""
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    h = np.zeros((ny, nx))
    for _ in range(6):
        cy, cx = rng.uniform(0, ny), rng.uniform(0, nx)
        amp = rng.uniform(800.0, 3500.0)
        width = rng.uniform(2.0, 9.0) * 3000.0 / dx
        h += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / width ** 2)
    ridge = np.abs(x - 0.6 * nx - 0.2 * y) < 1.0
    h[ridge] += 1500.0
    h[:, : nx // 6] = 0.0                     # sea
    h[: ny // 5, nx // 6: nx // 3] = 250.0   # flat plain
    return h.astype(F)


def _geometry(rng, ny, nx, *, negative_cosa=False):
    lat0, lon0 = rng.uniform(-60, 60), rng.uniform(-170, 170)
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    xlat = (lat0 + 0.03 * (y - ny / 2)).astype(F)
    xlong = (lon0 + 0.04 * (x - nx / 2)).astype(F)
    sina = (0.25 * np.sin(0.05 * x + 0.03 * y)).astype(F)
    cosa = np.sqrt(1.0 - sina.astype(np.float64) ** 2).astype(F)
    if negative_cosa:
        cosa[ny // 2:, : nx // 2] *= F(-1.0)
    return xlat, xlong, sina, cosa


def slope_cases(rng, reals):
    cases = []
    for k, (ny, nx, dx) in enumerate(((23, 31, 3000.0), (40, 48, 1000.0),
                                       (17, 19, 9000.0))):
        h = _terrain(rng, ny, nx, dx)
        _, _, sina, cosa = _geometry(rng, ny, nx, negative_cosa=(k == 1))
        msft = (1.0 + 0.04 * rng.standard_normal((ny, nx))).astype(F)
        cases.append((f"synthetic{k}", h, msft, msft, sina, cosa, dx))
    for name, h in reals:
        ny, nx = h.shape
        _, _, sina, cosa = _geometry(rng, ny, nx)
        msft = np.full((ny, nx), 1.01, F)
        cases.append((name, h, msft, msft, sina, cosa, 1000.0))
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("build", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--real", action="append", default=[],
                    help="NAME=HGT.npy, a (ny, nx) float32 terrain plane")
    args = ap.parse_args(argv)
    build = args.build.resolve()
    oracles = {"cr": build / "oracle_crlibm",
               "glibc": build / "oracle"}
    rng = np.random.default_rng(20260930)
    reals = [(spec.split("=", 1)[0], np.load(spec.split("=", 1)[1]).astype(F))
             for spec in args.real]
    pack: dict[str, np.ndarray] = {}

    # ---- mode 1: slope geometry --------------------------------------
    for name, h, mx, my, sina, cosa, dx in slope_cases(rng, reals):
        ny, nx = h.shape
        rdx = F(1.0) / F(dx)
        payload = (_i(1, nx, ny) + _f(h) + _f(mx) + _f(my) + _f(sina)
                   + _f(cosa) + _s(rdx, rdx))
        key = f"slope/{name}"
        pack.update({f"{key}/ht": h, f"{key}/msft": mx, f"{key}/sina": sina,
                     f"{key}/cosa": cosa, f"{key}/rdx": np.asarray(rdx, F)})
        for tag, binary in oracles.items():
            raw = _run(binary, payload, 4 * 4 * nx * ny)
            slope, azi, _, _ = _planes(raw, 4, ny, nx)
            pack[f"{key}/{tag}/slope"] = slope
            pack[f"{key}/{tag}/slp_azi"] = azi

    # ---- mode 2: shadows ---------------------------------------------
    shadow_cases = []
    for k, (ny, nx, dx) in enumerate(((36, 44, 3000.0), (48, 40, 1000.0))):
        h = _terrain(rng, ny, nx, dx)
        xlat, xlong, sina, cosa = _geometry(rng, ny, nx,
                                            negative_cosa=(k == 1))
        shadow_cases.append((f"synthetic{k}", h, xlat, xlong, sina, cosa, dx))
    for name, h in reals:
        ny, nx = h.shape
        xlat, xlong, sina, cosa = _geometry(rng, ny, nx)
        shadow_cases.append((name, h, xlat, xlong, sina, cosa, 1000.0))
    for name, h, xlat, xlong, sina, cosa, dx in shadow_cases:
        ny, nx = h.shape
        base = f"shadow/{name}"
        # A parent shadow on the outer two rows for the nested runs: above
        # the terrain on some points (it raises them), below on others.
        parent = (h + rng.uniform(-300, 600, (ny, nx))).astype(F)
        pack.update({f"{base}/ht": h, f"{base}/xlat": xlat,
                     f"{base}/xlong": xlong, f"{base}/sina": sina,
                     f"{base}/cosa": cosa, f"{base}/ht_shad_in": parent,
                     f"{base}/dx": np.asarray(dx, F)})
        for sun in range(8):
            # Hour angles through the day and both hemispheres' seasons,
            # so every quarter and the night/grazing cut are reached.
            declin = F(rng.uniform(-0.40, 0.40))
            xtime = F(60.0 * sun * 3.0 + rng.uniform(0, 50))
            gmt = F(rng.integers(0, 24))
            radt = F(rng.choice([0.0, 3.0, 10.0, 30.0]))
            pack[f"{base}/sun{sun}/scalars"] = np.asarray(
                [xtime, gmt, radt, declin, dx, dx, 25000.0], F)
            for nested in (0, 1):
                shad_in = parent if nested else np.zeros((ny, nx), F)
                payload = (_i(2, nx, ny) + _i(nested) + _f(h) + _f(xlat)
                           + _f(xlong) + _f(sina) + _f(cosa) + _f(shad_in)
                           + _s(xtime, gmt, radt, declin, dx, dx, 25000.0))
                key = f"{base}/sun{sun}/nested{nested}"
                outputs = {}
                for tag, binary in oracles.items():
                    raw = _run(binary, payload, 3 * 4 * nx * ny)
                    mask, shad, loc = _planes(raw, 3, ny, nx, {0: np.int32})
                    outputs[tag] = (mask, shad)
                    pack[f"{key}/{tag}/shadowmask"] = mask
                    pack[f"{key}/{tag}/ht_shad"] = shad
                    if nested:
                        pack[f"{key}/{tag}/ht_loc"] = loc

    # ---- mode 3: slope/shadow adjustment of SWDOWN and GSW -----------
    ny, nx = 64, 96
    shape = (ny, nx)
    xlat = F(rng.uniform(-70, 70, shape))
    xlong = F(rng.uniform(-180, 180, shape))
    coszen = F(rng.uniform(-0.2, 1.0, shape))
    coszen.flat[:40] = F(1.0e-4)                  # the RETURN-IF-NIGHT edge
    frac = F(rng.uniform(0, 1, shape))
    frac.flat[40:80] = F(1.0)
    frac.flat[80:120] = F(0.0)
    swdown = F(rng.uniform(0, 1100, shape))
    swdown.flat[120:160] = F(1.0e-3)               # the daytime threshold
    swdown[coszen <= 0] = F(0.0)
    gsw = (swdown * F(rng.uniform(0.6, 0.95, shape))).astype(F)
    hrang = F(rng.uniform(-np.pi, np.pi, shape))
    slope = F(rng.uniform(0, 0.9, shape))
    slope.flat[160:200] = F(0.0)
    slp_azi = F(rng.uniform(0, 2 * np.pi, shape))
    mask = rng.integers(0, 2, shape).astype(np.int32)
    declin, solcon = F(0.31), F(1361.7)
    payload = (_i(3, nx, ny) + _f(xlat) + _f(xlong) + _f(coszen) + _f(frac)
               + _f(swdown) + _f(gsw) + _f(hrang) + _f(slope) + _f(slp_azi)
               + mask.tobytes() + _s(declin, solcon))
    key = "adjust/random"
    pack.update({f"{key}/xlat": xlat, f"{key}/coszen": coszen,
                 f"{key}/diffuse_frac": frac, f"{key}/swdown": swdown,
                 f"{key}/gsw": gsw, f"{key}/hrang": hrang,
                 f"{key}/slope": slope, f"{key}/slp_azi": slp_azi,
                 f"{key}/shadowmask": mask,
                 f"{key}/scalars": np.asarray([declin, solcon], F)})
    for tag, binary in oracles.items():
        raw = _run(binary, payload, 4 * 4 * nx * ny)
        sw_out, gsw_out, swnorm, gswsave = _planes(raw, 4, ny, nx)
        pack[f"{key}/{tag}/swdown"] = sw_out
        pack[f"{key}/{tag}/gsw"] = gsw_out
        pack[f"{key}/{tag}/swnorm"] = swnorm
        pack[f"{key}/{tag}/gswsave"] = gswsave

    # ---- mode 4: diffuse fraction ------------------------------------
    for ruiz in (0, 1):
        coszen = F(rng.uniform(-0.1, 1.0, shape))
        coszen.flat[:30] = F(1e-3)
        swdown = F(rng.uniform(0, 1100, shape))
        swdown.flat[30:60] = F(0.001)
        swdown[coszen <= 0] = F(0.0)
        ht = F(rng.uniform(-50, 4500, shape))
        swddif = (np.zeros(shape, F) if ruiz else
                  (swdown * F(rng.uniform(0, 1.2, shape))).astype(F))
        solcon = F(1366.1)
        payload = (_i(4, nx, ny) + _i(ruiz) + _f(coszen) + _f(swdown)
                   + _f(ht) + _f(swddif) + _s(solcon))
        key = f"diffuse/ruiz{ruiz}"
        pack.update({f"{key}/coszen": coszen, f"{key}/swdown": swdown,
                     f"{key}/ht": ht, f"{key}/swddif_in": swddif,
                     f"{key}/solcon": np.asarray(solcon, F)})
        for tag, binary in oracles.items():
            raw = _run(binary, payload, 2 * 4 * nx * ny)
            dif, frac = _planes(raw, 2, ny, nx)
            pack[f"{key}/{tag}/swddif"] = dif
            pack[f"{key}/{tag}/diffuse_frac"] = frac

    # ---- mode 5: smooth_cg_topo's blend_terrain on d01 ---------------
    blend_cases = [("synthetic", _terrain(rng, 33, 41, 3000.0))]
    blend_cases += [(name, h) for name, h in reals]
    for name, h in blend_cases:
        ny, nx = h.shape
        # A source terrain: the same ground seen coarsely (a 5x5 box mean,
        # shifted), plus sea points below zero, as a global model's is.
        pad = np.pad(h.astype(np.float64), 2, mode="edge")
        coarse = sum(pad[dj:dj + ny, di:di + nx]
                     for dj in range(5) for di in range(5)) / 25.0
        coarse = (coarse - 35.0).astype(F)
        pack[f"blend/{name}/ht"] = h
        pack[f"blend/{name}/toposoil"] = coarse
        for sbw, width in ((5, 5), (5, 0), (1, 3), (3, 8)):
            payload = (_i(5, nx, ny) + _i(sbw, width) + _f(coarse) + _f(h))
            key = f"blend/{name}/sbw{sbw}_bw{width}"
            for tag, binary in oracles.items():
                raw = _run(binary, payload, 4 * nx * ny)
                (out,) = _planes(raw, 1, ny, nx)
                pack[f"{key}/{tag}/ht"] = out

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **pack)
    print(f"{len(pack)} arrays -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
