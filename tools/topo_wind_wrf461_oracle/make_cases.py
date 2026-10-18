#!/usr/bin/env python3
"""Write the inputs of the topo_wind chain oracle (WRF v4.6.1).

topo_wind is two pieces of WRF that only make sense together:

* dyn_em/start_em.F:1539-1626 forms LAP_HGT from the terrain and CTOPO /
  CTOPO2 from it and from VAR_SSO (topo_wind = 1) or VAR (topo_wind = 2),
  once per domain, over a whole 2-D patch (the laplacian reads neighbours
  and clamps at the domain edge);
* phys/physics_mmm/bl_ysu.F90 reads CTOPO in its surface-drag diagonal and
  CTOPO2 in its 10 m wind blend, column by column.

So the oracle's unit is a terrain PATCH with one atmospheric column on
every cell.  Both sides run the chain: WRF's start_em block on the patch,
then bl_ysu_run on each cell with the CTOPO it produced; WOOF's
topo_wind_coefficients on the same patch, then launch_ysu with the
coefficients WOOF produced.  Nothing WRF computed is handed to WOOF.

Patches are synthetic but shaped like real terrain (ridges and valleys at
3 km, a coastline, an extreme high range, plains), plus a probe patch that
puts LAP_HGT and sqrt(VAR_SSO) exactly on, and one float32 step either side
of, every threshold start_em.F tests, and the smallest domains the
edge clamp can see (1x1, 2x1, 1x3, 2x2).  The columns come from the wide
YSU oracle's regime builders (tools/ysu_wrf461_oracle/make_wide_columns.py)
with the surface pressure set from the cell's terrain and the land mask
set from the patch, so the drag acts on convective, stable, snow-covered
and very-low-pressure columns, and on the YSU edge probes.

Files (little endian):

  topo-wind-patches.bin   per patch: int32[4] nx, ny, patch id, family id;
                          float32[ny*nx] ht, xland, var_sso, var2d
                          (i fastest: Fortran (i, j), NumPy (j, i))
  topo-wind-columns.bin   one wide-oracle column record per cell, patch by
                          patch, j outer and i inner (make_wide_columns.py
                          layout; its ctopo/ctopo2 words are ignored, the
                          chain forms them)
  topo-wind-cases.json    a summary
  topo-wind-shinhong.bin  what Shin-Hong reads beyond the YSU record, per
                          cell in the same order: int32 tke_diag; float32
                          corf, dx, dy; float32[nz] tke (the scheme's own SGS
                          TKE on entry).  Drawn from its own seed after the
                          rest, so the files above do not depend on it.

    python make_cases.py OUT_DIR                  the committed fixture
    python make_cases.py --sweep NPATCH SEED OUT_DIR   a randomized sweep
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "ysu_wrf461_oracle"))
import make_wide_columns as wide  # noqa: E402

F32 = np.float32
SEED = 20261008

FAMILIES = {
    1: "alpine-ridges",
    2: "coast",
    3: "threshold-probes",
    4: "plains",
    5: "tiny-domains",
    6: "extreme-high-range",
}


class Patch:
    def __init__(self, family: int, nx: int, ny: int):
        self.family = family
        self.nx, self.ny = nx, ny
        self.ht = np.zeros((ny, nx), np.float32)
        self.xland = np.ones((ny, nx), np.float32)
        self.var_sso = np.zeros((ny, nx), np.float32)
        self.var2d = np.zeros((ny, nx), np.float32)
        self.columns: list[dict] = []


def _psfc_for(ht: float, rng) -> float:
    """Surface pressure of a cell, from its height (scale height ~8 km)."""
    return float(101325.0 * np.exp(-max(ht, 0.0) / 8000.0)
                 * rng.uniform(0.985, 1.015))


def _column_for(patch: Patch, j: int, i: int, rng, chooser) -> dict:
    land = float(patch.xland[j, i]) < 1.5
    c = chooser(rng, land, float(patch.ht[j, i]))
    c.psfc = _psfc_for(float(patch.ht[j, i]), rng)
    c.topo = 1
    rec = c.finish()
    # One land mask: start_em.F and bl_ysu_run both read grid%xland.
    rec["scalars"]["xland"] = F32(1.0 if land else 2.0)
    # finish() puts u10/v10 at 0.85 of the first-level wind; over land
    # spread that so the ctopo2 blend has a measurable gap to close (an
    # edge probe's own 10 m wind, e.g. a subnormal one, is kept).
    if land and "u10" not in c.raw:
        f = F32(rng.uniform(0.35, 0.9))
        rec["scalars"]["u10"] = F32(rec["u"][0] * f)
        rec["scalars"]["v10"] = F32(rec["v"][0] * f)
    return rec


def _land_regime(rng, ht: float):
    """A land column fitting the cell's height."""
    r = rng.random()
    if ht > 2800.0 and r < 0.4:
        return wide.snow_ice(rng)
    if ht > 1200.0:
        return wide.high_terrain(rng, True) if r < 0.75 else (
            wide.stable_night(rng) if r < 0.9 else wide.convective_land(rng))
    if r < 0.4:
        return wide.convective_land(rng)
    if r < 0.75:
        return wide.stable_night(rng)
    if r < 0.85:
        return wide.snow_ice(rng)
    if r < 0.93:
        return wide.stratocumulus(rng)
    return wide.tropical(rng)


def _default_chooser(rng, land, ht):
    return _land_regime(rng, ht) if land else wide.ocean(rng)


def _fill_columns(patch: Patch, rng, chooser=_default_chooser) -> None:
    patch.columns = [_column_for(patch, j, i, rng, chooser)
                     for j in range(patch.ny) for i in range(patch.nx)]


def _subgrid(rng, shape, std_lo, std_hi):
    std = rng.uniform(std_lo, std_hi, shape)
    return (std * std).astype(np.float32)


# --------------------------------------------------------------------------
# patch families
# --------------------------------------------------------------------------

def alpine_ridges(rng, nx=12, ny=10) -> Patch:
    p = Patch(1, nx, ny)
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    phase = rng.uniform(0, 2 * np.pi, 3)
    ht = (2300.0 + 900.0 * np.sin(0.9 * ii + phase[0]) * np.cos(0.7 * jj + phase[1])
          + 450.0 * np.sin(1.7 * ii + 1.3 * jj + phase[2])
          + rng.normal(0.0, 60.0, (ny, nx)))
    p.ht[...] = np.maximum(ht, 600.0)
    p.var_sso[...] = _subgrid(rng, (ny, nx), 15.0, 520.0)
    p.var2d[...] = rng.uniform(40.0, 850.0, (ny, nx))
    _fill_columns(p, rng)
    return p


def coast(rng, nx=14, ny=8) -> Patch:
    p = Patch(2, nx, ny)
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    shore = 4 + (jj % 3 == 0)
    land = ii >= shore
    hills = 80.0 * (ii - shore) + 260.0 * np.sin(1.1 * jj + 0.6 * ii) ** 2
    p.ht[...] = np.where(land, np.maximum(hills, 2.0), 0.0)
    p.xland[...] = np.where(land, 1.0, 2.0)
    p.var_sso[...] = _subgrid(rng, (ny, nx), 0.5, 250.0)
    p.var2d[...] = rng.uniform(0.0, 400.0, (ny, nx))
    # Water cells carry statistics too: start_em.F reads them only on land
    # (topo_wind 1: ctopo stays 1 before the laplacian test; 2: vfac = 1).
    p.var_sso[1, 0] = 360000.0
    p.var2d[1, 0] = 900.0
    # A lake in a high basin (water, LAP_HGT > 0) and a lake on a summit
    # (water, LAP_HGT < -30): start_em.F applies the hill-top taper to
    # water cells as well, from ctopo = 1.
    p.ht[3, 10], p.xland[3, 10] = 900.0, 2.0
    p.ht[6, 12], p.xland[6, 12] = p.ht[6, 12] + 400.0, 2.0
    _fill_columns(p, rng)
    return p


def threshold_probes(rng) -> Patch:
    """Every start_em.F threshold, exactly and one float32 step either side.

    A flat plain at 0 m: one raised cell of height b has neighbours summing
    to 0, so LAP_HGT = (0 - 4b)/4 = -b exactly (4b and its quarter are
    exact for any float32 b of normal size).  Raised cells sit two apart so
    no cell is the neighbour of two of them.
    """
    nx, ny = 24, 8
    p = Patch(3, nx, ny)
    nxt = lambda x, d: np.nextafter(F32(x), F32(d))  # noqa: E731
    bumps = [F32(5.0), nxt(10, 0), F32(10.0), nxt(10, 100), F32(15.0),
             nxt(20, 0), F32(20.0), nxt(20, 100), F32(25.0), nxt(30, 0),
             F32(30.0), nxt(30, 100), F32(45.0), F32(-40.0)]
    cells = [(1, 1 + 2 * k) for k in range(11)] + [(4, 1 + 2 * k)
                                                    for k in range(3)]
    for (j, i), b in zip(cells, bumps):
        p.ht[j, i] = b
    # sqrt(VAR_SSO) on and around WRF's 2.718 test, spread over probe cells
    # and flat cells, and the topo_wind = 2 cap var2d*0.4/200+1.175 = 1.575
    # (var2d = 200) on and around it.
    roots = [F32(2.718), nxt(2.718, 9), nxt(2.718, 0), F32(0.0), F32(200.0),
             F32(np.e), nxt(np.e, 9), F32(1.0), F32(1.0e-3), F32(1000.0)]
    var2 = [F32(200.0), nxt(200, 0), nxt(200, 900), F32(0.0), F32(1.0e4),
            F32(199.99998), F32(200.00002)]
    flat = [(j, i) for j in range(ny) for i in range(nx)
            if (j, i) not in cells]
    for k, (j, i) in enumerate(cells + flat):
        p.var_sso[j, i] = F32(roots[k % len(roots)] * roots[k % len(roots)])
        p.var2d[j, i] = var2[k % len(var2)]
    # Edge columns of the YSU oracle (zero coupling, subnormal br, ...)
    # twice over, then regime columns.
    probes = wide.edge_probes() + wide.edge_probes()
    order = 0

    def chooser(rng_, land, ht):
        nonlocal order
        order += 1
        if order <= len(probes):
            return probes[order - 1]
        return _default_chooser(rng_, land, ht)
    _fill_columns(p, rng, chooser)
    return p


def plains(rng, nx=10, ny=6) -> Patch:
    p = Patch(4, nx, ny)
    p.ht[...] = (320.0 + rng.normal(0.0, 4.0, (ny, nx))).astype(np.float32)
    # Mostly smooth (sqrt(VAR_SSO) <= 2.718, ctopo = 1), some rougher.
    std = np.where(rng.random((ny, nx)) < 0.6, rng.uniform(0.0, 2.6, (ny, nx)),
                   rng.uniform(2.8, 40.0, (ny, nx)))
    p.var_sso[...] = (std * std).astype(np.float32)
    p.var2d[...] = rng.uniform(0.0, 120.0, (ny, nx))
    _fill_columns(p, rng)
    return p


def tiny(rng, nx, ny) -> Patch:
    p = Patch(5, nx, ny)
    p.ht[...] = rng.uniform(200.0, 2600.0, (ny, nx))
    p.var_sso[...] = _subgrid(rng, (ny, nx), 3.0, 400.0)
    p.var2d[...] = rng.uniform(10.0, 600.0, (ny, nx))
    _fill_columns(p, rng)
    return p


def extreme_high(rng, nx=8, ny=8) -> Patch:
    p = Patch(6, nx, ny)
    jj, ii = np.mgrid[0:ny, 0:nx].astype(np.float64)
    ht = 5200.0 + 2400.0 * np.cos(1.3 * ii) * np.sin(1.1 * jj + 0.4) \
        + rng.normal(0.0, 300.0, (ny, nx))
    p.ht[...] = np.clip(ht, 2500.0, 8600.0)
    p.var_sso[...] = _subgrid(rng, (ny, nx), 200.0, 1100.0)
    p.var2d[...] = rng.uniform(150.0, 1500.0, (ny, nx))

    def chooser(rng_, land, ht_):
        return wide.snow_ice(rng_) if rng_.random() < 0.5 else \
            wide.high_terrain(rng_, True)
    _fill_columns(p, rng, chooser)
    return p


def build_patches() -> list[Patch]:
    rng = np.random.default_rng(SEED)
    patches = [alpine_ridges(rng), coast(rng), threshold_probes(rng),
               plains(rng)]
    patches += [tiny(rng, nx, ny) for nx, ny in ((1, 1), (2, 1), (1, 3),
                                                 (2, 2))]
    patches.append(extreme_high(rng))
    return patches


def sweep_patches(n: int, seed: int) -> list[Patch]:
    rng = np.random.default_rng(seed)
    makers = (lambda r: alpine_ridges(r, int(r.integers(4, 20)),
                                      int(r.integers(4, 16))),
              lambda r: coast(r, int(r.integers(14, 22)),
                              int(r.integers(8, 14))),
              lambda r: plains(r, int(r.integers(3, 16)),
                               int(r.integers(3, 12))),
              lambda r: extreme_high(r, int(r.integers(3, 12)),
                                     int(r.integers(3, 12))),
              lambda r: tiny(r, int(r.integers(1, 4)), int(r.integers(1, 4))))
    return [makers[int(rng.integers(len(makers)))](rng) for _ in range(n)]


#: Grid lengths for the Shin-Hong consumer: the scale-aware partition moves
#: with sqrt(dx*dy), and topo_wind is a 1-12 km option.
SHINHONG_DX = (500.0, 1000.0, 3000.0, 9000.0, 27000.0)


def _shinhong_extras(records: list[dict], seed: int) -> bytes:
    """Coriolis, grid length and entry TKE per cell, for Shin-Hong."""
    rng = np.random.default_rng(seed)
    out = bytearray()
    for rec in records:
        nz = rec["nz"]
        tke_diag = int(rng.random() < 0.75)
        lat = rng.uniform(-75.0, 75.0)
        corf = 2.0 * 7.292e-5 * np.sin(np.radians(lat))
        dx = float(rng.choice(SHINHONG_DX))
        dy = dx * float(rng.choice((1.0, 1.0, 1.0, 4.0)))
        z = np.cumsum(np.asarray(rec["dz"], np.float64))
        zi = rng.uniform(150.0, 2500.0)
        peak = rng.uniform(0.05, 2.5)
        tke = np.maximum(peak * np.clip(1.0 - z / zi, 0.0, 1.0) ** 1.5,
                         0.005)
        out += struct.pack("<i3f", tke_diag, corf, dx, dy)
        out += np.asarray(tke, "<f4").tobytes()
    return bytes(out)


def write(patches: list[Patch], out: Path, *, seed: int = SEED) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "topo-wind-patches.bin").open("wb") as fh:
        for pid, p in enumerate(patches):
            fh.write(struct.pack("<4i", p.nx, p.ny, pid, p.family))
            for a in (p.ht, p.xland, p.var_sso, p.var2d):
                fh.write(np.ascontiguousarray(a, "<f4").tobytes())
    records = [rec for p in patches for rec in p.columns]
    for p in patches:
        assert len(p.columns) == p.nx * p.ny
    wide.write_records(records, out / "topo-wind-columns.bin")
    (out / "topo-wind-shinhong.bin").write_bytes(
        _shinhong_extras(records, seed + 1))
    summary = {
        "seed": SEED, "npatch": len(patches), "ncell": len(records),
        "families": {FAMILIES[f]: sum(p.nx * p.ny for p in patches
                                      if p.family == f) for f in FAMILIES},
        "regimes": {wide.REGIMES[r]: sum(1 for x in records
                                         if x["regime"] == r)
                    for r in wide.REGIMES},
        "nz": sorted({x["nz"] for x in records}),
        "dt": sorted({float(x["scalars"]["dt"]) for x in records}),
    }
    return summary


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "--sweep":
        summary = write(sweep_patches(int(argv[2]), int(argv[3])),
                        Path(argv[4]), seed=int(argv[3]))
        summary["seed"] = int(argv[3])
        print(json.dumps(summary))
        return 0
    if len(argv) != 2:
        print(__doc__)
        return 2
    out = Path(argv[1])
    summary = write(build_patches(), out)
    (out / "topo-wind-cases.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
