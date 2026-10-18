"""Column oracle: WOOF's topo-radiation kernels against WRF v4.6.1's Fortran.

    python column_oracle.py oracle  <BUILD_DIR> <REFS.npz>
                                    [--stock NAME] [--libm NAME=ROOT ...]
    python column_oracle.py fixture <REFS.npz> <FIXTURE.npz>
    python column_oracle.py woof    <REFS.npz> <WOOF.npz>        (GPU)
    python column_oracle.py compare <REFS.npz> <WOOF.npz> [<WOOF.npz> ...]
                                    [--json OUT.json]

``BUILD_DIR`` holds ``oracle`` and ``oracle_crlibm`` from build.sh.  The
case set (``cases()``) is a list of realistic patches, each labelled with a
regime: convective noon, grazing sun, stable night, snow and ice, open
ocean, very high terrain, a rotated grid with negative COSA, an overhead
tropical sun, a nest with its parent's shadow on the edges, and the
threshold edges (COSZEN at 1e-4 and 1e-3, SWDOWN at 1e-3, flat cells,
diffuse fraction 0 and 1, a diffuse flux above the total).  Every grid
point of a patch is one column.

Each patch runs WRF's whole slope_rad / topo_shading chain, every stage fed
by that side's own previous stage, as wrf.exe chains them:

1. SLOPE, SLP_AZI from HGT            (start_em.F)
2. SHADOWMASK, HT_SHAD [, HT_LOC]      (toposhad_init + toposhad)
3. SWDDIF, DIFFUSE_FRAC                (radiation driver; Ruiz-Arias split
                                        for a scheme without a direct beam)
4. SWDOWN, GSW, SWNORM, GSWSAVE        (TOPO_RAD_ADJ_DRVR + TOPO_RAD_ADJ)
5. the restore after the land surface  (surface_driver; WOOF side checked
                                        against stage 4's WRF values)

COSZEN, HRANG, DECLIN and SOLCON are inputs (the radiation driver's own
calc_coszen / radconst, outside this scheme), computed once on the host
and handed identically to both sides.

Reference sides, each the same WRF objects:

* ``wrf_<STOCK>`` -- linked to the box's stock libm, as wrf.exe runs on
  the box (``--stock`` names it, e.g. glibc239);
* ``cr``   -- linked ahead of libm_cr.c (correctly rounded float words), a
  diagnostic that separates a libm difference from an arithmetic one;
* ``wrf_<NAME>`` -- with ``--libm NAME=ROOT``, the stock ``oracle`` run
  under another C library unpacked at ROOT (its own ld.so, libc and libm,
  e.g. a distribution's libc6 package), so one box grades the port
  against WRF on more than one libm.

``compare`` states, per side, stage, field and regime, how many values
differ and by how many ULP (float32 bit distance), and the largest absolute
difference.  ``fixture`` packs the inputs and the ``wrf_*`` sides for
tests/test_topo_radiation.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

F = np.float32
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ALPS = (REPO / "tests" / "fixtures" / "wrf_topo_radiation_v471"
        / "topo_radiation_v471.npz")

# ---------------------------------------------------------------------------
# Case set
# ---------------------------------------------------------------------------


def _mountains(rng, ny, nx, dx, *, peaks, lo, hi, base=0.0, width_km=(3, 9)):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    h = np.full((ny, nx), base)
    for _ in range(peaks):
        cy, cx = rng.uniform(0, ny), rng.uniform(0, nx)
        amp = rng.uniform(lo, hi)
        width = rng.uniform(*width_km) * 1000.0 / dx
        h += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / width ** 2)
    return h


def _latlon(ny, nx, lat0, lon0, dx):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    dlat = dx / 111_000.0
    xlat = lat0 + dlat * (y - ny / 2)
    xlong = lon0 + dlat * (x - nx / 2) / np.cos(np.radians(lat0))
    return xlat.astype(F), xlong.astype(F)


def _rotation(ny, nx, amp, *, negative=False):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    sina = amp * np.sin(0.07 * x + 0.05 * y)
    cosa = np.sqrt(1.0 - sina ** 2)
    if negative:
        cosa[:, : nx // 2] *= -1.0
    return sina.astype(F), cosa.astype(F)


def _sun(julian, xtime, gmt, radt, xlat, xlong):
    from gpuwm.core.rrtmg_legacy import calc_coszen_hrang, radconst
    declin, solcon = radconst(F(julian))
    coszen, hrang = calc_coszen_hrang(
        F(julian), F(xtime) + F(radt) * F(0.5), F(gmt),
        xlat.reshape(-1), xlong.reshape(-1), declin)
    return (F(declin), F(solcon), coszen.reshape(xlat.shape),
            hrang.reshape(xlat.shape))


def cases():
    """The patches, each a dict of float32/int32 inputs plus scalars."""
    rng = np.random.default_rng(20261007)
    alps = np.load(ALPS)["slope/alps_geogrid/ht"].astype(F)
    out = []

    def add(name, regime, ht, dx, lat0, lon0, *, julian, xtime, gmt=0.0,
            radt=10.0, ruiz, albedo, clear=0.75, diffuse=0.15,
            shading=True, nested=False, rot=0.12, negative_cosa=False,
            msft=1.0, shadlen=25000.0, edit=None):
        ht = np.asarray(ht, F)
        ny, nx = ht.shape
        xlat, xlong = _latlon(ny, nx, lat0, lon0, dx)
        sina, cosa = _rotation(ny, nx, rot, negative=negative_cosa)
        declin, solcon, coszen, hrang = _sun(julian, xtime, gmt, radt,
                                             xlat, xlong)
        # SWDOWN: a clear-sky transmissivity with patchy cloud, zero at
        # night, as a shortwave scheme hands it to the surface.
        tau = np.clip(clear - 0.45 * (rng.random((ny, nx)) < 0.2)
                      * rng.random((ny, nx)), 0.05, 1.0)
        swdown = (solcon * np.maximum(coszen, 0.0) * tau).astype(F)
        swdown[coszen <= 0] = F(0.0)
        alb = np.broadcast_to(np.asarray(albedo, np.float64), (ny, nx))
        gsw = ((1.0 - alb) * swdown).astype(F)
        if ruiz:
            swddif = np.zeros((ny, nx), F)
        else:
            dfr = np.clip(diffuse + 0.6 * (tau < 0.5) + 0.05
                          * rng.standard_normal((ny, nx)), 0.02, 1.0)
            swddif = (dfr * swdown).astype(F)
        parent = (ht + rng.uniform(-300.0, 600.0, (ny, nx))).astype(F)
        c = dict(name=name, regime=regime, ht=ht,
                 msft=np.full((ny, nx), msft, F), sina=sina, cosa=cosa,
                 xlat=xlat, xlong=xlong, coszen=coszen, hrang=hrang,
                 swdown=swdown, gsw=gsw, swddif_in=swddif,
                 parent_shadow=parent,
                 scalars=np.asarray([dx, xtime, gmt, radt, declin, solcon,
                                     shadlen], F),
                 flags=np.asarray([int(ruiz), int(shading), int(nested)],
                                  np.int32))
        if edit is not None:
            edit(c)
        out.append(c)

    # 1. Convective summer noon over the real Alps (RRTMG diffuse split).
    add("alps_noon", "convective", alps, 1000.0, 46.2, 8.3, julian=172.4,
        xtime=690.0, ruiz=False, albedo=0.18, clear=0.72)
    # 2. Morning sun low over the Alps, a nest (parent shadow on edges),
    #    Dudhia-style shortwave (Ruiz-Arias split).
    add("alps_morning_nest", "grazing", alps, 1000.0, 46.2, 8.3,
        julian=172.4, xtime=240.0, ruiz=True, albedo=0.18, nested=True)
    # 3. Stable night over the Alps: no sun anywhere.
    add("alps_night", "stable_night", alps, 1000.0, 46.2, 8.3,
        julian=10.0, xtime=1320.0, ruiz=True, albedo=0.2)
    # 4. A diurnal sweep over the Alps: the sun in every quarter.
    for k, xt in enumerate((300.0, 420.0, 540.0, 780.0, 900.0, 1020.0,
                            1110.0)):
        add(f"alps_sweep{k}", "diurnal_sweep", alps, 1000.0, 46.2, 8.3,
            julian=100.0 + 30 * k, xtime=xt, ruiz=bool(k % 2),
            albedo=0.18, nested=bool(k % 3 == 0))
    # 5. Snow and ice: a high polar plateau, low spring sun, albedo 0.82.
    plateau = _mountains(rng, 36, 44, 3000.0, peaks=7, lo=400.0, hi=1500.0,
                         base=2600.0, width_km=(8, 30))
    add("polar_plateau_snow", "snow_ice", plateau, 3000.0, 72.5, -38.0,
        julian=110.0, xtime=900.0, ruiz=True, albedo=0.82, clear=0.8)
    glacier = _mountains(rng, 32, 40, 1000.0, peaks=5, lo=1500.0, hi=3000.0,
                         base=1800.0)
    add("glacier_valley_snow", "snow_ice", glacier, 1000.0, 61.0, 7.0,
        julian=60.0, xtime=660.0, ruiz=False, albedo=0.7, clear=0.7)
    # 6. Open ocean: flat at 0 m (slope 0 everywhere), overcast patches
    #    with an all-diffuse sky, tropical noon.
    def overcast(c):
        sw = c["swdown"]
        dif = c["swddif_in"]
        dif[:8, :] = sw[:8, :]                          # diffuse fraction 1
        dif[8:12, :] = (sw[8:12, :] * F(1.2)).astype(F)  # above the total
    add("ocean_tropics", "ocean", np.zeros((24, 32), F), 9000.0, 12.0,
        -140.0, julian=200.0, xtime=1290.0, ruiz=False, albedo=0.06,
        edit=overcast)
    # Ocean with an island (a coast), Ruiz-Arias split, no shading.
    isle = _mountains(rng, 30, 30, 3000.0, peaks=2, lo=300.0, hi=900.0)
    isle[isle < 5.0] = 0.0
    add("ocean_island", "ocean", isle, 3000.0, -17.6, -149.5, julian=20.0,
        xtime=1260.0, ruiz=True, albedo=0.07, shading=False)
    # 7. Very high terrain: 8 km peaks, winter afternoon, deep shadows.
    hima = _mountains(rng, 44, 52, 1000.0, peaks=9, lo=2500.0, hi=5000.0,
                      base=3000.0, width_km=(2, 7))
    add("himalaya_winter", "high_terrain", hima, 1000.0, 28.0, 86.9,
        julian=355.0, xtime=540.0, ruiz=False, albedo=0.3, msft=1.02)
    add("himalaya_dusk_nest", "high_terrain", hima, 1000.0, 28.0, 86.9,
        julian=355.0, xtime=650.0, ruiz=True, albedo=0.3, nested=True)
    # 8. A rotated grid with negative COSA (rotated-pole / polar Lambert),
    #    southern-hemisphere summer.
    andes = _mountains(rng, 40, 36, 3000.0, peaks=6, lo=1000.0, hi=4500.0)
    add("andes_rotated", "rotated_grid", andes, 3000.0, -33.0, -70.0,
        julian=355.0, xtime=1000.0, ruiz=True, albedo=0.2, rot=0.6,
        negative_cosa=True, msft=0.97)
    # 9. Overhead tropical sun (COSZEN near 1, the azimuth clamp).
    hills = _mountains(rng, 28, 28, 3000.0, peaks=4, lo=200.0, hi=1200.0)
    add("tropics_overhead", "overhead_sun", hills, 3000.0, 23.4, 0.0,
        julian=172.0, xtime=720.0, ruiz=False, albedo=0.15, rot=0.0)
    # 10. The thresholds, set exactly on the stage inputs.
    def edges(c):
        cz, sw = c["coszen"], c["swdown"]
        cz.flat[0:12] = F(1.0e-4)     # TOPO_RAD_ADJ's RETURN-IF-NIGHT
        cz.flat[12:24] = F(1.0e-3)    # the Ruiz-Arias cut
        cz.flat[24:30] = F(1.0)
        sw.flat[30:42] = F(1.0e-3)    # the daytime cut
        sw.flat[42:48] = F(1.1e-3)
        sw.flat[48:54] = F(0.0)
        c["gsw"][...] = (F(0.8) * sw).astype(F)
    add("threshold_edges", "edges", hills, 3000.0, 40.0, -105.0,
        julian=80.0, xtime=780.0, ruiz=True, albedo=0.2, edit=edges)
    def edges_rrtmg(c):
        sw, dif = c["swdown"], c["swddif_in"]
        sw.flat[0:12] = F(1.0e-3)
        sw.flat[12:20] = F(0.0)
        dif.flat[20:40] = sw.flat[20:40]
        dif.flat[40:60] = F(0.0)
        c["gsw"][...] = (F(0.8) * sw).astype(F)
    add("threshold_edges_rrtmg", "edges", hills, 3000.0, 40.0, -105.0,
        julian=80.0, xtime=780.0, ruiz=False, albedo=0.2, edit=edges_rrtmg)
    return out


# ---------------------------------------------------------------------------
# WRF side: the oracle executables
# ---------------------------------------------------------------------------


def _f(a):
    return np.ascontiguousarray(np.asarray(a, F)).tobytes()


def _i(*v):
    return np.asarray(v, np.int32).tobytes()


def _s(*v):
    return np.asarray(v, F).tobytes()


def _run(binary, payload, nbytes):
    """``binary`` is a path or a full argv list (a loader prefix)."""
    argv = [str(b) for b in binary] if isinstance(binary, list) else [
        str(binary)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.bin").write_bytes(payload)
        subprocess.run(argv, cwd=tmp, check=True)
        out = (tmp / "out.bin").read_bytes()
    if len(out) != nbytes:
        raise SystemExit(f"{binary} wrote {len(out)} bytes, not {nbytes}")
    return out


def _planes(raw, count, shape, ints=()):
    out, off = [], 0
    n = shape[0] * shape[1]
    for k in range(count):
        dt = np.int32 if k in ints else F
        out.append(np.frombuffer(raw[off:off + 4 * n], dt).reshape(shape)
                   .copy())
        off += 4 * n
    return out


def wrf_chain(binary, c):
    ht = c["ht"]
    ny, nx = ht.shape
    shape = (ny, nx)
    dx, xtime, gmt, radt, declin, solcon, shadlen = (F(v) for v in
                                                     c["scalars"])
    ruiz, shading, nested = (int(v) for v in c["flags"])
    r = {}
    rdx = F(1.0) / dx
    raw = _run(binary, _i(1, nx, ny) + _f(ht) + _f(c["msft"])
               + _f(c["msft"]) + _f(c["sina"]) + _f(c["cosa"])
               + _s(rdx, rdx), 16 * nx * ny)
    r["slope"], r["slp_azi"], _, _ = _planes(raw, 4, shape)
    if shading:
        shad_in = c["parent_shadow"] if nested else np.zeros(shape, F)
        raw = _run(binary, _i(2, nx, ny) + _i(nested) + _f(ht)
                   + _f(c["xlat"]) + _f(c["xlong"]) + _f(c["sina"])
                   + _f(c["cosa"]) + _f(shad_in)
                   + _s(xtime, gmt, radt, declin, dx, dx, shadlen),
                   12 * nx * ny)
        r["shadowmask"], r["ht_shad"], loc = _planes(raw, 3, shape, (0,))
        if nested:
            r["ht_loc"] = loc
    mask = r.get("shadowmask", np.zeros(shape, np.int32))
    raw = _run(binary, _i(4, nx, ny) + _i(ruiz) + _f(c["coszen"])
               + _f(c["swdown"]) + _f(ht) + _f(c["swddif_in"]) + _s(solcon),
               8 * nx * ny)
    r["swddif"], r["diffuse_frac"] = _planes(raw, 2, shape)
    raw = _run(binary, _i(3, nx, ny) + _f(c["xlat"]) + _f(c["xlong"])
               + _f(c["coszen"]) + _f(r["diffuse_frac"]) + _f(c["swdown"])
               + _f(c["gsw"]) + _f(c["hrang"]) + _f(r["slope"])
               + _f(r["slp_azi"]) + mask.astype(np.int32).tobytes()
               + _s(declin, solcon), 16 * nx * ny)
    r["adj_swdown"], r["adj_gsw"], r["swnorm"], r["gswsave"] = _planes(
        raw, 4, shape)
    # The restore (module_surface_driver.F, after the land surface):
    # daytime points get the flat SWDOWN/GSW back and SWNORM the slope flux.
    day = r["swnorm"] > F(1.0e-3)
    r["rst_swdown"] = np.where(day, r["swnorm"], r["adj_swdown"]).astype(F)
    r["rst_gsw"] = np.where(day, r["gswsave"], r["adj_gsw"]).astype(F)
    r["rst_swnorm"] = np.where(day, r["adj_swdown"], r["swnorm"]).astype(F)
    return r


# ---------------------------------------------------------------------------
# WOOF side: the device kernels through their public entry points
# ---------------------------------------------------------------------------


def woof_chain(c):
    import cupy as cp

    from gpuwm.core.topo_radiation import (adjust_surface_shortwave,
                                           diffuse_fraction,
                                           restore_surface_shortwave,
                                           slope_geometry, terrain_shadow)
    ht = c["ht"]
    shape = ht.shape
    dx, xtime, gmt, radt, declin, solcon, shadlen = (float(v) for v in
                                                     c["scalars"])
    ruiz, shading, nested = (int(v) for v in c["flags"])
    r = {}
    slope, azi = slope_geometry(ht, c["msft"], c["msft"], c["sina"],
                                c["cosa"], dx=dx, dy=dx)
    r["slope"], r["slp_azi"] = cp.asnumpy(slope), cp.asnumpy(azi)
    mask = cp.zeros(shape, dtype=cp.int32)
    if shading:
        mask, shad = terrain_shadow(
            ht, c["xlat"], c["xlong"], c["sina"], c["cosa"],
            xtime_minutes=xtime, gmt=gmt, radt_minutes=radt, declin=declin,
            dx=dx, dy=dx, shadlen=shadlen,
            parent_shadow=c["parent_shadow"] if nested else None)
        r["shadowmask"], r["ht_shad"] = cp.asnumpy(mask), cp.asnumpy(shad)
        if nested:
            # ht_loc as toposhad_init leaves it on the patch (the edge
            # points a parent shadow raised).
            from gpuwm.core.topo_radiation import _grid, _kernel, widened_height
            loc = widened_height(ht)
            hs = cp.asarray(c["parent_shadow"], dtype=cp.float32).copy()
            mk = cp.zeros(shape, dtype=cp.int32)
            ny, nx = shape
            _kernel("topo_shadow_init")(*_grid(nx * ny), (
                loc, hs, mk, np.int32(1), np.int32(nx), np.int32(ny)))
            r["ht_loc"] = cp.asnumpy(loc)[2:-2, 2:-2]
    swddif = cp.asarray(c["swddif_in"], dtype=cp.float32).copy()
    frac = diffuse_fraction(c["coszen"], c["swdown"], ht, swddif,
                            solcon=solcon,
                            scheme_splits_direct_beam=not ruiz)
    r["swddif"], r["diffuse_frac"] = cp.asnumpy(swddif), cp.asnumpy(frac)
    swdown = cp.asarray(c["swdown"], dtype=cp.float32).copy()
    gsw = cp.asarray(c["gsw"], dtype=cp.float32).copy()
    save = adjust_surface_shortwave(
        swdown, gsw, xlat=c["xlat"], coszen=c["coszen"], shadowmask=mask,
        diffuse_frac=frac, hrang=c["hrang"], slope=slope, slp_azi=azi,
        declin=declin)
    r["adj_swdown"], r["adj_gsw"] = cp.asnumpy(swdown), cp.asnumpy(gsw)
    r["swnorm"], r["gswsave"] = (cp.asnumpy(save.swnorm),
                                 cp.asnumpy(save.gswsave))
    swnorm = restore_surface_shortwave(swdown, gsw, save)
    r["rst_swdown"], r["rst_gsw"] = cp.asnumpy(swdown), cp.asnumpy(gsw)
    r["rst_swnorm"] = cp.asnumpy(swnorm)
    return r


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

STAGE = {"slope": 1, "slp_azi": 1, "shadowmask": 2, "ht_shad": 2,
         "ht_loc": 2, "swddif": 3, "diffuse_frac": 3, "adj_swdown": 4,
         "adj_gsw": 4, "swnorm": 4, "gswsave": 4, "rst_swdown": 5,
         "rst_gsw": 5, "rst_swnorm": 5}


def _ulp(a, b):
    """Float32 distance in units in the last place (signed-magnitude map)."""
    def key(v):
        i = np.asarray(v, F).view(np.int32).astype(np.int64)
        return np.where(i < 0, -(i & 0x7FFFFFFF), i)
    return np.abs(key(a) - key(b))


def compare(ref, woof, names, regimes, side):
    rows, per_regime = [], {}
    for name, regime in zip(names, regimes):
        for field in STAGE:
            k_ref, k_w = f"{name}/{side}/{field}", f"{name}/woof/{field}"
            if k_ref not in ref.files:
                continue
            a, b = ref[k_ref], woof[k_w]
            if a.dtype == np.int32:
                nd = int((a != b).sum())
                mu, mad = (0 if nd == 0 else 1), float(np.abs(a - b).max())
            else:
                u = _ulp(a, b)
                nd, mu = int((u > 0).sum()), int(u.max())
                mad = float(np.abs(a.astype(np.float64)
                                   - b.astype(np.float64)).max())
            rows.append(dict(case=name, regime=regime, field=field,
                             stage=STAGE[field], values=int(a.size),
                             differ=nd, max_ulp=mu, max_abs=mad))
            g = per_regime.setdefault(regime, dict(values=0, differ=0,
                                                   max_ulp=0, max_abs=0.0,
                                                   columns=0))
            g["values"] += int(a.size)
            g["differ"] += nd
            g["max_ulp"] = max(g["max_ulp"], mu)
            g["max_abs"] = max(g["max_abs"], mad)
    for name, regime in zip(names, regimes):
        per_regime[regime]["columns"] += int(ref[f"{name}/ht"].size)
    total = dict(values=sum(r["values"] for r in rows),
                 differ=sum(r["differ"] for r in rows),
                 max_ulp=max(r["max_ulp"] for r in rows),
                 max_abs=max(r["max_abs"] for r in rows),
                 columns=sum(int(ref[f"{n}/ht"].size) for n in names),
                 cases=len(names))
    return dict(total=total, regimes=per_regime,
                differing=[r for r in rows if r["differ"]], rows=rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("build", type=Path)
    o.add_argument("out", type=Path)
    o.add_argument("--stock", default="stock",
                   help="label of the box's own libm side")
    o.add_argument("--libm", action="append", default=[],
                   help="NAME=ROOT: also run the stock oracle under the C "
                        "library unpacked at ROOT")
    w = sub.add_parser("woof")
    w.add_argument("refs", type=Path)
    w.add_argument("out", type=Path)
    x = sub.add_parser("fixture")
    x.add_argument("refs", type=Path)
    x.add_argument("out", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("refs", type=Path)
    c.add_argument("woof", type=Path, nargs="+")
    c.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    if a.cmd == "oracle":
        build = a.build.resolve()
        sides = [(f"wrf_{a.stock}", build / "oracle"),
                 ("cr", build / "oracle_crlibm")]
        for spec in a.libm:
            label, root = spec.split("=", 1)
            lib = Path(root).resolve() / "usr" / "lib" / "x86_64-linux-gnu"
            sides.append((f"wrf_{label}", [
                lib / "ld-linux-x86-64.so.2", "--library-path",
                f"{lib}:/usr/lib/x86_64-linux-gnu", build / "oracle"]))
        pack = {}
        names, regimes = [], []
        for case in cases():
            n = case["name"]
            names.append(n)
            regimes.append(case["regime"])
            for k, v in case.items():
                if isinstance(v, np.ndarray):
                    pack[f"{n}/{k}"] = v
            for side, exe in sides:
                for k, v in wrf_chain(exe, case).items():
                    pack[f"{n}/{side}/{k}"] = v
        pack["_names"] = np.asarray(names)
        pack["_regimes"] = np.asarray(regimes)
        pack["_sides"] = np.asarray([side for side, _ in sides])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, **pack)
        print(f"{len(names)} cases, "
              f"{sum(pack[f'{n}/ht'].size for n in names)} columns -> {a.out}")
        return 0

    if a.cmd == "fixture":
        ref = np.load(a.refs)
        pack = {k: ref[k] for k in ref.files if "/cr/" not in k}
        pack["_sides"] = np.asarray([s for s in ref["_sides"] if s != "cr"])
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out, **pack)
        print(f"{len(pack)} arrays -> {a.out}")
        return 0

    if a.cmd == "woof":
        import gpuwm  # noqa: F401  (installs the strict compiler when asked)
        from gpuwm import wrf_exact
        ref = np.load(a.refs)
        names = [str(s) for s in ref["_names"]]
        pack = {}
        for n in names:
            case = {k.split("/", 1)[1]: ref[k] for k in ref.files
                    if k.startswith(n + "/") and k.count("/") == 1}
            for k, v in woof_chain(case).items():
                pack[f"{n}/woof/{k}"] = v
        pack["_strict"] = np.asarray(int(wrf_exact.ENABLED))
        import cupy as cp
        pack["_device"] = np.asarray(
            cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
        np.savez_compressed(a.out, **pack)
        print(f"strict={wrf_exact.ENABLED} {len(names)} cases -> {a.out}")
        return 0

    ref = np.load(a.refs)
    names = [str(s) for s in ref["_names"]]
    regimes = [str(s) for s in ref["_regimes"]]
    sides = [str(s) for s in ref["_sides"]]
    report = {"sides": sides}
    for path in a.woof:
        woof = np.load(path)
        label = ("strict" if int(woof["_strict"]) else "default")
        report[label] = {"device": str(woof["_device"])}
        for side in sides:
            report[label][f"vs_{side}"] = compare(ref, woof, names, regimes,
                                                  side)
    stock = sides[0]
    report["libm_seams"] = {f"{stock}_vs_{side}": _self(ref, names, regimes,
                                                        stock, side)
                            for side in sides[1:]}
    for key, rep in report["libm_seams"].items():
        t = rep["total"]
        print(f"[WRF itself, {key}] {t['differ']} of {t['values']} values "
              f"differ, max {t['max_ulp']} ulp, max abs {t['max_abs']:.3g}")
    for label, rep in report.items():
        if label in ("sides", "libm_seams"):
            continue
        for side in (f"vs_{s}" for s in sides):
            t = rep[side]["total"]
            print(f"[{label} {side}] {t['cases']} cases, {t['columns']} "
                  f"columns, {t['values']} values: {t['differ']} differ, "
                  f"max {t['max_ulp']} ulp, max abs {t['max_abs']:.3g}")
            for r in rep[side]["differing"]:
                print(f"    {r['case']:24s} {r['field']:13s} "
                      f"{r['differ']:5d}/{r['values']:5d} "
                      f"max {r['max_ulp']} ulp ({r['max_abs']:.3g})")
    if a.json:
        a.json.write_text(json.dumps(report, indent=1) + "\n")
    return 0


def _self(ref, names, regimes, base, other):
    """WRF on the box's libm against WRF on another: a libm seam alone."""
    class _View:
        def __init__(self, f):
            self.f = f
            self.files = [k.replace(f"/{other}/", "/woof/") for k in f.files]

        def __getitem__(self, k):
            return self.f[k.replace("/woof/", f"/{other}/")]
    return compare(ref, _View(ref), names, regimes, base)


if __name__ == "__main__":
    sys.exit(main())
