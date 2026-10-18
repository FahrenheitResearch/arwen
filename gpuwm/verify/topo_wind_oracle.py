"""The topo_wind chain oracle: read it, run WOOF's chain on it, compare.

``tools/topo_wind_wrf461_oracle`` runs WRF v4.6.1's own statements end to
end on terrain patches: start_em.F:1539-1626 forms LAP_HGT, CTOPO and
CTOPO2 on the patch, then bl_ysu_run runs on every cell with the CTOPO it
formed, for topo_wind = 0, 1 and 2.  This module runs WOOF's chain on the
same patches the way the model does (``gpuwm.core.terrain_drag.
topo_wind_coefficients`` on the patch, then ``gpuwm.core.ysu.launch_ysu``
with the coefficients WOOF formed: the ``topo`` arm and the blended 10 m
wind for topo_wind = 1 and 2, the default entry point for 0) and compares
every word: the three static planes and every YSU output.

Nothing WRF computed is handed to WOOF, so a coefficient off by one ULP
shows twice: in the static plane and in the drag it feeds.

Nothing here knows a tolerance; the gate and the CLI decide.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.verify.ysu_wide_oracle import (LEVEL_FIELDS, REGIMES,
                                          SURFACE_FIELDS, _LEVELS, _SCALARS,
                                          _WRF_LEVELS)

TOPO_WIND_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
                 / "oracles" / "topo_wind")
FAMILIES = {1: "alpine-ridges", 2: "coast", 3: "threshold-probes",
            4: "plains", 5: "tiny-domains", 6: "extreme-high-range"}
#: reference -> file stem.  ``wrfflags`` is the reference: WRF's own
#: FCOPTIM (-O2 -ftree-vectorize -funroll-loops), the arithmetic a stock
#: wrf.exe performs; ``O0`` is the same source at -O0.
REFERENCES = {"wrfflags": "wrf-wrfflags", "O0": "wrf-O0"}
OPTIONS = (0, 1, 2)
STATIC_FIELDS = ("lap_hgt", "ctopo", "ctopo2")


@dataclass
class Cell:
    patch: int
    family: int
    j: int
    i: int
    nz: int
    topdown: int
    regime: int
    scalars: dict
    levels: dict
    p_interface: np.ndarray
    #: topo_wind option -> WRF words of that call
    wrf: dict = field(default_factory=dict)
    wrf_kpbl: dict = field(default_factory=dict)

    def wrf_finite(self, opt: int) -> bool:
        return all(np.all(np.isfinite(v)) for k, v in self.wrf[opt].items()
                   if k != "qmixtnp")


@dataclass
class Patch:
    index: int
    family: int
    nx: int
    ny: int
    ht: np.ndarray
    xland: np.ndarray
    var_sso: np.ndarray
    var2d: np.ndarray
    cells: list
    #: topo_wind option -> {lap_hgt, ctopo, ctopo2} WRF planes, (ny, nx)
    wrf_static: dict = field(default_factory=dict)


def _read(directory: Path, name: str) -> bytes:
    path = directory / name
    if path.exists():
        return path.read_bytes()
    return gzip.decompress((directory / (name + ".gz")).read_bytes())


def load_topo_wind_oracle(directory: Path | None = None, *,
                          reference: str = "wrfflags") -> list[Patch]:
    directory = Path(directory) if directory is not None else TOPO_WIND_DIR
    praw = _read(directory, "topo-wind-patches.bin")
    craw = _read(directory, "topo-wind-columns.bin")
    sraw = _read(directory, REFERENCES[reference] + "-static.bin")
    yraw = _read(directory, REFERENCES[reference] + "-ysu.bin")
    patches: list[Patch] = []
    pp = cp_ = sp = yp = 0
    while pp < len(praw):
        nx, ny, pid, fam = (int(x) for x in np.frombuffer(praw, "<i4", 4, pp))
        pp += 16
        planes = []
        for _ in range(4):
            planes.append(np.frombuffer(praw, "<f4", nx * ny, pp)
                          .reshape(ny, nx).astype(np.float32))
            pp += 4 * nx * ny
        cells = []
        for j in range(ny):
            for i in range(nx):
                nz, topdown, regime, _topo = (
                    int(x) for x in np.frombuffer(craw, "<i4", 4, cp_))
                cp_ += 16
                sc = np.frombuffer(craw, "<f4", len(_SCALARS), cp_)
                cp_ += 4 * len(_SCALARS)
                levels = {}
                for name in _LEVELS:
                    levels[name] = np.frombuffer(craw, "<f4", nz,
                                                 cp_).astype(np.float32)
                    cp_ += 4 * nz
                pin = np.frombuffer(craw, "<f4", nz + 1, cp_).astype(np.float32)
                cp_ += 4 * (nz + 1)
                cells.append(Cell(pid, fam, j, i, nz, topdown, regime,
                                  {k: np.float32(v) for k, v in
                                   zip(_SCALARS, sc)}, levels, pin))
        patch = Patch(pid, fam, nx, ny, *planes, cells)
        for opt in OPTIONS:
            st = {}
            for name in STATIC_FIELDS:
                st[name] = (np.frombuffer(sraw, "<f4", nx * ny, sp)
                            .reshape(ny, nx).astype(np.float32))
                sp += 4 * nx * ny
            patch.wrf_static[opt] = st
            for cell in cells:
                cell.wrf_kpbl[opt] = int(np.frombuffer(yraw, "<i4", 1, yp)[0])
                yp += 4
                s = np.frombuffer(yraw, "<f4", 5, yp).astype(np.float32)
                yp += 20
                w = dict(hpbl=s[0:1], wstar=s[1:2], delta=s[2:3], u10=s[3:4],
                         v10=s[4:5])
                for name in _WRF_LEVELS:
                    w[name] = np.frombuffer(yraw, "<f4", cell.nz,
                                            yp).astype(np.float32)
                    yp += 4 * cell.nz
                cell.wrf[opt] = w
        patches.append(patch)
    for what, pos, raw in (("columns", cp_, craw), ("static", sp, sraw),
                           ("ysu", yp, yraw)):
        if pos != len(raw):
            raise ValueError(f"{what} file has {len(raw) - pos} trailing bytes;"
                             " inputs and outputs are not from the same run")
    return patches


def run_port(patches: list[Patch]) -> dict:
    """WOOF's chain on every patch and option.

    Returns ``{(patch, opt): {"static": {...}, "cells": [per-cell dict]}}``
    with host arrays.
    """
    import cupy as cp

    from gpuwm.core.terrain_drag import topo_wind_coefficients
    from gpuwm.core.ysu import launch_ysu

    out = {}
    for p in patches:
        for opt in OPTIONS:
            ctopo, ctopo2, lap = topo_wind_coefficients(
                cp.asarray(p.ht), cp.asarray(p.xland), topo_wind=opt,
                var_sso=cp.asarray(p.var_sso), var2d=cp.asarray(p.var2d))
            static = {"lap_hgt": cp.asnumpy(lap), "ctopo": cp.asnumpy(ctopo),
                      "ctopo2": cp.asnumpy(ctopo2)}
            flat_c = ctopo.reshape(-1)
            flat_c2 = ctopo2.reshape(-1)
            per_cell: list[dict | None] = [None] * len(p.cells)
            groups: dict[tuple, list[int]] = {}
            for n, c in enumerate(p.cells):
                key = (c.nz, float(c.scalars["dt"]), c.topdown)
                groups.setdefault(key, []).append(n)
            for (nz, dt, topdown), members in sorted(groups.items()):
                cells = [p.cells[n] for n in members]
                m = len(cells)

                def lev(name):
                    a = np.stack([c.levels[name] for c in cells], axis=1)
                    return cp.asarray(np.ascontiguousarray(a.reshape(nz, 1, m)))

                def sfc(name):
                    a = np.asarray([c.scalars[name] for c in cells], np.float32)
                    return cp.asarray(np.ascontiguousarray(a.reshape(1, m)))

                pin = np.stack([c.p_interface for c in cells], axis=1)
                kwargs = {}
                if opt:
                    idx = cp.asarray(np.asarray(members, np.int64))
                    u10o = cp.zeros((1, m), dtype=cp.float32)
                    v10o = cp.zeros((1, m), dtype=cp.float32)
                    kwargs = dict(
                        topo=(cp.ascontiguousarray(flat_c[idx].reshape(1, m)),
                              cp.ascontiguousarray(flat_c2[idx].reshape(1, m))),
                        u10_out=u10o, v10_out=v10o)
                res = launch_ysu(
                    lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"),
                    lev("qi"), lev("p"),
                    cp.asarray(np.ascontiguousarray(pin.reshape(nz + 1, 1, m))),
                    lev("pi"), lev("dz"), lev("rthraten"),
                    psfc=sfc("psfc"), znt=sfc("znt"), ust=sfc("ust"),
                    hfx=sfc("hfx"), qfx=sfc("qfx"), wspd=sfc("wspd"),
                    br=sfc("br"), psim=sfc("psim"), psih=sfc("psih"),
                    xland=sfc("xland"), u10=sfc("u10"), v10=sfc("v10"),
                    dt=dt, ysu_topdown_pblmix=topdown,
                    # The referee is WRF v4.6.1's YSU (2.9.1 defaults to
                    # the WRF 4.8 generation).
                    ysu_version="wrf_461", **kwargs)
                host = {k: cp.asnumpy(v) for k, v in res.items()}
                if opt:
                    host["u10"] = cp.asnumpy(kwargs["u10_out"])
                    host["v10"] = cp.asnumpy(kwargs["v10_out"])
                else:
                    # The model leaves the surface layer's U10/V10 alone
                    # when topo_wind is off (gpuwm.core.physics._run_ysu).
                    host["u10"] = sfc("u10").get()
                    host["v10"] = sfc("v10").get()
                for jj, n in enumerate(members):
                    per_cell[n] = {k: (v[:, 0, jj] if v.ndim == 3
                                       else v[0, jj:jj + 1])
                                   for k, v in host.items()}
            out[(p.index, opt)] = {"static": static, "cells": per_cell}
    return out


@dataclass
class FieldStat:
    max_ulp: int = 0
    max_abs: float = 0.0
    differing: int = 0
    total: int = 0
    worst: tuple | None = None


def _bits_equal(a, b):
    a = np.ascontiguousarray(a, np.float32).view(np.uint32)
    b = np.ascontiguousarray(b, np.float32).view(np.uint32)
    return a == b


def _accumulate(st: FieldStat, got, want, where):
    g = np.ascontiguousarray(got, np.float32).reshape(-1)
    w = np.ascontiguousarray(want, np.float32).reshape(-1)
    ulp = np.asarray(fp32_ulp_distance(g, w)).astype(np.int64)
    diff = ~_bits_equal(g, w)
    with np.errstate(invalid="ignore"):
        absd = np.abs(g.astype(np.float64) - w.astype(np.float64))
    absd = np.where(np.isfinite(absd), absd, np.inf)
    st.total += g.size
    st.differing += int(diff.sum())
    if ulp.size and int(ulp.max()) > st.max_ulp:
        k = int(np.argmax(ulp))
        st.max_ulp = int(ulp.max())
        st.worst = (*where, k, float(g[k]), float(w[k]))
    if absd.size:
        st.max_abs = max(st.max_abs, float(absd.max()))
    return int(ulp.max()) if ulp.size else 0, int(diff.sum())


YSU_FIELDS = dict(LEVEL_FIELDS, **SURFACE_FIELDS, u10="u10", v10="v10")


def compare(patches: list[Patch], port: dict, *, include=None) -> dict:
    """Per-field statistics.

    ``include(patch, opt, cell) -> bool`` selects YSU calls (default: every
    call whose WRF output is finite, on a column WRF's get_pblh stays
    inside: :func:`wrf_undefined_shallow`).  Static planes are always compared.
    A differing word counts even at 0 ULP (a signed zero).
    """
    if include is None:
        include = lambda p, opt, c: (  # noqa: E731
            c.wrf_finite(opt) and not wrf_undefined_shallow(c))
    static: dict[str, FieldStat] = {}
    ysu: dict[str, FieldStat] = {}
    kpbl_mismatch = []
    per_call = {}
    for p in patches:
        for opt in OPTIONS:
            got = port[(p.index, opt)]
            for name in STATIC_FIELDS:
                _accumulate(static.setdefault(f"{name}[{opt}]", FieldStat()),
                            got["static"][name], p.wrf_static[opt][name],
                            (p.index, opt))
            for n, c in enumerate(p.cells):
                if not include(p, opt, c):
                    continue
                g = got["cells"][n]
                worst = 0
                ndiff = 0
                for name, wname in YSU_FIELDS.items():
                    u, d = _accumulate(ysu.setdefault(name, FieldStat()),
                                       g[name], c.wrf[opt][wname],
                                       (p.index, opt, n))
                    worst = max(worst, u)
                    ndiff += d
                if int(g["kpbl"].reshape(-1)[0]) != c.wrf_kpbl[opt]:
                    kpbl_mismatch.append((p.index, opt, n,
                                          int(g["kpbl"].reshape(-1)[0]),
                                          c.wrf_kpbl[opt]))
                per_call[(p.index, opt, n)] = (worst, ndiff)
    return {"static": static, "ysu": ysu, "kpbl_mismatch": kpbl_mismatch,
            "per_call": per_call}


# --------------------------------------------------------------------------
# The Shin-Hong consumer (run_shinhong_topo_chain.F90)
# --------------------------------------------------------------------------

#: Shin-Hong reference -> file.  ``O2s`` (WRF's flags, vectoriser off, every
#: libm call scalar) is the reference, as in tools/shinhong_column_oracle;
#: ``wrfflags`` is WRF's flags exactly (libmvec powf); ``O0`` is -O0.
SHINHONG_REFERENCES = {"O2s": "wrf-O2s-shinhong.bin",
                       "wrfflags": "wrf-wrfflags-shinhong.bin",
                       "O0": "wrf-O0-shinhong.bin"}
SHINHONG_TOPO_FIELDS = {"u10": "u10", "v10": "v10"}


def load_shinhong_chain(patches: list[Patch], directory: Path | None = None,
                        *, reference: str = "O2s") -> dict:
    """``{(patch, opt, cell): shinhong_column_oracle.Column}``.

    Each Column carries the cell's YSU record re-expressed as the Shin-Hong
    oracle's record (psfc = p_interface(1), t = theta*pi as stored), the
    extras (tke_diag, corf, dx, dy, entry TKE) and the WRF words of that
    call.
    """
    from gpuwm.verify.shinhong_column_oracle import Column as ShColumn
    from gpuwm.verify.shinhong_column_oracle import _WRF_LEVELS as SH_LEVELS
    from gpuwm.verify.shinhong_column_oracle import _WRF_SURFACE as SH_SURFACE

    directory = Path(directory) if directory is not None else TOPO_WIND_DIR
    eraw = _read(directory, "topo-wind-shinhong.bin")
    oraw = _read(directory, SHINHONG_REFERENCES[reference])
    extras = []
    pos = 0
    for p in patches:
        for c in p.cells:
            tkd = int(np.frombuffer(eraw, "<i4", 1, pos)[0])
            corf, dx, dy = np.frombuffer(eraw, "<f4", 3, pos + 4)
            pos += 16
            tke = np.frombuffer(eraw, "<f4", c.nz, pos).astype(np.float32)
            pos += 4 * c.nz
            extras.append((tkd, np.float32(corf), np.float32(dx),
                           np.float32(dy), tke))
    if pos != len(eraw):
        raise ValueError("Shin-Hong extras do not match the columns")
    out = {}
    opos = 0
    k = 0
    for p in patches:
        base = k
        for opt in OPTIONS:
            for n, c in enumerate(p.cells):
                tkd, corf, dx, dy, tke = extras[base + n]
                s = c.scalars
                scal = {"dt": s["dt"], "dx": dx, "dy": dy,
                        "psfc": c.p_interface[0], "znt": s["znt"],
                        "ust": s["ust"], "hfx": s["hfx"], "qfx": s["qfx"],
                        "wspd": s["wspd"], "br": s["br"], "psim": s["psim"],
                        "psih": s["psih"], "xland": s["xland"],
                        "u10": s["u10"], "v10": s["v10"], "corf": corf}
                lv = c.levels
                lv = {"u": lv["u"], "v": lv["v"], "theta": lv["theta"],
                      "pi": lv["pi"], "t": lv["tx"], "qv": lv["qv"],
                      "qc": lv["qc"], "qi": lv["qi"], "p": lv["p"],
                      "dz": lv["dz"], "tke": tke}
                col = ShColumn(len(out), c.nz, tkd, p.family, n,
                               {kk: np.float32(v) for kk, v in scal.items()},
                               lv, c.p_interface)
                onz, kpbl, opatch, ocell = (
                    int(x) for x in np.frombuffer(oraw, "<i4", 4, opos))
                opos += 16
                if (onz, opatch, ocell) != (c.nz, p.index, n):
                    raise ValueError("Shin-Hong output record out of order")
                col.wrf_kpbl = kpbl
                for name in SH_LEVELS:
                    col.wrf[name] = np.frombuffer(oraw, "<f4", c.nz,
                                                  opos).astype(np.float32)
                    opos += 4 * c.nz
                sv = np.frombuffer(oraw, "<f4", len(SH_SURFACE),
                                   opos).astype(np.float32)
                opos += 4 * len(SH_SURFACE)
                for j, name in enumerate(SH_SURFACE):
                    col.wrf[name] = sv[j:j + 1]
                out[(p.index, opt, n)] = col
        k += len(p.cells)
    if opos != len(oraw):
        raise ValueError("Shin-Hong output has trailing bytes")
    return out


def run_port_shinhong(patches: list[Patch], cols: dict) -> dict:
    """WOOF's chain with Shin-Hong as the consumer: ``{key: outputs}``.

    topo_wind_coefficients on the patch, then launch_shinhong with WOOF's
    own coefficients (arm B) for topo_wind = 1 and 2, arm A for 0, grouped
    by what the launcher takes launch-wide (nz, dt, dx, dy, tke_diag).
    """
    import cupy as cp

    from gpuwm.core.shinhong import launch_shinhong
    from gpuwm.core.terrain_drag import topo_wind_coefficients

    result = {}
    for p in patches:
        for opt in OPTIONS:
            ctopo, ctopo2, _ = topo_wind_coefficients(
                cp.asarray(p.ht), cp.asarray(p.xland), topo_wind=opt,
                var_sso=cp.asarray(p.var_sso), var2d=cp.asarray(p.var2d))
            flat_c, flat_c2 = ctopo.reshape(-1), ctopo2.reshape(-1)
            groups: dict[tuple, list[int]] = {}
            for n in range(len(p.cells)):
                c = cols[(p.index, opt, n)]
                s = c.scalars
                key = (c.nz, float(s["dt"]), float(s["dx"]), float(s["dy"]),
                       c.tke_diag)
                groups.setdefault(key, []).append(n)
            for (nz, dt, dx, dy, tkd), members in sorted(groups.items()):
                mc = [cols[(p.index, opt, n)] for n in members]
                m = len(mc)

                def lev(name):
                    a = np.stack([c.levels[name] for c in mc], axis=1)
                    return cp.asarray(np.ascontiguousarray(a.reshape(nz, 1, m)))

                def sfc(name):
                    a = np.asarray([c.scalars[name] for c in mc], np.float32)
                    return cp.asarray(np.ascontiguousarray(a.reshape(1, m)))

                pin = np.stack([c.p_interface for c in mc], axis=1)
                kwargs = {}
                if opt:
                    idx = cp.asarray(np.asarray(members, np.int64))
                    kwargs = dict(
                        topo=(cp.ascontiguousarray(flat_c[idx].reshape(1, m)),
                              cp.ascontiguousarray(flat_c2[idx].reshape(1, m))),
                        u10_out=cp.zeros((1, m), cp.float32),
                        v10_out=cp.zeros((1, m), cp.float32))
                res = launch_shinhong(
                    lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"),
                    lev("qi"), lev("p"),
                    cp.asarray(np.ascontiguousarray(pin.reshape(nz + 1, 1, m))),
                    lev("pi"), lev("dz"), lev("tke"),
                    psfc=sfc("psfc"), znt=sfc("znt"), ust=sfc("ust"),
                    hfx=sfc("hfx"), qfx=sfc("qfx"), wspd=sfc("wspd"),
                    br=sfc("br"), psim=sfc("psim"), psih=sfc("psih"),
                    xland=sfc("xland"), u10=sfc("u10"), v10=sfc("v10"),
                    corf=sfc("corf"), dt=dt, dx=dx, dy=dy, tke_diag=tkd,
                    **kwargs)
                host = {k: cp.asnumpy(v) for k, v in res.items()}
                if opt:
                    host["u10"] = cp.asnumpy(kwargs["u10_out"])
                    host["v10"] = cp.asnumpy(kwargs["v10_out"])
                else:
                    host["u10"] = sfc("u10").get()
                    host["v10"] = sfc("v10").get()
                for jj, n in enumerate(members):
                    result[(p.index, opt, n)] = {
                        k: (v[:, 0, jj] if v.ndim == 3 else v[0, jj:jj + 1])
                        for k, v in host.items()}
    return result


def compare_shinhong(cols: dict, port: dict, *, include=None) -> dict:
    """The Shin-Hong oracle's own per-field comparison, plus u10/v10.

    ``include(key, col)`` selects calls (default all).  The level and
    surface fields go through gpuwm.verify.shinhong_column_oracle.compare
    (NaN geometry, the kpbl == kte tke/el exclusion, FTZ lanes); u10/v10,
    which that oracle never varies, are compared word for word here.
    """
    from gpuwm.verify.shinhong_column_oracle import compare as sh_compare

    keys = [k for k, c in cols.items() if include is None or include(k, c)]
    flat = [cols[k] for k in keys]
    by_index = {cols[k].index: port[k] for k in keys}
    res = sh_compare(flat, by_index)
    topo: dict[str, FieldStat] = {}
    for k in keys:
        for name, wname in SHINHONG_TOPO_FIELDS.items():
            _accumulate(topo.setdefault(name, FieldStat()), port[k][name],
                        cols[k].wrf[wname], k)
    res["topo"] = topo
    res["keys"] = keys
    return res


def wrf_undefined_shallow(cell: Cell) -> bool:
    """A column whose levels all sit at or below 500 m (zw(kte) <= 500).

    bl_ysu.F90 get_pblh searches ``do while (zw1d(k) .le. 500.)`` with no
    k <= kte bound and reads qke1d(kte), which bl_ysu_run never sets; on
    such a column WRF's YSU result is undefined (bounds_check.sh).
    """
    return float(np.sum(cell.levels["dz"][:-1], dtype=np.float64)) <= 500.0


def nonfinite_calls(patches: list[Patch]):
    return [(p.index, opt, n) for p in patches for opt in OPTIONS
            for n, c in enumerate(p.cells) if not c.wrf_finite(opt)]


__all__ = ["TOPO_WIND_DIR", "FAMILIES", "REGIMES", "REFERENCES", "OPTIONS",
           "STATIC_FIELDS", "YSU_FIELDS", "Patch", "Cell",
           "load_topo_wind_oracle", "run_port", "compare", "nonfinite_calls",
           "SHINHONG_REFERENCES", "load_shinhong_chain", "run_port_shinhong",
           "compare_shinhong", "wrf_undefined_shallow"]
