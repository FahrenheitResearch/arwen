"""The wide YSU oracle: read it, run the port on it, measure every word.

``tools/ysu_wrf461_oracle/make_wide_columns.py`` writes the columns and
``run_bl_ysu_wide.F90`` drives WRF v4.6.1's byte-unmodified ``bl_ysu_run``
on each of them with the argument set ``module_bl_ysu.F`` passes in a WRF
run.  This module reads both files, launches ``gpuwm.core.ysu.launch_ysu``
on the same columns, and returns the per-field, per-column comparison.

The port is launched the way the model launches it: the default entry
point (``ysu_column``) for every column whose ``ctopo = ctopo2 = 1`` and the
topo_wind entry point (``ysu_column_topo``) for the rest, which also writes
the blended 10 m wind.  Columns are grouped by everything ``launch_ysu``
takes launch-wide (nz, dt, ysu_topdown_pblmix, topo arm) and each group is
one launch, so neighbouring columns share a launch as they do in a run.

Nothing here knows a tolerance.  The gate (tests/test_ysu_wrf461_wide_parity.py)
and the CLI (tools/ysu_wrf461_oracle/compare_wide.py) decide what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "YSU_WIDE_DIR", "REFERENCES", "WideColumn", "load_wide_oracle", "run_port_wide",
    "compare_wide", "LEVEL_FIELDS", "SURFACE_FIELDS", "REGIMES",
]

YSU_WIDE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
                / "oracles" / "ysu_wide")

REGIMES = {
    1: "convective-land", 2: "stable-night-land", 3: "ocean",
    4: "high-terrain", 5: "high-terrain-topo-wind", 6: "snow-ice-cold",
    7: "stratocumulus-topdown", 8: "tropical-moist", 9: "edge-probe",
}

_SCALARS = ("dt", "psfc", "znt", "ust", "hfx", "qfx", "wspd", "br", "psim",
            "psih", "xland", "u10", "v10", "ctopo", "ctopo2", "uox", "vox",
            "_pad")
_LEVELS = ("u", "v", "theta", "pi", "tx", "qv", "qc", "qi", "p", "dz",
           "rthraten")
_WRF_LEVELS = ("utnp", "vtnp", "ttnp", "rthblten", "qvtnp", "qctnp",
               "qitnp", "exch_hx", "exch_mx", "qmixtnp")

#: port output -> WRF word, per level.  dtheta is module_bl_ysu.F:452's
#: ttnp/pi2d, the tendency WRF hands the solver.
LEVEL_FIELDS = {
    "du": "utnp", "dv": "vtnp", "dtheta": "rthblten", "dqv": "qvtnp",
    "dqc": "qctnp", "dqi": "qitnp", "exch_h": "exch_hx", "exch_m": "exch_mx",
}
#: port output -> WRF word, per column.  u10/v10 are compared only on the
#: topo_wind columns, the only ones where the port writes them.
SURFACE_FIELDS = {"hpbl": "hpbl", "wstar": "wstar", "delta": "delta"}
TOPO_FIELDS = {"u10": "u10", "v10": "v10"}


@dataclass
class WideColumn:
    index: int
    nz: int
    topdown: int
    regime: int
    topo: int
    scalars: dict[str, np.float32]
    levels: dict[str, np.ndarray]
    p_interface: np.ndarray
    wrf: dict[str, np.ndarray] = field(default_factory=dict)
    wrf_kpbl: int = 0

    @property
    def regime_name(self) -> str:
        return REGIMES[self.regime]

    @property
    def wrf_finite(self) -> bool:
        """False where WRF's own output is non-finite (its 0/0 columns)."""
        return all(np.all(np.isfinite(v)) for k, v in self.wrf.items()
                   if k != "qmixtnp")


#: reference name -> file.  "stock" is bl_ysu_run built with WRF's own
#: configure.defaults flags (-O2 -ftree-vectorize -funroll-loops), the
#: arithmetic a WRF run performs; "O0" is the same source at -O0.
REFERENCES = {"stock": "ysu-wide-wrf.bin", "O0": "ysu-wide-wrf-O0.bin"}


def load_wide_oracle(directory: Path | None = None, *,
                     reference: str = "stock") -> list[WideColumn]:
    directory = Path(directory) if directory is not None else YSU_WIDE_DIR
    raw = (directory / "ysu-wide-inputs.bin").read_bytes()
    out_raw = (directory / REFERENCES[reference]).read_bytes()
    cols: list[WideColumn] = []
    pos = opos = 0
    while pos < len(raw):
        nz, topdown, regime, topo = np.frombuffer(raw, "<i4", 4, pos)
        nz = int(nz)
        pos += 16
        sc = np.frombuffer(raw, "<f4", len(_SCALARS), pos)
        pos += 4 * len(_SCALARS)
        levels = {}
        for name in _LEVELS:
            levels[name] = np.frombuffer(raw, "<f4", nz, pos).astype(np.float32)
            pos += 4 * nz
        pin = np.frombuffer(raw, "<f4", nz + 1, pos).astype(np.float32)
        pos += 4 * (nz + 1)
        col = WideColumn(len(cols), nz, int(topdown), int(regime), int(topo),
                         {k: np.float32(v) for k, v in zip(_SCALARS, sc)},
                         levels, pin)
        col.wrf_kpbl = int(np.frombuffer(out_raw, "<i4", 1, opos)[0])
        opos += 4
        s = np.frombuffer(out_raw, "<f4", 5, opos).astype(np.float32)
        opos += 20
        col.wrf.update(hpbl=s[0:1], wstar=s[1:2], delta=s[2:3], u10=s[3:4],
                       v10=s[4:5])
        for name in _WRF_LEVELS:
            col.wrf[name] = np.frombuffer(out_raw, "<f4", nz,
                                          opos).astype(np.float32)
            opos += 4 * nz
        cols.append(col)
    if opos != len(out_raw):
        raise ValueError(f"WRF output has {len(out_raw) - opos} trailing bytes;"
                         " inputs and outputs are not from the same run")
    return cols


def _group_key(c: WideColumn):
    return (c.nz, float(c.scalars["dt"]), c.topdown, c.topo)


def run_port_wide(cols: list[WideColumn], *, ysu_version="wrf_480") -> dict[int, dict[str, np.ndarray]]:
    """Launch the port on every column; return host outputs by column index."""
    import cupy as cp

    from gpuwm.core.ysu import launch_ysu

    groups: dict[tuple, list[WideColumn]] = {}
    for c in cols:
        groups.setdefault(_group_key(c), []).append(c)
    result: dict[int, dict[str, np.ndarray]] = {}
    for (nz, dt, topdown, topo), members in sorted(groups.items()):
        n = len(members)

        def lev(name):
            a = np.stack([m.levels[name] for m in members], axis=1)
            return cp.asarray(np.ascontiguousarray(a.reshape(nz, 1, n)))

        def sfc(name):
            a = np.asarray([m.scalars[name] for m in members], np.float32)
            return cp.asarray(np.ascontiguousarray(a.reshape(1, n)))

        pin = np.stack([m.p_interface for m in members], axis=1)
        kwargs = {}
        if topo == 1:
            u10o = cp.zeros((1, n), dtype=cp.float32)
            v10o = cp.zeros((1, n), dtype=cp.float32)
            kwargs = dict(topo=(sfc("ctopo"), sfc("ctopo2")),
                          u10_out=u10o, v10_out=v10o)
        elif topo == 2:
            from gpuwm.core.ysu import YSU_BEP_TERMS
            bep = {name: cp.zeros((nz, 1, n), dtype=cp.float32) for name in YSU_BEP_TERMS}
            values = {"a_u_bep": -.002, "a_v_bep": -.002, "a_t_bep": -.00001,
                      "a_q_bep": -.0001, "b_u_bep": .0002, "b_v_bep": .0002,
                      "b_t_bep": .003, "b_q_bep": .00000002}
            for name, value in values.items():
                bep[name][:min(4, nz)] = cp.float32(value)
            bep["sf_bep"].fill(1)
            bep["vl_bep"].fill(1)
            kwargs = dict(bep=bep, frc_urb2d=sfc("_pad"))
        out = launch_ysu(
            lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"),
            lev("qi"), lev("p"),
            cp.asarray(np.ascontiguousarray(pin.reshape(nz + 1, 1, n))),
            lev("pi"), lev("dz"), lev("rthraten"),
            psfc=sfc("psfc"), znt=sfc("znt"), ust=sfc("ust"),
            hfx=sfc("hfx"), qfx=sfc("qfx"), wspd=sfc("wspd"), br=sfc("br"),
            psim=sfc("psim"), psih=sfc("psih"), xland=sfc("xland"),
            u10=sfc("u10"), v10=sfc("v10"), dt=dt,
            ysu_topdown_pblmix=topdown, ysu_version=ysu_version, **kwargs)
        host = {k: cp.asnumpy(v) for k, v in out.items()}
        if topo == 1:
            host["u10"] = cp.asnumpy(u10o)
            host["v10"] = cp.asnumpy(v10o)
        for j, m in enumerate(members):
            per = {}
            for k, v in host.items():
                per[k] = (v[:, 0, j] if v.ndim == 3 else v[0, j:j + 1])
            result[m.index] = per
    return result


@dataclass
class FieldStat:
    max_ulp: int = 0
    max_abs: float = 0.0
    differing: int = 0
    total: int = 0
    worst: tuple | None = None   # (column, level, got, want)


def _bits_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, np.float32).view(np.uint32)
    b = np.ascontiguousarray(b, np.float32).view(np.uint32)
    return a == b


def compare_wide(cols: list[WideColumn], port: dict[int, dict[str, np.ndarray]],
                 *, include=None) -> dict:
    """Per-field statistics over the selected columns, plus kpbl.

    ``include(col) -> bool`` selects columns (default: every column where
    WRF's output is finite).  ULP distance is bit-pattern distance with
    -0 == +0 counted as 0 ULP but still counted as ``differing`` (a sign of
    zero is a different word).
    """
    if include is None:
        include = lambda c: c.wrf_finite  # noqa: E731
    stats: dict[str, FieldStat] = {}
    kpbl_mismatch = []
    per_column: dict[int, dict[str, int]] = {}
    for c in cols:
        if not include(c):
            continue
        got = port[c.index]
        fields = dict(LEVEL_FIELDS)
        fields.update(SURFACE_FIELDS)
        if c.topo == 1:
            fields.update(TOPO_FIELDS)
        colmax = {}
        for name, wname in fields.items():
            g = np.ascontiguousarray(got[name], np.float32).reshape(-1)
            w = np.ascontiguousarray(c.wrf[wname], np.float32).reshape(-1)
            ulp = fp32_ulp_distance(g, w).astype(np.int64)
            diff = ~_bits_equal(g, w)
            with np.errstate(invalid="ignore"):
                absd = np.abs(g.astype(np.float64) - w.astype(np.float64))
            absd = np.where(np.isfinite(absd), absd, np.inf)
            st = stats.setdefault(name, FieldStat())
            st.total += g.size
            st.differing += int(diff.sum())
            if ulp.size and int(ulp.max()) > st.max_ulp:
                k = int(np.argmax(ulp))
                st.max_ulp = int(ulp.max())
                st.worst = (c.index, k + 1 if g.size > 1 else None,
                            float(g[k]), float(w[k]))
            if absd.size:
                st.max_abs = max(st.max_abs, float(np.max(absd)))
            colmax[name] = int(ulp.max()) if ulp.size else 0
        per_column[c.index] = colmax
        if int(got["kpbl"].reshape(-1)[0]) != c.wrf_kpbl:
            kpbl_mismatch.append((c.index, int(got["kpbl"].reshape(-1)[0]),
                                  c.wrf_kpbl))
    return {"fields": stats, "kpbl_mismatch": kpbl_mismatch,
            "per_column": per_column}
