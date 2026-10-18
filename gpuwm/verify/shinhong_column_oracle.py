"""The wide Shin-Hong column oracle: read it, run WOOF on it, measure every word.

``tools/shinhong_column_oracle/make_columns.py`` writes the columns and
``run_shinhong_columns.F90`` drives WRF v4.6.1's byte-unmodified
``shinhong`` entry on each of them with the argument set
``module_pbl_driver.F`` passes in a WRF run (t3d = th*pi as phy_prep forms
it, psfc = p8w(1), tendencies zeroed, ctopo = ctopo2 = 1).  This module reads
both files, launches ``gpuwm.core.shinhong.launch_shinhong`` on the same
columns (theta and exner, as the engine hands them), and returns the
per-field, per-column comparison.  ``run_authority`` does the same for the
float32 NumPy authority ``gpuwm.verify.shinhong_ref.np_shinhong_column``.

Columns are grouped by everything ``launch_shinhong`` takes launch-wide
(nz, dt, dx, dy, tke_diag) and each group is one launch.

Two lane classes are reported apart from the arithmetic, never hidden:

* NaN lanes.  WRF's own prfac2 0/0 (module_bl_shinhong.F:1010) writes NaN
  on the two probe columns built for it (unstable ``br`` with
  ``ust**3 == wstar**3 == 0``, an input WRF's surface layers never produce).
  The comparison counts those lanes, requires the NaN GEOMETRY to agree, and
  reports whether the payload bits agree; x86 SSE's default NaN is
  0xFFC00000 and CUDA's is 0x7FFFFFFF, so the payload is a property of the
  hardware that produced the NaN, not of the scheme.
* ``kpbl == nz`` with ``tke_diag == 1``.  WRF reads ``q2xk(kte+1)``, one
  past the end (:1557), and its ``tke``/``el`` there carry whatever the stack
  held.  Those two fields of those columns are excluded and counted.

Nothing here knows a tolerance.  The gate
(tests/test_shinhong_column_oracle.py) and the CLI
(tools/shinhong_column_oracle/compare.py) decide what to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "SHINHONG_COLUMNS_DIR", "Column", "load_columns", "run_port",
    "run_authority", "compare", "LEVEL_FIELDS", "SURFACE_FIELDS", "FAMILIES",
]

SHINHONG_COLUMNS_DIR = (Path(__file__).resolve().parents[2] / "tests"
                        / "data" / "oracles" / "shinhong_columns")

FAMILIES = {
    1: "convective-land-dry", 2: "convective-land-moist-cumulus",
    3: "convective-desert", 4: "morning-transition",
    5: "stable-night-land-weak", 6: "stable-night-land-radiative",
    7: "snow-arctic-mixed-phase", 8: "snow-ice-fog", 9: "snow-spring-daytime",
    10: "ocean-tropical-trades", 11: "ocean-stratocumulus",
    12: "ocean-cold-air-outbreak", 13: "ocean-warm-advection-fog",
    14: "high-terrain-day", 15: "high-terrain-night", 16: "high-terrain-snow",
    17: "windy-neutral-overcast", 18: "edge-probe",
}

_SCALARS = ("dt", "dx", "dy", "psfc", "znt", "ust", "hfx", "qfx", "wspd",
            "br", "psim", "psih", "xland", "u10", "v10", "corf")
_LEVELS = ("u", "v", "theta", "pi", "t", "qv", "qc", "qi", "p", "dz", "tke")
_WRF_LEVELS = ("rublten", "rvblten", "rthblten", "rqvblten", "rqcblten",
               "rqiblten", "exch_h", "tke", "el")
_WRF_SURFACE = ("hpbl", "wstar", "delta", "u10", "v10", "ust", "znt",
                "wspd", "regime")

#: WOOF output -> WRF word, per level.  dtheta is the wrapper's
#: rthblten = ttnp/pi (:253).
LEVEL_FIELDS = {
    "du": "rublten", "dv": "rvblten", "dtheta": "rthblten",
    "dqv": "rqvblten", "dqc": "rqcblten", "dqi": "rqiblten",
    "exch_h": "exch_h", "tke": "tke", "el": "el",
}
SURFACE_FIELDS = {"hpbl": "hpbl", "wstar": "wstar", "delta": "delta"}
#: The TKE-chain fields WRF leaves undefined when kpbl == kte (:1557).
OOB_FIELDS = ("tke", "el")


@dataclass
class Column:
    index: int
    nz: int
    tke_diag: int
    family: int
    base: int
    scalars: dict[str, np.float32]
    levels: dict[str, np.ndarray]
    p_interface: np.ndarray
    wrf: dict[str, np.ndarray] = field(default_factory=dict)
    wrf_kpbl: int = 0

    @property
    def family_name(self) -> str:
        return FAMILIES[self.family]

    @property
    def oob_tke(self) -> bool:
        """WRF read q2xk(kte+1) here; its tke/el are undefined."""
        return self.tke_diag == 1 and self.wrf_kpbl == self.nz


def load_columns(directory: Path | None = None, *,
                 inputs: str = "columns-in.bin",
                 outputs: str = "wrf-out.bin") -> list[Column]:
    directory = Path(directory) if directory is not None else SHINHONG_COLUMNS_DIR
    raw = _read(directory / inputs)
    out_raw = _read(directory / outputs)
    cols: list[Column] = []
    pos = opos = 0
    while pos < len(raw):
        nz, tke_diag, family, base = (int(x) for x in
                                      np.frombuffer(raw, "<i4", 4, pos))
        pos += 16
        sc = np.frombuffer(raw, "<f4", len(_SCALARS), pos)
        pos += 4 * len(_SCALARS)
        levels = {}
        for name in _LEVELS:
            levels[name] = np.frombuffer(raw, "<f4", nz, pos).astype(np.float32)
            pos += 4 * nz
        pin = np.frombuffer(raw, "<f4", nz + 1, pos).astype(np.float32)
        pos += 4 * (nz + 1)
        col = Column(len(cols), nz, tke_diag, family, base,
                     {k: np.float32(v) for k, v in zip(_SCALARS, sc)},
                     levels, pin)
        onz, kpbl, ofam, obase = (int(x) for x in
                                  np.frombuffer(out_raw, "<i4", 4, opos))
        opos += 16
        if (onz, ofam, obase) != (nz, family, base):
            raise ValueError(f"column {col.index}: WRF output record "
                             f"{(onz, ofam, obase)} does not match the input "
                             f"{(nz, family, base)}")
        col.wrf_kpbl = kpbl
        for name in _WRF_LEVELS:
            col.wrf[name] = np.frombuffer(out_raw, "<f4", nz,
                                          opos).astype(np.float32)
            opos += 4 * nz
        s = np.frombuffer(out_raw, "<f4", len(_WRF_SURFACE),
                          opos).astype(np.float32)
        opos += 4 * len(_WRF_SURFACE)
        for j, name in enumerate(_WRF_SURFACE):
            col.wrf[name] = s[j:j + 1]
        cols.append(col)
    if opos != len(out_raw):
        raise ValueError(f"WRF output has {len(out_raw) - opos} trailing bytes;"
                         " inputs and outputs are not from the same run")
    return cols


def _read(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    gz = path.with_name(path.name + ".gz")
    import gzip

    return gzip.decompress(gz.read_bytes())


def _group_key(c: Column):
    s = c.scalars
    return (c.nz, float(s["dt"]), float(s["dx"]), float(s["dy"]), c.tke_diag)


def run_port(cols: list[Column]) -> dict[int, dict[str, np.ndarray]]:
    """Launch the WOOF kernel on every column; host outputs by column index."""
    import cupy as cp

    from gpuwm.core.shinhong import launch_shinhong

    groups: dict[tuple, list[Column]] = {}
    for c in cols:
        groups.setdefault(_group_key(c), []).append(c)
    result: dict[int, dict[str, np.ndarray]] = {}
    for (nz, dt, dx, dy, tke_diag), members in sorted(groups.items()):
        n = len(members)

        def lev(name):
            a = np.stack([m.levels[name] for m in members], axis=1)
            return cp.asarray(np.ascontiguousarray(a.reshape(nz, 1, n)))

        def sfc(name):
            a = np.asarray([m.scalars[name] for m in members], np.float32)
            return cp.asarray(np.ascontiguousarray(a.reshape(1, n)))

        pin = np.stack([m.p_interface for m in members], axis=1)
        out = launch_shinhong(
            lev("u"), lev("v"), lev("theta"), lev("qv"), lev("qc"),
            lev("qi"), lev("p"),
            cp.asarray(np.ascontiguousarray(pin.reshape(nz + 1, 1, n))),
            lev("pi"), lev("dz"), lev("tke"),
            psfc=sfc("psfc"), znt=sfc("znt"), ust=sfc("ust"),
            hfx=sfc("hfx"), qfx=sfc("qfx"), wspd=sfc("wspd"), br=sfc("br"),
            psim=sfc("psim"), psih=sfc("psih"), xland=sfc("xland"),
            u10=sfc("u10"), v10=sfc("v10"), corf=sfc("corf"),
            dt=dt, dx=dx, dy=dy, tke_diag=tke_diag)
        host = {k: cp.asnumpy(v) for k, v in out.items()}
        for j, m in enumerate(members):
            result[m.index] = {k: (v[:, 0, j] if v.ndim == 3 else v[0, j:j + 1])
                               for k, v in host.items()}
    return result


def run_authority(cols: list[Column]) -> dict[int, dict[str, np.ndarray]]:
    """The float32 NumPy authority on every column, in the kernel's names.

    The authority takes temperature as WRF does, so it gets the record's
    t (= theta*pi, the phy_prep product) exactly as the Fortran does.
    """
    from gpuwm.verify.shinhong_ref import np_shinhong_column

    result = {}
    for c in cols:
        L, s = c.levels, c.scalars
        o = np_shinhong_column(
            L["u"], L["v"], L["t"], L["qv"], L["qc"], L["qi"], L["p"],
            L["pi"], c.p_interface, L["dz"], L["tke"],
            psfc=s["psfc"], znt=s["znt"], ust=s["ust"], hfx=s["hfx"],
            qfx=s["qfx"], wspd=s["wspd"], br=s["br"], psim=s["psim"],
            psih=s["psih"], xland=s["xland"], corf=s["corf"], u10=s["u10"],
            v10=s["v10"], dt=s["dt"], dx=s["dx"], dy=s["dy"],
            tke_diag=c.tke_diag)
        per = {}
        for port_name, wrf_name in LEVEL_FIELDS.items():
            per[port_name] = np.asarray(o[wrf_name], np.float32)
        for name in SURFACE_FIELDS:
            per[name] = np.asarray([o[name]], np.float32).reshape(1)
        per["kpbl"] = np.asarray([o["kpbl"]], np.int32).reshape(1)
        result[c.index] = per
    return result


@dataclass
class FieldStat:
    max_ulp: int = 0
    max_abs: float = 0.0
    differing: int = 0
    total: int = 0
    nan_lanes: int = 0
    nan_geometry_mismatch: int = 0
    nan_payload_mismatch: int = 0
    #: lanes where the reference wrote a subnormal and the port wrote a zero
    #: (a flush-to-zero build), and the worst ULP over every other
    #: arithmetic lane
    ftz_lanes: int = 0
    max_ulp_outside_ftz: int = 0
    worst: tuple | None = None   # (column, level, got, want)

    def as_dict(self) -> dict:
        return dict(max_ulp=self.max_ulp, max_abs=self.max_abs,
                    differing=self.differing, total=self.total,
                    nan_lanes=self.nan_lanes,
                    nan_geometry_mismatch=self.nan_geometry_mismatch,
                    nan_payload_mismatch=self.nan_payload_mismatch,
                    ftz_lanes=self.ftz_lanes,
                    max_ulp_outside_ftz=self.max_ulp_outside_ftz,
                    worst=self.worst)


def _bits(a) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32).reshape(-1).view(np.uint32)


def compare(cols: list[Column], port: dict[int, dict[str, np.ndarray]],
            *, include=None) -> dict:
    """Per-field statistics over the selected columns, plus kpbl.

    Arithmetic lanes (neither side NaN) give ``max_ulp``/``max_abs``;
    ``differing`` counts every lane whose 32-bit word differs, including a
    signed-zero flip (0 ULP but a different word) and NaN payloads.  NaN
    lanes are counted separately with their geometry and payload agreement.
    """
    if include is None:
        include = lambda c: True  # noqa: E731
    stats: dict[str, FieldStat] = {}
    kpbl_mismatch = []
    per_column: dict[int, dict[str, int]] = {}
    oob_excluded = 0
    for c in cols:
        if not include(c):
            continue
        got = port[c.index]
        fields = dict(LEVEL_FIELDS)
        fields.update(SURFACE_FIELDS)
        colmax = {}
        for name, wname in fields.items():
            if c.oob_tke and name in OOB_FIELDS:
                oob_excluded += 1
                continue
            g = np.ascontiguousarray(got[name], np.float32).reshape(-1)
            w = np.ascontiguousarray(c.wrf[wname], np.float32).reshape(-1)
            gb, wb = _bits(g), _bits(w)
            gnan, wnan = np.isnan(g), np.isnan(w)
            nan = gnan | wnan
            st = stats.setdefault(name, FieldStat())
            st.total += g.size
            st.differing += int((gb != wb).sum())
            st.nan_lanes += int(nan.sum())
            st.nan_geometry_mismatch += int((gnan != wnan).sum())
            st.nan_payload_mismatch += int((gnan & wnan & (gb != wb)).sum())
            ulp = np.where(nan, np.int64(0), fp32_ulp_distance(g, w))
            sub = (wb & np.uint32(0x7F800000)) == 0
            ftz = sub & ((wb & np.uint32(0x007FFFFF)) != 0) & (g == 0.0)
            st.ftz_lanes += int(ftz.sum())
            outside = np.where(ftz, np.int64(0), ulp)
            if outside.size:
                st.max_ulp_outside_ftz = max(st.max_ulp_outside_ftz,
                                             int(outside.max()))
            with np.errstate(invalid="ignore", over="ignore"):
                absd = np.abs(g.astype(np.float64) - w.astype(np.float64))
            absd = np.where(nan, 0.0, absd)
            if ulp.size and int(ulp.max()) > st.max_ulp:
                k = int(np.argmax(ulp))
                st.max_ulp = int(ulp.max())
                st.worst = (c.index, k + 1 if g.size > 1 else None,
                            float(g[k]), float(w[k]))
            if absd.size:
                st.max_abs = max(st.max_abs, float(np.max(absd)))
            colmax[name] = int(ulp.max()) if ulp.size else 0
        per_column[c.index] = colmax
        gk = int(np.asarray(got["kpbl"]).reshape(-1)[0])
        if gk != c.wrf_kpbl:
            kpbl_mismatch.append((c.index, gk, c.wrf_kpbl))
    return {"fields": stats, "kpbl_mismatch": kpbl_mismatch,
            "per_column": per_column, "oob_excluded": oob_excluded}


def wrf_passthrough_violations(cols: list[Column]) -> list[tuple]:
    """INOUT surface fields WRF must hand back word for word on arm A.

    ust, znt and wspd are INOUT and never assigned; u10/v10 go through
    ctopo2*u10 + (1-ctopo2)*u1 (:1472) which with ctopo2 = 1 is the input
    word; regime is never written.  The WOOF kernel has none of them as
    outputs, so this is what licenses leaving them out.
    """
    bad = []
    for c in cols:
        for name in ("ust", "znt", "wspd", "u10", "v10"):
            if _bits(c.wrf[name])[0] != _bits(np.float32(c.scalars[name]))[0]:
                bad.append((c.index, name, float(c.wrf[name][0]),
                            float(c.scalars[name])))
        if _bits(c.wrf["regime"])[0] != 0:
            bad.append((c.index, "regime", float(c.wrf["regime"][0]), 0.0))
    return bad
