"""Shared driver for the BEP+BEM regime column oracle (WRF v4.6.1).

``tools/urban_bem_regime_oracle/run_bep_bem_regime.F90`` drives WRF's own
``BEP_BEM`` over weather regimes x urban classes (plus edge columns) and dumps
every input, every output of every call, the state before the first call and
the state after the last.  This module drives the engine's GPU column
(:func:`gpuwm.core.urban_bem.launch_bep_bem_columns`) through the same calls,
carrying the engine's OWN state from WRF's initial state, and measures every
output word against WRF's, per field and per column.

Used by ``tests/test_urban_bem_regime_wrf461.py`` and by
``tools/urban_bem_regime_oracle/compare.py``.  Nothing here decides a
tolerance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIXTURE_DIR = (Path(__file__).resolve().parent / "data" / "oracles" / "urban"
               / "bem_regime461")
VARIANTS = ("nlcd-summer", "nlcd-winter", "lcz-summer", "lcz-winter",
            "gr1pv-summer", "gr2-winter", "long-summer", "stop-summer")

LEVEL_OUTPUTS = ("a_u", "a_v", "a_t", "a_e", "b_u", "b_v", "b_t", "b_e",
                 "b_q", "dlg", "dl_u", "vl")
SURFACE_OUTPUTS = ("rl_up", "rs_abs", "emiss", "grdflx_urb")
BEM_2D_OUTPUTS = ("sf_ac_urb3d", "lf_ac_urb3d", "cm_ac_urb3d",
                  "sfvent_urb3d", "lfvent_urb3d", "ep_pv_urb3d",
                  "qgr_urb3d", "tgr_urb3d", "draingr_urb3d")
NDM_OUTPUTS = ("sfg_urb3d", "dg_urb3d", "lfg_urb3d")
STATE_ARRAYS = (
    "trb_urb4d", "tw1_urb4d", "tw2_urb4d", "tgb_urb4d",
    "tlev_urb3d", "qlev_urb3d", "tw1lev_urb3d", "tw2lev_urb3d",
    "tglev_urb3d", "tflev_urb3d", "sf_ac_urb3d", "lf_ac_urb3d",
    "cm_ac_urb3d", "sfvent_urb3d", "lfvent_urb3d",
    "sfwin1_urb3d", "sfwin2_urb3d",
    "sfw1_urb3d", "sfw2_urb3d", "sfr_urb3d", "sfg_urb3d",
    "ep_pv_urb3d", "t_pv_urb3d",
    "trv_urb4d", "qr_urb4d", "qgr_urb3d", "tgr_urb3d",
    "drain_urb4d", "draingr_urb3d", "sfrv_urb3d",
    "lfrv_urb3d", "dgr_urb3d", "dg_urb3d", "lfr_urb3d", "lfg_urb3d",
)
MORPHOLOGY = ("frc_urb2d", "utype_urb2d", "lp_urb2d", "lb_urb2d",
              "hgt_urb2d", "hi_urb2d")
ATMOSPHERE_3D = ("dz8w", "u_phy", "v_phy", "th_phy", "rho", "p_phy",
                 "qv_phy")
ATMOSPHERE_2D = ("swdown", "glw", "swddir", "swddif", "rainbl",
                 "cosz_urb2d", "omg_urb2d", "xlat", "xlong")
EDGES = {1: "rural land (FRC 0)", 2: "open water (FRC 0)", 3: "FRC 0.01",
         4: "FRC 1.0", 5: "gridded morphology", 6: "gridded tall towers",
         7: "first level 40 m", 8: "first level 2 m"}


def load_fixture(variant: str, root: Path | None = None) -> dict[str, np.ndarray]:
    """``{dump name: array}``; arrays keep the Fortran index order."""
    root = FIXTURE_DIR if root is None else Path(root)
    with np.load(root / f"bep_bem_{variant}.npz") as npz:
        return {k.replace("__", "/"): npz[k] for k in npz.files}


def to_gpuwm(a: np.ndarray) -> np.ndarray:
    """WRF ``(ims:ime, [k,] jms:jme)`` with one j row -> gpuwm (k, ny, nx)."""
    if a.ndim == 2:
        return np.ascontiguousarray(a.T)
    if a.ndim == 3:
        return np.ascontiguousarray(a.transpose(1, 2, 0))
    raise ValueError(a.shape)


def table_inputs(fx: dict[str, np.ndarray]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in fx.items():
        if not key.startswith("tbl/"):
            continue
        name = key[4:]
        out[name] = (value.reshape(()).item() if value.size == 1
                     and name in ("icate", "gr_flag_tbl", "gr_type_tbl")
                     else value)
    return out


def wrf_stop_call(fx: dict[str, np.ndarray]) -> int | None:
    """The call in which WRF's BEP_BEM hit a Fortran ``stop``, if any.

    The driver dumps a call's inputs before the call and its outputs after
    it, so a call with inputs and no outputs is the one WRF stopped in.
    """
    nsteps = int(fx["nsteps"][0])
    for step in range(1, nsteps + 1):
        if f"step{step}/dz8w" in fx and f"step{step}/out/a_u" not in fx:
            return step
    return None


def run_port(fx: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
    """Drive the engine's column through every call; arrays after each call.

    Where WRF stopped (:func:`wrf_stop_call`) the engine must stop in the
    same call: the returned dict then holds ``"stop"`` = ``(call, message)``
    of the engine's :class:`~gpuwm.core.urban_bem.BepBemError`, or
    ``(call, None)`` if the engine ran on.
    """
    import cupy as cp

    from gpuwm.core import urban_bem as ub

    nz = int(fx["nz"][0])
    nsteps = int(fx["nsteps"][0])
    num_urban_hi = int(fx["num_urban_hi"][0])
    tblf, tbli = ub.pack_bem_tables(table_inputs(fx))
    cls = ub.build_class_tables(tblf, tbli)

    dev: dict[str, object] = {}
    for name in STATE_ARRAYS + MORPHOLOGY:
        dev[name] = cp.asarray(to_gpuwm(fx[f"state0/{name}"]))
    for name in LEVEL_OUTPUTS:
        dev[name] = cp.asarray(to_gpuwm(fx[f"state0/{name}_bep"])[:nz])
    dev["sf"] = cp.asarray(to_gpuwm(fx["state0/sf_bep"])[:nz + 1])
    for name in SURFACE_OUTPUTS:
        dev[name] = cp.zeros_like(dev["frc_urb2d"])

    steps: dict = {}
    plan = None
    stop = wrf_stop_call(fx)
    for step in range(1, (stop or nsteps) + 1):
        tag = f"step{step}"
        for name in ATMOSPHERE_3D:
            dev[name] = cp.asarray(to_gpuwm(fx[f"{tag}/{name}"])[:nz])
        for name in ATMOSPHERE_2D:
            dev[name] = cp.asarray(to_gpuwm(fx[f"{tag}/{name}"]))
        # module_sf_noahdrv.F: zeroed on every column before every call.
        for name in SURFACE_OUTPUTS:
            dev[name][...] = 0.0
        dev["b_q"][...] = 0.0
        kwargs = dict(
            gmt=float(fx[f"{tag}/gmt"][0]),
            julday=int(fx[f"{tag}/julday"][0]),
            declin_urb=float(fx[f"{tag}/declin_urb"][0]),
            dt=float(fx[f"{tag}/dt"][0]),
            itimestep=int(fx[f"{tag}/itimestep"][0]),
            num_urban_hi=num_urban_hi, plan=plan)
        if step == stop:
            try:
                ub.launch_bep_bem_columns(dev, cls, **kwargs)
            except ub.BepBemError as exc:
                steps["stop"] = (step, str(exc))
            else:
                steps["stop"] = (step, None)
            break
        plan = ub.launch_bep_bem_columns(dev, cls, **kwargs)
        steps[step] = {k: cp.asnumpy(v) for k, v in dev.items()
                       if hasattr(v, "shape")}
    return steps


def _ulp(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """ULP distance of float32 words; NaN == NaN counts as equal."""
    # The project's one monotone float32 line (the same INT32_MIN
    # reflection this helper used to spell out itself).
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    d = np.abs(monotone_fp32_key(got) - monotone_fp32_key(want))
    both_nan = np.isnan(got) & np.isnan(want)
    return np.where(both_nan, 0, d)


def measure(fx: dict[str, np.ndarray], steps=None) -> dict:
    """Every output of every call and the final state, word for word.

    Returns ``{"fields": {name: {"max_ulp", "max_abs", "words",
    "words_differ"}}, "column_bad": bool array (ncol,), "words": int,
    "words_differ": int}``.
    """
    nz = int(fx["nz"][0])
    if steps is None:
        steps = run_port(fx)
    stop = wrf_stop_call(fx)
    nsteps = int(fx["nsteps"][0]) if stop is None else stop - 1
    ncol = int(fx["ncol"][0])
    fields: dict[str, dict] = {}
    column_bad = np.zeros(ncol, bool)

    def note(name: str, got: np.ndarray, want: np.ndarray) -> None:
        assert got.shape == want.shape, (name, got.shape, want.shape)
        if want.dtype.kind == "i":
            d = np.abs(got.astype(np.int64) - want.astype(np.int64))
            ad = d.astype(np.float64)
        else:
            d = _ulp(got, want)
            ad = np.abs(got.astype(np.float64) - want.astype(np.float64))
            ad = np.where(np.isnan(ad), 0.0, ad)
        bad = d != 0
        row = fields.setdefault(name, {"max_ulp": 0, "max_abs": 0.0,
                                       "words": 0, "words_differ": 0})
        row["max_ulp"] = max(row["max_ulp"], int(d.max()) if d.size else 0)
        row["max_abs"] = max(row["max_abs"], float(ad.max()) if ad.size else 0.0)
        row["words"] += int(d.size)
        row["words_differ"] += int(bad.sum())
        column_bad[:] |= bad.reshape(-1, ncol).any(axis=0)

    for step in range(1, nsteps + 1):
        got = steps[step]
        tag = f"step{step}/out"
        for name in LEVEL_OUTPUTS:
            note(name, got[name], to_gpuwm(fx[f"{tag}/{name}"])[:nz])
        note("sf", got["sf"], to_gpuwm(fx[f"{tag}/sf"])[:nz + 1])
        for name in SURFACE_OUTPUTS + BEM_2D_OUTPUTS + NDM_OUTPUTS:
            note(name, got[name], to_gpuwm(fx[f"{tag}/{name}"]))
    if stop is None:
        final = steps[nsteps]
        for name in STATE_ARRAYS:
            note(f"final/{name}", final[name], to_gpuwm(fx[f"final/{name}"]))
    return {"fields": fields, "column_bad": column_bad,
            "wrf_stop_call": stop, "engine_stop": steps.get("stop"),
            "words": sum(r["words"] for r in fields.values()),
            "words_differ": sum(r["words_differ"] for r in fields.values())}


def column_labels(fx: dict[str, np.ndarray], regimes: dict[int, str]) -> list[str]:
    """A human label per column: regime x class, or the edge case."""
    reg = fx["col/regime"]
    cls = fx["col/class"]
    edge = fx["col/edge"]
    out = []
    for r, c, e in zip(reg.tolist(), cls.tolist(), edge.tolist()):
        if e:
            out.append(f"edge: {EDGES[e]} ({regimes.get(r, r)})")
        else:
            out.append(f"{regimes.get(r, r)} / class {c}")
    return out


def read_regimes(path: Path, winter: bool) -> dict[int, str]:
    import csv

    want = "W" if winter else "S"
    with open(path, newline="") as handle:
        return {int(row["id"]): row["regime"]
                for row in csv.DictReader(handle) if row["season"] == want}
