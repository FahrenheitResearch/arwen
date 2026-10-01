"""Read the WRF v4.7.1 UCM column oracle and measure the port against it.

The fixture is written by ``tools/urban_wrf471_oracle/ucm_column_oracle.F90`` (driven
by ``build_ucm.sh``), which calls ``urban`` in the byte-unmodified
``phys/module_sf_urban.F`` after WRF's own ``urban_param_init`` has read
``URBPARM.TBL`` or ``URBPARM_LCZ.TBL``.  Per variant there are three files in
``gpuwm/data/urban/oracle/ucm/``:

* ``ucm-<table>-<variant>.csv.gz`` -- one row per column step: the inputs,
  the state before (``*_in``), the outputs, the state after;
* ``...-table.csv`` -- what ``read_param`` hands ``urban`` for every UTYPE;
* ``...-switches.csv`` -- the module switches and arrays ``urban`` reads.

Values are 9-significant-digit decimals, which round-trip float32 exactly.

Nothing here knows a tolerance; ``tests/test_urban_ucm_wrf471_parity.py``
decides.
"""
from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

__all__ = [
    "UCM_ORACLE_DIR",
    "UCM_VARIANTS",
    "STATE_COLUMNS",
    "OUTPUT_COLUMNS",
    "load_variant",
    "load_noah",
    "noah_replay",
    "load_noahmp",
    "noahmp_replay",
    "port_params",
    "replay",
    "carried_replay",
    "ulp_table",
]

UCM_ORACLE_DIR = (Path(__file__).resolve().parents[1]
                  / "data" / "urban" / "oracle" / "ucm")

#: Every variant ``build_ucm.sh`` writes.  The ``gr`` ones are the
#: ``-finit-real=zero`` build (the ETR defined read, see its header).
UCM_VARIANTS = (
    "ucm-nlcd-default", "ucm-lcz-default", "ucm-nlcd-ch1", "ucm-nlcd-ts2",
    "ucm-nlcd-ch1ts2", "ucm-nlcd-ahalh", "ucm-nlcd-imp2", "ucm-nlcd-bound2",
    "ucm-lcz-imp2ahalh", "ucm-nlcd-gr", "ucm-nlcd-griri", "ucm-lcz-gr",
)

#: fixture state column (without ``_in``) -> (kernel plane, layer or None)
STATE_COLUMNS: dict[str, tuple[str, int | None]] = {}
for _base, _plane in (("tr", "tr_urb2d"), ("tb", "tb_urb2d"),
                      ("tg", "tg_urb2d"), ("tc", "tc_urb2d"),
                      ("qc", "qc_urb2d"), ("uc", "uc_urb2d"),
                      ("xxxr", "xxxr_urb2d"), ("xxxb", "xxxb_urb2d"),
                      ("xxxg", "xxxg_urb2d"), ("xxxc", "xxxc_urb2d"),
                      ("cmr", "cmr_sfcdif"), ("chr", "chr_sfcdif"),
                      ("cmc", "cmc_sfcdif"), ("chc", "chc_sfcdif"),
                      ("cmgr", "cmgr_sfcdif"), ("chgr", "chgr_sfcdif"),
                      ("cmcr", "cmcr_urb2d"), ("tgr", "tgr_urb2d"),
                      ("drelr", "drelr_urb2d"), ("drelb", "drelb_urb2d"),
                      ("drelg", "drelg_urb2d"), ("flxhumr", "flxhumr_urb2d"),
                      ("flxhumb", "flxhumb_urb2d"),
                      ("flxhumg", "flxhumg_urb2d")):
    STATE_COLUMNS[_base] = (_plane, None)
for _base, _plane in (("trl", "trl_urb3d"), ("tbl", "tbl_urb3d"),
                      ("tgl", "tgl_urb3d"), ("tgrl", "tgrl_urb3d"),
                      ("smr", "smr_urb3d")):
    for _k in range(4):
        STATE_COLUMNS[f"{_base}{_k + 1}"] = (_plane, _k)

OUTPUT_COLUMNS = ("ts", "qs", "sh", "lh", "lh_kinematic", "sw", "alb", "lw",
                  "g", "rn", "psim", "psih", "gz1oz0", "u10", "v10", "th2",
                  "q2", "ust", "znt")

_INT_COLUMNS = ("utype", "scenario", "step", "jmonth")


def _f32(text: str) -> np.float32:
    t = text.strip()
    if t.lower() in ("nan", "-nan", "+nan"):
        return np.float32(np.nan)
    if t.lower() in ("infinity", "+infinity", "inf"):
        return np.float32(np.inf)
    if t.lower() in ("-infinity", "-inf"):
        return np.float32(-np.inf)
    return np.float32(float(t))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        text = gzip.decompress(path.read_bytes()).decode("ascii")
    else:
        text = path.read_text(encoding="ascii")
    return list(csv.DictReader(io.StringIO(text)))


def load_variant(name: str, root: Path | None = None):
    """Return ``(rows, table, switches)`` for one variant.

    ``rows`` maps every column to a numpy array (int32 for the integer
    columns, float32 otherwise, ``variant`` dropped); ``table`` maps every
    ``read_param`` name to a per-UTYPE float32 array; ``switches`` maps the
    switch ints and the module arrays.
    """
    d = Path(root) if root is not None else UCM_ORACLE_DIR
    raw = _read_csv(d / f"{name}.csv.gz")
    rows: dict[str, np.ndarray] = {}
    for key in raw[0]:
        if key == "variant":
            continue
        if key in _INT_COLUMNS:
            rows[key] = np.asarray([int(r[key]) for r in raw], dtype=np.int32)
        else:
            rows[key] = np.asarray([_f32(r[key]) for r in raw], dtype=np.float32)
    traw = _read_csv(d / f"{name}-table.csv")
    table = {key: np.asarray([_f32(r[key]) for r in traw], dtype=np.float32)
             for key in traw[0] if key != "utype"}
    sraw = _read_csv(d / f"{name}-switches.csv")
    switches: dict[str, object] = {}
    arrays: dict[str, list[np.float32]] = {}
    for r in sraw:
        key, value = r["name"], r["value"]
        base = key.rstrip("0123456789")
        if base != key:
            arrays.setdefault(base, []).append(_f32(value))
        elif key in ("fgr", "oasis"):
            switches[key] = _f32(value)
        else:
            switches[key] = int(value)
    for base, values in arrays.items():
        switches[base] = np.asarray(values, dtype=np.float32)
    return rows, table, switches


def port_params(table, switches):
    from gpuwm.core.urban_ucm import pack_params_from_rows

    return pack_params_from_rows(table, switches)


def _inputs(rows, sel):
    return {
        "utype": rows["utype"][sel], "jmonth": rows["jmonth"][sel],
        "ta": rows["ta"][sel], "qa": rows["qa"][sel], "ua": rows["ua"][sel],
        "u1": rows["u1"][sel], "v1": rows["v1"][sel], "ssg": rows["ssg"][sel],
        "llg": rows["llg"][sel], "rain": rows["rain"][sel],
        "rhoo": rows["rhoo"][sel], "za": rows["za"][sel],
        "omg": rows["omg"][sel], "delt": rows["delt"][sel],
        "znt_in": rows["znt_in"][sel], "chs": rows["chs"][sel],
        "chs2": rows["chs2"][sel],
    }


def _state_from(rows, sel, suffix: str):
    n = int(np.count_nonzero(sel)) if sel.dtype == bool else len(sel)
    state: dict[str, np.ndarray] = {}
    for col, (plane, layer) in STATE_COLUMNS.items():
        values = rows[f"{col}{suffix}"][sel]
        if layer is None:
            state[plane] = values.astype(np.float32)
        else:
            state.setdefault(plane, np.zeros((4, n), np.float32))[layer] = values
    return state


def _flatten(outputs, after):
    port = dict(outputs)
    for col, (plane, layer) in STATE_COLUMNS.items():
        arr = after[plane]
        port[f"after:{col}"] = arr if layer is None else arr[layer]
    return port


def _reference(rows, sel):
    ref = {name: rows[name][sel] for name in OUTPUT_COLUMNS}
    for col in STATE_COLUMNS:
        ref[f"after:{col}"] = rows[col][sel]
    return ref


def replay(name: str, root: Path | None = None):
    """Every row from its own ``*_in`` state: ``(port, ref, codes)``."""
    from gpuwm.core.urban_ucm import run_columns

    rows, table, switches = load_variant(name, root)
    params = port_params(table, switches)
    sel = np.ones(rows["ta"].shape[0], dtype=bool)
    outputs, after, codes = run_columns(params, _inputs(rows, sel),
                                        _state_from(rows, sel, "_in"))
    return _flatten(outputs, after), _reference(rows, sel), codes


def carried_replay(name: str, root: Path | None = None):
    """Each column's first step from the fixture, later steps from the
    port's OWN state: the prognostic layers must stay on WRF's words."""
    from gpuwm.core.urban_ucm import run_columns

    rows, table, switches = load_variant(name, root)
    params = port_params(table, switches)
    steps = rows["step"]
    first = steps == steps.min()
    state = _state_from(rows, first, "_in")
    order = np.nonzero(first)[0]
    port_parts, ref_parts = [], []
    for step in range(int(steps.min()), int(steps.max()) + 1):
        idx = order + (step - int(steps.min()))
        sel = np.zeros(rows["ta"].shape[0], dtype=bool)
        sel[idx] = True
        if not np.all(rows["step"][idx] == step):
            raise ValueError(f"{name}: rows are not step-major within a column")
        outputs, after, codes = run_columns(params, _inputs(rows, idx), state)
        if np.any(codes):
            raise RuntimeError(f"{name}: kernel error codes {np.unique(codes)}")
        state = after
        port_parts.append(_flatten(outputs, after))
        ref_parts.append(_reference(rows, idx))
    port = {k: np.concatenate([p[k] for p in port_parts]) for k in port_parts[0]}
    ref = {k: np.concatenate([p[k] for p in ref_parts]) for k in ref_parts[0]}
    return port, ref


def ulp_table(port, ref) -> dict[str, int]:
    """Worst ULP distance per field (NaN-for-NaN counts as equal)."""
    out = {}
    for key, want in ref.items():
        got = np.asarray(port[key], dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        out[key] = int(np.max(fp32_ulp_distance(got, want))) if want.size else 0
    return out


# ---------------------------------------------------------------------------
# Noah coupling (tools/urban_wrf471_oracle/ucm_noah_oracle.F90)
# ---------------------------------------------------------------------------

#: The Noah hand-off, fixture ``tap_*`` column -> ``UrbanState.rural`` name.
NOAH_TAP_RURAL = {
    "tap_t1": "t1", "tap_sheat": "sheat",
    "tap_eta_kinematic": "eta_kinematic", "tap_eta": "eta",
    "tap_ssoil": "ssoil", "tap_albedok": "albedok", "tap_q1": "q1",
    "tap_sfctmp": "sfctmp", "tap_q2k": "q2k", "tap_sfcprs": "sfcprs",
    "tap_zlvl": "zlvl", "tap_soldn": "soldn", "tap_rainbl": "rainbl_used",
}
#: Grid words as WRF held them at the UCM's entry.
NOAH_TAP_FIELDS = {"tap_chs": "chs", "tap_chs2": "chs2", "tap_cqs2": "cqs2",
                   "tap_ust": "ust", "tap_znt": "znt", "tap_glw": "glw"}
#: Grid fields lsm leaves after its UCM block.
NOAH_FIELD_OUTPUTS = ("tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "qsfc",
                      "ust", "chs", "chs2", "cqs2")
#: Per-step urban outputs lsm writes, fixture name -> state plane.
NOAH_URBAN_OUTPUTS = {
    "ts_urb": "ts_urb2d", "sh_urb": "sh_urb2d", "lh_urb": "lh_urb2d",
    "g_urb": "g_urb2d", "rn_urb": "rn_urb2d", "psim_urb": "psim_urb2d",
    "psih_urb": "psih_urb2d", "gz1oz0_urb": "gz1oz0_urb2d",
    "u10_urb": "u10_urb2d", "v10_urb": "v10_urb2d", "th2_urb": "th2_urb2d",
    "q2_urb": "q2_urb2d", "ust_urb": "ust_urb2d", "akms_urb": "akms_urb2d",
}
#: ``ucm_noah_oracle.F90``'s fixed clock and step.
NOAH_FIXTURE_JULDAY = 196
NOAH_FIXTURE_JULYR = 2025
NOAH_FIXTURE_DT = 60.0


def load_noah(root: Path | None = None) -> dict[str, np.ndarray]:
    d = Path(root) if root is not None else UCM_ORACLE_DIR
    raw = _read_csv(d / "ucm-noah.csv.gz")
    ints = ("step", "case", "ivgtyp", "utype", "tapped")
    return {key: (np.asarray([int(r[key]) for r in raw], np.int32) if key in ints
                  else np.asarray([_f32(r[key]) for r in raw], np.float32))
            for key in raw[0]}


class _Solar:
    def __init__(self, hrang, julday, julyr):
        self.hrang = hrang
        self.julday = julday
        self.julyr = julyr


def noah_replay(params, *, carried: bool, root: Path | None = None,
                mutate=None):
    """Drive ``urban_ucm.after_lsm(lsm=2)`` step by step on the fixture.

    Returns ``(port, ref, untouched)``: per-step fields and urban state of
    the urban columns, WRF's words for the same, and whether the non-urban
    column's fields came through unchanged.
    """
    import cupy as cp

    from gpuwm.core.urban_ucm import UCM_STATE_PLANES, after_lsm

    rows = load_noah(root)
    steps = sorted(set(rows["step"].tolist()))
    ncol = int(np.count_nonzero(rows["step"] == steps[0]))
    ports, refs, untouched = [], [], True
    state_arrays = None
    for step in steps:
        sel = np.nonzero(rows["step"] == step)[0]
        assert np.all(rows["case"][sel] == np.arange(1, ncol + 1))
        urban_cols = rows["tapped"][sel] == 1

        def dev(values):
            return cp.asarray(np.asarray(values, np.float32).reshape(1, ncol))

        if state_arrays is None or not carried:
            state_arrays = {}
            for name, layers in UCM_STATE_PLANES:
                shape = (1, ncol) if layers == 0 else (layers, 1, ncol)
                state_arrays[name] = cp.zeros(shape, cp.float32)
            for col, (plane, layer) in STATE_COLUMNS.items():
                values = rows[f"{col}_in"][sel]
                if layer is None:
                    state_arrays[plane][...] = dev(values)
                else:
                    state_arrays[plane][layer] = dev(values)
        state = dict(state_arrays)
        state["utype_urb2d"] = cp.asarray(
            np.ascontiguousarray(rows["utype"][sel].reshape(1, ncol)))
        state["frc_urb2d"] = dev(rows["frc"][sel])
        state["urban_mask"] = cp.asarray(
            np.ascontiguousarray(urban_cols.reshape(1, ncol).astype(np.int32)))
        state["mh_urb2d"] = cp.zeros((1, ncol), cp.float32)
        state["rural"] = {name: dev(rows[col][sel])
                          for col, name in NOAH_TAP_RURAL.items()}
        fields = {name: dev(rows[col][sel]) for col, name in NOAH_TAP_FIELDS.items()}
        for name in ("albedo", "hfx", "qfx", "lh", "grdflx", "tsk", "qsfc"):
            fields[name] = dev(np.full(ncol, 777.0, np.float32))
        fields["swdown"] = dev(rows["swdown"][sel])
        fields["rainbl"] = dev(rows["rainbl_grid"][sel])
        before = {k: v.get().copy() for k, v in fields.items()}
        atmosphere = {"u": dev(rows["u1"][sel])[None],
                      "v": dev(rows["v1"][sel])[None]}
        solar = _Solar(dev(rows["omg"][sel]), NOAH_FIXTURE_JULDAY,
                       NOAH_FIXTURE_JULYR)
        if mutate is not None:
            mutate(state, fields)
        after_lsm(state, params, lsm=2, fields=fields, atmosphere=atmosphere,
                  dt=NOAH_FIXTURE_DT, itimestep=step + 1, solar=solar, cfg=None)
        for name, arr in fields.items():
            got = arr.get()[0]
            if not np.array_equal(got[~urban_cols].view(np.uint32),
                                  before[name][0][~urban_cols].view(np.uint32)):
                untouched = False
        port, ref = {}, {}
        for name in NOAH_FIELD_OUTPUTS:
            port[name] = fields[name].get()[0][urban_cols]
            ref[name] = rows[name][sel][urban_cols]
        for col, (plane, layer) in STATE_COLUMNS.items():
            arr = state_arrays[plane].get()
            got = arr[0] if layer is None else arr[layer][0]
            port[f"after:{col}"] = got[urban_cols]
            ref[f"after:{col}"] = rows[col][sel][urban_cols]
        for col, plane in NOAH_URBAN_OUTPUTS.items():
            port[col] = state_arrays[plane].get()[0][urban_cols]
            ref[col] = rows[col][sel][urban_cols]
        ports.append(port)
        refs.append(ref)
    port = {k: np.concatenate([p[k] for p in ports]) for k in ports[0]}
    ref = {k: np.concatenate([p[k] for p in refs]) for k in refs[0]}
    return port, ref, untouched


# ---------------------------------------------------------------------------
# Noah-MP coupling (tools/urban_wrf471_oracle/ucm_noahmp_oracle.F90)
# ---------------------------------------------------------------------------

#: Grid fields noahmp_urban reads and writes, fixture ``<name>_in`` inputs.
NOAHMP_FIELDS_IN = ("tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "qsfc",
                    "ust", "chs", "chs2", "cqs2")
#: The surface-driver override block's outputs (surface_driver.F:3383-3405).
NOAHMP_OVERRIDE_OUTPUTS = ("t2", "th2", "q2", "u10", "v10", "psim", "psih",
                           "gz1oz0", "akhs", "akms")


#: WRF v4.7.1's stock Noah-MP coupling fixture, and the same build with the
#: one named divergence the port carries (build_ucm_noahmp.sh with
#: UCM_T2_TEMPERATURE_FIX=1): the override block blends the UCM's 2 m value
#: as the absolute temperature module_sf_urban.F:1686 computes, where WRF's
#: surface_driver.F:3393 converts it again as a potential temperature.  The
#: two fixtures differ in T2 and TH2 on the urban rows and nowhere else.
NOAHMP_FIXTURE = "ucm-noahmp.csv.gz"
NOAHMP_T2FIX_FIXTURE = "ucm-noahmp-t2fix.csv.gz"


def load_noahmp(root: Path | None = None, *,
                fixture: str = NOAHMP_FIXTURE) -> dict[str, np.ndarray]:
    d = Path(root) if root is not None else UCM_ORACLE_DIR
    raw = _read_csv(d / fixture)
    ints = ("step", "case", "ivgtyp", "utype")
    return {key: (np.asarray([int(r[key]) for r in raw], np.int32) if key in ints
                  else np.asarray([_f32(r[key]) for r in raw], np.float32))
            for key in raw[0]}


def noahmp_replay(params, *, carried: bool, root: Path | None = None):
    """Drive ``after_lsm(lsm=4)`` then ``after_surface_diagnostics(lsm=4)``.

    Returns ``(port, ref, untouched)`` as :func:`noah_replay`.
    """
    import cupy as cp

    from gpuwm.core.urban_ucm import (UCM_STATE_PLANES, after_lsm,
                                      after_surface_diagnostics)

    rows = load_noahmp(root)
    steps = sorted(set(rows["step"].tolist()))
    ncol = int(np.count_nonzero(rows["step"] == steps[0]))
    ports, refs, untouched = [], [], True
    state_arrays = None
    for step in steps:
        sel = np.nonzero(rows["step"] == step)[0]
        assert np.all(rows["case"][sel] == np.arange(1, ncol + 1))
        urban_cols = rows["utype"][sel] > 0

        def dev(values):
            return cp.asarray(np.asarray(values, np.float32).reshape(1, ncol))

        if state_arrays is None or not carried:
            state_arrays = {}
            for name, layers in UCM_STATE_PLANES:
                shape = (1, ncol) if layers == 0 else (layers, 1, ncol)
                state_arrays[name] = cp.zeros(shape, cp.float32)
            for col, (plane, layer) in STATE_COLUMNS.items():
                values = rows[f"{col}_in"][sel]
                if layer is None:
                    state_arrays[plane][...] = dev(values)
                else:
                    state_arrays[plane][layer] = dev(values)
        state = dict(state_arrays)
        state["utype_urb2d"] = cp.asarray(
            np.ascontiguousarray(rows["utype"][sel].reshape(1, ncol)))
        state["frc_urb2d"] = dev(rows["frc"][sel])
        state["urban_mask"] = cp.asarray(
            np.ascontiguousarray(urban_cols.reshape(1, ncol).astype(np.int32)))
        state["mh_urb2d"] = cp.zeros((1, ncol), cp.float32)
        state["rural"] = {name: dev(rows[name][sel]) for name in
                          ("fvegxy", "t2mvxy", "t2mbxy", "q2mvxy", "q2mbxy")}
        fields = {name: dev(rows[f"{name}_in"][sel]) for name in NOAHMP_FIELDS_IN}
        fields["glw"] = dev(rows["glw"][sel])
        fields["znt"] = dev(rows["znt_in"][sel])
        fields["swdown"] = dev(rows["swdown"][sel])
        fields["rainbl"] = dev(rows["rainbl"][sel])
        fields["psfc"] = dev(rows["psfc"][sel])
        fields["t2"] = dev(rows["t2mbxy"][sel])
        fields["q2"] = dev(rows["q2mbxy"][sel])
        for name in ("th2", "u10", "v10", "psim", "psih", "gz1oz0", "akhs", "akms"):
            fields[name] = dev(np.zeros(ncol, np.float32))
        before = {k: v.get().copy() for k, v in fields.items()}
        atmosphere = {
            "u": dev(rows["u1"][sel])[None], "v": dev(rows["v1"][sel])[None],
            "temperature": dev(rows["t3d1"][sel])[None],
            "qv": dev(rows["qv1"][sel])[None],
            "p_interface": cp.concatenate([dev(rows["p8w1"][sel])[None],
                                           dev(rows["p8w2"][sel])[None]]),
            "dz": dev(rows["dz8w1"][sel])[None],
        }
        solar = _Solar(dev(rows["omg"][sel]), NOAH_FIXTURE_JULDAY,
                       NOAH_FIXTURE_JULYR)
        after_lsm(state, params, lsm=4, fields=fields, atmosphere=atmosphere,
                  dt=NOAH_FIXTURE_DT, itimestep=step + 1, solar=solar, cfg=None)
        after_surface_diagnostics(state, lsm=4, fields=fields,
                                  atmosphere=atmosphere, cfg=None)
        for name, arr in fields.items():
            got = arr.get()[0]
            if not np.array_equal(got[~urban_cols].view(np.uint32),
                                  before[name][0][~urban_cols].view(np.uint32)):
                untouched = False
        port, ref = {}, {}
        for name in NOAHMP_FIELDS_IN + NOAHMP_OVERRIDE_OUTPUTS:
            port[name] = fields[name].get()[0][urban_cols]
            ref[name] = rows[name][sel][urban_cols]
        for col, (plane, layer) in STATE_COLUMNS.items():
            arr = state_arrays[plane].get()
            got = arr[0] if layer is None else arr[layer][0]
            port[f"after:{col}"] = got[urban_cols]
            ref[f"after:{col}"] = rows[col][sel][urban_cols]
        for col, plane in NOAH_URBAN_OUTPUTS.items():
            port[col] = state_arrays[plane].get()[0][urban_cols]
            ref[col] = rows[col][sel][urban_cols]
        ports.append(port)
        refs.append(ref)
    port = {k: np.concatenate([p[k] for p in ports]) for k in ports[0]}
    ref = {k: np.concatenate([p[k] for p in refs]) for k in refs[0]}
    return port, ref, untouched
