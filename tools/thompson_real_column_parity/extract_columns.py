#!/usr/bin/env python3
"""Cut microphysics input columns out of saved gpuwm model states.

Three kinds of source, each giving the exact float32 inputs one
``mp_physics=28`` call would receive for every column of a domain interior:

``history FILE``
    A wrfout-style history frame: pressure ``P + PB``, full theta
    ``T + 300``, geopotential ``PH + PHB``, ``W``, the eleven mp=28 moments
    and ``QNWFA2D`` / ``QNIFA2D``.
``restart RESTART.npz NESTBASE.npz``
    A gpuwm restart: ``state/p`` (full pressure), ``thb + state/thp``,
    ``phb + state/php``, ``state/w`` and the moments, with the base state
    from the nest base file the restart was written against.
``analysis RESTART.npz NESTBASE.npz INCREMENT.npz``
    The restart plus a saved analysis increment (theta, vapour, w and every
    moment), negative masses and numbers set to zero, pressure and
    geopotential held at the restart's.  This is not the product's analysis
    step; it is a far-from-equilibrium state built from real fields, which
    is what the condensation, activation and nucleation branches need.

Every sum is formed in float32, the way the model forms it.  The output is
one ``.npz`` of ``(ncol, nz)`` float32 fields (``w`` and ``geop`` are
``(ncol, nz + 1)``) plus the column indices, readable without gpuwm.

usage: extract_columns.py OUT.npz RIM KIND PATH [PATH ...]
"""

from __future__ import annotations

import sys

import numpy as np

MOMENTS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
           "nwfa", "nifa")
HISTORY_NAMES = {
    "qv": "QVAPOR", "qc": "QCLOUD", "qr": "QRAIN", "qi": "QICE",
    "qs": "QSNOW", "qg": "QGRAUP", "ni": "QNICE", "nr": "QNRAIN",
    "nc": "QNCLOUD", "nwfa": "QNWFA", "nifa": "QNIFA",
}
f32 = np.float32


def _from_history(path):
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)

        def v(name):
            a = np.asarray(nc.variables[name][:])
            return a[0] if a.shape[0] == 1 else a
        out = {
            "p": v("P").astype(f32) + v("PB").astype(f32),
            "th": v("T").astype(f32) + f32(300.0),
            "geop": v("PH").astype(f32) + v("PHB").astype(f32),
            "w": v("W").astype(f32),
            "nwfa2d": v("QNWFA2D").astype(f32),
            "nifa2d": v("QNIFA2D").astype(f32),
        }
        for key, name in HISTORY_NAMES.items():
            out[key] = v(name).astype(f32)
        meta = {"mp_physics": int(nc.getncattr("MP_PHYSICS")),
                "dt": float(nc.getncattr("DT"))}
    return out, meta


def _from_restart(path, base_path):
    import json
    z = np.load(path)
    base = np.load(base_path)
    header = json.loads(bytes(z["__gpuwm_restart_header__"]).decode())
    cfg = header["config"]
    out = {
        "p": z["state/p"].astype(f32),
        "th": base["thb"].astype(f32) + z["state/thp"].astype(f32),
        "geop": base["phb"].astype(f32) + z["state/php"].astype(f32),
        "w": z["state/w"].astype(f32),
        "nwfa2d": z["state/nwfa2d"].astype(f32),
        "nifa2d": z["state/nifa2d"].astype(f32),
    }
    for key in MOMENTS:
        out[key] = z[f"state/{key}"].astype(f32)
    meta = {"mp_physics": int(cfg["mp_physics"]), "dt": float(cfg["dt"])}
    return out, meta


def _add_increment(state, inc_path):
    inc = np.load(inc_path)
    out = dict(state)
    th = state["th"].astype(np.float64) + inc["thp"].astype(np.float64)
    out["th"] = th.astype(f32)
    out["w"] = (state["w"].astype(np.float64)
                + inc["w"].astype(np.float64)).astype(f32)
    for key in MOMENTS:
        if key in inc.files:
            value = state[key].astype(np.float64) + inc[key].astype(np.float64)
            out[key] = np.maximum(value, 0.0).astype(f32)
    return out


def _columns(state, rim):
    nz, ny, nx = state["p"].shape
    jj, ii = np.meshgrid(np.arange(rim, ny - rim), np.arange(rim, nx - rim),
                         indexing="ij")
    jj, ii = jj.ravel(), ii.ravel()
    cols = {}
    for key, value in state.items():
        if value.ndim == 3:
            cols[key] = np.ascontiguousarray(value[:, jj, ii].T)
        else:
            cols[key] = np.ascontiguousarray(value[jj, ii])
    cols["col_j"] = jj.astype(np.int32)
    cols["col_i"] = ii.astype(np.int32)
    return cols


def main(argv):
    if len(argv) < 5:
        print(__doc__)
        return 2
    out_path, rim, kind, paths = argv[1], int(argv[2]), argv[3], argv[4:]
    if kind == "history":
        state, meta = _from_history(paths[0])
    elif kind == "restart":
        state, meta = _from_restart(paths[0], paths[1])
    elif kind == "analysis":
        state, meta = _from_restart(paths[0], paths[1])
        state = _add_increment(state, paths[2])
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    if meta["mp_physics"] != 28:
        raise SystemExit(f"source runs mp_physics={meta['mp_physics']}")
    cols = _columns(state, rim)
    np.savez_compressed(out_path, dt=np.float32(meta["dt"]),
                        kind=np.array(kind), sources=np.array(paths), **cols)
    print(out_path, cols["p"].shape, "dt", meta["dt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
