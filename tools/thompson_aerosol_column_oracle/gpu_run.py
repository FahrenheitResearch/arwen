#!/usr/bin/env python3
"""Run WOOF's production mp=28 adapter on the GPU over an oracle column set.

One call of ``gpuwm.core.microphysics_aerosol._apply_thompson_aerosol``
(the ``mp_physics=28`` path the model runs, ``thompson_version=wrf_461``,
``thompson_fork_snow_fall=blend``) over every column at once, laid out as
``(nz, 1, ncol)`` device arrays.  The whole potential temperature sits in
``thp`` over a zero ``thb`` and the whole geopotential in ``php`` over a
zero ``phb``, so the adapter's own sums reproduce the raw inputs bit for
bit.  The adapter forms the Exner function and the layer depths itself;
those device arrays are saved too and are what the WRF side is handed, so
both codes see identical microphysics inputs.

The arithmetic is chosen by the environment before import:
``GPUWM_WRF_EXACT=1`` is the strict build (no FMA, no flush-to-zero, IEEE
divide and square root, WOOF's own float32 libm where a unit carries it),
unset is default arithmetic.  The NVRTC options actually used are saved
with the outputs.

usage: gpu_run.py COLUMNS.npz OUT.npz --dt DT [--stages]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oracle_io as io  # noqa: E402

f32 = np.float32

STAGE_LAUNCHERS = (
    ("cold", "gpuwm.core.thompson_aerosol_cold",
     "launch_aa_cold_network_from_owner"),
    ("warm", "gpuwm.core.thompson_aerosol_warm",
     "launch_aerosol_warm_source_network_from_owner"),
    ("sources", "gpuwm.core.thompson_aerosol_warm", "launch_ncten_balance"),
    ("condensation", "gpuwm.core.thompson_aerosol_sat",
     "launch_aerosol_saturation_adjust"),
    ("rain_evaporation", "gpuwm.core.thompson_aerosol_sat",
     "launch_aerosol_rain_evaporation"),
)


class ColumnState:
    """The mp=28 ``DomainState`` surface over ``(nz, 1, ncol)`` device
    arrays (the attribute set and scratch protocol of
    tools/thompson_real_column_parity/real_column_parity.py's host
    ``ColumnState``)."""

    def __init__(self, cols, cp):
        ncol, nz = cols["p"].shape
        self._cp = cp

        def vol(a):
            a = np.asarray(a, f32)
            return cp.asarray(np.ascontiguousarray(
                a.T.reshape(a.shape[1], 1, ncol)))

        self.p = vol(cols["p"])
        self.thb = cp.zeros((nz,), f32)
        self.thp = vol(cols["th"])
        self.phb = cp.zeros((nz + 1,), f32)
        self.php = vol(cols["geop"])
        self.w = vol(cols["w"])
        for s in io.SPECIES:
            setattr(self, s, vol(cols[s]))
        self.effc = cp.zeros((nz, 1, ncol), f32)
        self.effi = cp.zeros((nz, 1, ncol), f32)
        self.effs = cp.zeros((nz, 1, ncol), f32)
        self.nwfa2d = cp.asarray(np.asarray(cols["nwfa2d"], f32)
                                 .reshape(1, ncol))
        self.nifa2d = cp.asarray(np.asarray(cols["nifa2d"], f32)
                                 .reshape(1, ncol))
        self.h_diabatic = cp.zeros((nz, 1, ncol), f32)
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self._scratch = {}

    def scratch(self, shape, slot, dtype=None):
        cp = self._cp
        shape = tuple(shape)
        want = np.dtype(f32 if dtype is None else dtype)
        value = self._scratch.get(slot)
        if value is None:
            value = cp.zeros(shape, dtype=want)
            self._scratch[slot] = value
        elif value.shape != shape or value.dtype != want:
            raise ValueError(f"scratch slot {slot!r} reused with a new "
                             "shape or dtype")
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def cols2d(a):
    """``(nz, 1, ncol)`` device or host array -> ``(ncol, nz)`` float32."""
    a = a.get() if hasattr(a, "get") else np.asarray(a)
    return np.ascontiguousarray(a.reshape(a.shape[0], -1).T).astype(f32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("columns")
    ap.add_argument("out")
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--stages", action="store_true")
    ap.add_argument("--serial-fallout", action="store_true",
                    help="measurement only: use the retained pre-review fallout")
    ap.add_argument("--legacy-warm-mask", action="store_true",
                    help="negative control: discard the melting-level value 2")
    a = ap.parse_args(argv)

    import cupy as cp
    import gpuwm  # noqa: F401  (installs the strict compiler hook if selected)
    from gpuwm import wrf_exact
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol
    if a.serial_fallout:
        from gpuwm.core import thompson_aerosol_sed as sed
        original_kernel = sed.aerosol_kernel
        def serial_kernel(module, symbol):
            if "_levels_" not in symbol:
                return original_kernel(module, symbol)
            kernel = original_kernel(module, symbol.replace("_levels", ""))
            def launch(grid, block, args):
                ncol = int(args[-2]) * int(args[-1])
                kernel(((ncol + 31) // 32,), (32,), args)
            return launch
        sed.aerosol_kernel = serial_kernel
    if a.legacy_warm_mask:
        from gpuwm.core import thompson_aerosol_state as aa_state
        def legacy_mask(temperature, mask):
            cp.greater_equal(temperature, cp.float32(273.15), out=mask)
        aa_state.launch_aa_entry_warm_mask = legacy_mask

    cols = io.load(a.columns)
    state = ColumnState(cols, cp)

    stages, originals = {}, []
    if a.stages:
        for stage, module_name, attr in STAGE_LAUNCHERS:
            module = importlib.import_module(module_name)
            original = getattr(module, attr)
            originals.append((module, attr, original))

            def wrapped(*args, _orig=original, _stage=stage, **kwargs):
                result = _orig(*args, **kwargs)
                cp.cuda.Device().synchronize()
                snap = {s: cols2d(getattr(state, s)) for s in io.SPECIES}
                snap["T"] = cols2d(state._scratch["mp_thompson_temperature"])
                for slot, value in state._scratch.items():
                    if slot.startswith("mp_thompson_aero_") and \
                            value.ndim == 3:
                        snap[slot[len("mp_thompson_aero_"):]] = cols2d(value)
                stages[_stage] = snap
                return result

            setattr(module, attr, wrapped)

    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_461",
                          thompson_fork_snow_fall="blend")
    try:
        diag = _apply_thompson_aerosol(state, cfg, float(a.dt),
                                       refl_10cm_due=True)
        cp.cuda.Device().synchronize()
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)

    out = {s: cols2d(getattr(state, s)) for s in io.SPECIES}
    out["th"] = cols2d(state._scratch["mp_th"])
    out["refl"] = cols2d(state.physics.refl_10cm)
    out["re_cloud"] = cols2d(state.effc)
    out["re_ice"] = cols2d(state.effi)
    out["re_snow"] = cols2d(state.effs)
    for name in io.OUT2:
        v = getattr(diag, name)
        v = v.get() if hasattr(v, "get") else np.asarray(v)
        out[name] = np.asarray(v, f32).ravel()
    # The microphysics inputs the adapter formed on the device.
    out["in_pii"] = cols2d(state._scratch["mp_pii"])
    out["in_dz"] = cols2d(state._scratch["mp_dz8w"])
    z8w = cols2d(state._scratch["mp_z8w"])
    out["in_hgt"] = np.ascontiguousarray(z8w[:, :-1])
    for stage, snap in stages.items():
        for k, v in snap.items():
            out[f"stage_{stage}_{k}"] = v
    receipt = wrf_exact.compile_receipt()
    meta = {"dt": a.dt, "strict": bool(wrf_exact.ENABLED),
            "serial_fallout": a.serial_fallout,
            "legacy_warm_mask": a.legacy_warm_mask,
            "strict_options": receipt["strict_options"],
            "nvrtc_compiles": sum(1 for r in receipt["records"]
                                  if r["kind"] == "nvrtc_compile"),
            "cupy": cp.__version__,
            "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
            "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "cc": cp.cuda.Device().compute_capability,
            "env_exact": os.environ.get("GPUWM_WRF_EXACT", "")}
    np.savez(a.out, meta=np.array(json.dumps(meta)), **out)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
