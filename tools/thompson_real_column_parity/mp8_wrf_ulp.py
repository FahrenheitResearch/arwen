#!/usr/bin/env python3
"""Classic Thompson (mp_physics=8) against WRF v4.6.1, in float32 ULP.

The committed fixture ``tests/data/thompson_real_columns_wrf461_mp8.npz``
holds 42 saved real-data model columns and every word unmodified WRF v4.6.1
``module_mp_thompson.F`` wrote for them as classic Thompson
(``make_fixture.py --mp8``: gfortran 13.3, glibc 2.39, ``-O2
-fno-tree-vectorize``, no libmvec).  This command runs the production
adapter ``gpuwm.core.microphysics._apply_thompson`` on the same float32
inputs and reports, per output field, the largest FP32 total-order ULP
distance from WRF (``gpuwm.core.fp32_ulp``), the count of differing words,
and the worst cell.

``--backend gpu`` (the default) runs the CUDA kernels on the current device,
exactly as a forecast compiles them.  ``--backend host`` runs the same
kernel source compiled as C++ on the CPU (``host_backend``: IEEE per-op
rounding, no contraction, glibc libm), which isolates the transcription
from the device toolchain.

The inputs are formed as ``real_column_parity.prepare`` forms them for WRF:
``pii = (p/P0)**RCP`` in numpy float32 (glibc powf), the whole geopotential
in ``php`` over a zero ``phb``, the whole theta in ``thp`` over a zero
``thb``.

Effective radii: WRF writes metres (``mp_gt_driver``'s clamps); the port
stores microns after the same clamps, as the last operation ``* 1.0e6f``.
The comparison applies that float32 multiply to WRF's word.

``--rates`` (host build only) also compares every process rate the unit
reads back (thompson_classic.cu's GPUWM_HOST_RATES block, WRF's cp1/cp2
order) with the fixture's ``rate_*`` words, in float32 as the fixture
stores them.

usage: mp8_wrf_ulp.py [FIXTURE] [--backend gpu|host] [--rates] [--json PATH]
                      [--json-only] [--save WORDS.npz]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.thompson_aerosol_column_oracle.column_state import (
    ColumnState as OracleColumnState,
)

DEFAULT_FIXTURE = ROOT / "tests" / "data" / "thompson_real_columns_wrf461_mp8.npz"
SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")
LEVEL_FIELDS = SPECIES + ("th", "refl", "re_cloud", "re_ice", "re_snow")
SURFACE_FIELDS = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                  "graupelncv", "sr")
f32 = np.float32


def _prepare(z, xp):
    from gpuwm.core import constants as C
    p = np.ascontiguousarray(z["col_p"], dtype=f32)
    geop = np.ascontiguousarray(z["col_geop"], dtype=f32)
    inp = {
        "th": np.ascontiguousarray(z["col_th"], dtype=f32),
        "p": p,
        "geop": geop,
        "w": np.ascontiguousarray(z["col_w"], dtype=f32),
    }
    for s in SPECIES:
        inp[s] = np.ascontiguousarray(z[f"col_{s}"], dtype=f32)
    return inp


class _State(OracleColumnState):
    _scratch_error = " reused"

    def __init__(self, inp, xp):
        self._xp = xp
        super().__init__(inp, xp, SPECIES, copy_inputs=True, host_zeros=True)


def _host(a):
    a = a.get() if hasattr(a, "get") else np.asarray(a)
    return np.asarray(a, dtype=f32)


def _cols(a):
    a = _host(a)
    return np.ascontiguousarray(a.reshape(a.shape[0], -1).T)


#: WRF rates classic Thompson does not carry: the aerosol scavenging terms
#: feed only nwfaten/nifaten, which mp_gt_driver writes back only when
#: is_aerosol_aware (:1316-1340).  The unit reads them back as zero.
RATES_NOT_CARRIED = ("pna_rca", "pna_sca", "pna_gca",
                     "pnd_rcd", "pnd_scd", "pnd_gcd")


def run(fixture=DEFAULT_FIXTURE, backend="gpu", rates=False, save=None):
    rate_buffer = None
    if backend == "host":
        import host_backend
        host_backend.install()
        host_backend.set_rates(bool(rates))
    elif rates:
        raise ValueError("the rate readback is a host-build measurement")
    import cupy as xp
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    from gpuwm.core.microphysics import _apply_thompson

    z = np.load(fixture)
    if int(z["mp_physics"]) != 8:
        raise SystemExit(f"{fixture} is not an mp=8 fixture")
    dt = float(z["dt"])
    inp = _prepare(z, xp)
    state = _State(inp, xp)
    if rates:
        import ctypes
        ncol0, nz0 = z["col_p"].shape
        rate_buffer = np.full((64, ncol0 * nz0), np.nan)
        lib = host_backend.build_module("thompson_classic").lib
        lib.gpuwm_host_rate_bind.argtypes = [
            np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
            ctypes.c_longlong, ctypes.c_int]
        lib.gpuwm_host_rate_bind(rate_buffer, ncol0 * nz0, 64)
    cfg = SimpleNamespace(mp_physics=8, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_461",
                          thompson_fork_snow_fall="blend")
    diag = _apply_thompson(state, cfg, dt, refl_10cm_due=True)
    if hasattr(xp, "cuda"):
        xp.cuda.get_current_stream().synchronize()

    got = {s: _cols(getattr(state, s)) for s in SPECIES}
    got["th"] = _cols(state._scratch["mp_th"])
    got["refl"] = _cols(state.physics.refl_10cm)
    got["re_cloud"] = _cols(state.effc)
    got["re_ice"] = _cols(state.effi)
    got["re_snow"] = _cols(state.effs)
    for name in SURFACE_FIELDS:
        got[name] = _host(getattr(diag, name)).ravel()
    if save is not None:
        # The unit's own words, for cell-level comparisons by the caller.
        np.savez(save, **{k: np.asarray(v, dtype=f32) for k, v in got.items()})

    fields = {}
    for name in LEVEL_FIELDS + SURFACE_FIELDS:
        want = np.asarray(z[f"wrf_{name}"], dtype=f32)
        if name.startswith("re_"):
            want = (want * f32(1.0e6)).astype(f32)
        have = np.asarray(got[name], dtype=f32).reshape(want.shape)
        dist = fp32_ulp_distance(have, want)
        diff = have.view(np.uint32) != want.view(np.uint32)
        entry = {"max_ulp": int(dist.max()), "mismatches": int(diff.sum()),
                 "values": int(dist.size)}
        if diff.any():
            flat = int(np.argmax(dist))
            idx = np.unravel_index(flat, want.shape)
            entry["worst"] = {
                "cell": [int(i) for i in idx],
                "got": float(have[idx]), "want": float(want[idx]),
                "got_bits": f"0x{int(have.view(np.uint32)[idx]):08x}",
                "want_bits": f"0x{int(want.view(np.uint32)[idx]):08x}"}
            if want.ndim == 2:
                cols, levs = np.nonzero(diff)
                entry["columns"] = int(np.unique(cols).size)
        fields[name] = entry
    if rates:
        lib.gpuwm_host_rate_bind(np.zeros((1, 1)), 0, 0)
        from instrument_wrf_rates import RATES
        ncol0, nz0 = z["col_p"].shape
        for i, name in enumerate(RATES):
            if name in RATES_NOT_CARRIED or f"rate_{name}" not in z.files:
                continue
            want = np.asarray(z[f"rate_{name}"], dtype=f32)
            have = np.nan_to_num(rate_buffer[i].reshape(nz0, ncol0).T,
                                 nan=0.0).astype(f32)
            # WRF writes no checkpoint row for a column it leaves at :2020
            # (NaN in the fixture); the unit reads back nothing there.
            ok = np.isfinite(want)
            dist = fp32_ulp_distance(np.where(ok, have, f32(0)),
                                     np.where(ok, want, f32(0)))
            diff = ok & (have.view(np.uint32) != want.view(np.uint32))
            fields[f"rate_{name}"] = {
                "max_ulp": int(dist.max()), "mismatches": int(diff.sum()),
                "values": int(ok.sum())}
    worst = max(fields, key=lambda n: (fields[n]["max_ulp"], n))
    return {
        "fixture": str(fixture), "backend": backend,
        "ncol": int(z["col_p"].shape[0]), "nz": int(z["col_p"].shape[1]),
        "fields": fields,
        "overall": {"max_ulp": fields[worst]["max_ulp"], "worst_field": worst,
                    "mismatches": sum(e["mismatches"] for e in fields.values()),
                    "values": sum(e["values"] for e in fields.values()),
                    "bitwise": all(e["mismatches"] == 0
                                   for e in fields.values())},
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture", nargs="?", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument("--backend", choices=("gpu", "host"), default="gpu")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--rates", action="store_true",
                    help="also compare every process rate (host build)")
    ap.add_argument("--save", type=Path,
                    help="also write the unit's output words to this .npz")
    ap.add_argument("--json-only", action="store_true",
                    help="print only the JSON report")
    args = ap.parse_args(argv)
    report = run(args.fixture, args.backend, rates=args.rates,
                 save=args.save)
    if args.json_only:
        print(json.dumps(report))
        return 0
    print(f"field,max_ulp,mismatches,values  [{report['backend']}]")
    for name, e in report["fields"].items():
        print(f"{name},{e['max_ulp']},{e['mismatches']},{e['values']}")
    o = report["overall"]
    print(f"overall,{o['max_ulp']},{o['mismatches']},{o['values']}"
          f"  worst={o['worst_field']} bitwise={o['bitwise']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
