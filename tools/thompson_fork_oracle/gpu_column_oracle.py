"""Bitwise column oracle: WOOF's GPU fork Thompson (mp_physics=28,
thompson_version = thompson_fork_snow_fall = "wrf_39_noaa") against the
unmodified NOAA-EMC/HRRR v4.1.21 fork Fortran, every output field.

usage (on a CUDA host, through the GPU mutex):
    GPUWM_WRF_EXACT=1 python gpu_column_oracle.py FORK_BUILD_DIR OUT_DIR \
        [--batch real:5] [--batch all:20] [--per-regime 4]

FORK_BUILD_DIR is tools/thompson_fork_oracle/build.sh's output
(run_columns_fork, the fork's pinned tables, CCN_ACTIVATE.BIN).  The fork
tables WOOF reads come from GPUWM_THOMPSON_FORK_TABLE_ROOT (or the default
staged root) and are verified against their pins.

A batch is ``SET:DT``: ``real`` is the 42 saved real-data columns
(tests/data/thompson_real_columns_wrf461.npz), ``all`` is those plus the
synthetic regime and edge columns of regime_columns.py.

Both codes receive the same float32 inputs bit for bit.  The adapter forms
the Exner function and layer depths itself (``prepare_microphysics_exner``
with WRF's moist-physics preparation, ``dz`` from the geopotential); the oracle reads those arrays
back from the adapter's scratch after the call (neither is written after
it forms them) and hands exactly those words to the Fortran.  The
comparison is per field, per cell: identical bits, ULP distance on the
float32 ordered line (+0 and -0 are 0 ULP apart but not bit-identical),
and the largest absolute difference.  Nothing is floored or masked: every
cell of every output counts.

OUT_DIR receives ``summary.json`` (every metric, the arithmetic mode, the
NVRTC compile receipt digest) and ``batch-<name>.npz`` (both sides' fields
and the inputs) for diagnosis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import regime_columns as RC  # noqa: E402

f32 = np.float32

SPECIES = RC.SPECIES
FORK_IN = ("th", "pii", "p", "w_lower", "dz", "hgt", *SPECIES)
FORK_OUT3 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
             "nifa", "th", "refl", "re_cloud", "re_ice", "re_snow")
FORK_OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
             "graupelncv", "sr", "frain")
#: Outputs WOOF publishes.  ``frain`` is the fork driver's own rain-fraction
#: diagnostic with no WOOF counterpart (the HRRR fork's driver argument);
#: it is reported, not compared.
COMPARED = FORK_OUT3 + FORK_OUT2[:-1]


class GpuColumnState:
    """The mp=28 DomainState surface over (nz, 1, ncol) CuPy arrays: the
    whole theta in ``thp`` over a zero ``thb``, the whole geopotential in
    ``php`` over a zero ``phb`` (as tools/thompson_real_column_parity's
    ColumnState, on the device)."""

    def __init__(self, cp, cols):
        ncol, nz = cols["p"].shape

        def vol(a):
            a = np.ascontiguousarray(np.asarray(a, f32).T)
            return cp.asarray(a.reshape(a.shape[0], 1, ncol))

        self._cp = cp
        self.p = vol(cols["p"])
        self.thb = cp.zeros((nz,), cp.float32)
        self.thp = vol(cols["th"])
        self.phb = cp.zeros((nz + 1,), cp.float32)
        self.php = vol(cols["geop"])
        self.w = vol(cols["w"])
        for s in SPECIES:
            setattr(self, s, vol(cols[s]))
        self.effc = cp.zeros((nz, 1, ncol), cp.float32)
        self.effi = cp.zeros((nz, 1, ncol), cp.float32)
        self.effs = cp.zeros((nz, 1, ncol), cp.float32)
        self.nwfa2d = cp.asarray(np.asarray(cols["nwfa2d"], f32)
                                 .reshape(1, ncol))
        self.nifa2d = cp.asarray(np.asarray(cols["nifa2d"], f32)
                                 .reshape(1, ncol))
        self.h_diabatic = cp.zeros((nz, 1, ncol), cp.float32)
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self._scratch = {}

    def scratch(self, shape, slot, dtype=None):
        cp = self._cp
        shape = tuple(shape)
        want = np.dtype(np.float32 if dtype is None else dtype)
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


def _cols(cp, a):
    a = cp.asnumpy(a)
    return np.ascontiguousarray(a.reshape(a.shape[0], -1).T, f32)


def run_woof(cols, dt, snow_fall="wrf_39_noaa", itimestep=2):
    import cupy as cp
    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    state = GpuColumnState(cp, cols)
    # WRF's ITIMESTEP from WOOF's carried clock (gpuwm.core.mynn_radiation.
    # wrf_itimestep): the fork switches the melting-snow echo off on step 1.
    state.elapsed_seconds = float(itimestep - 1) * float(dt)
    cfg = SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0,
                          thompson_version="wrf_39_noaa",
                          thompson_fork_snow_fall=snow_fall)
    diag = _apply_thompson_aerosol(state, cfg, float(dt), refl_10cm_due=True)
    if hasattr(cp.cuda, "Device"):
        cp.cuda.Device().synchronize()
    out = {s: _cols(cp, getattr(state, s)) for s in SPECIES}
    out["th"] = _cols(cp, state._scratch["mp_th"])
    out["refl"] = _cols(cp, state.physics.refl_10cm)
    out["re_cloud"] = _cols(cp, state.effc)
    out["re_ice"] = _cols(cp, state.effi)
    out["re_snow"] = _cols(cp, state.effs)
    for name in FORK_OUT2[:-1]:
        out[name] = np.ascontiguousarray(
            cp.asnumpy(getattr(diag, name)).ravel(), f32)
    used = {"pii": _cols(cp, state._scratch["mp_pii"]),
            "dz": _cols(cp, state._scratch["mp_dz8w"]),
            "hgt": _cols(cp, state._scratch["mp_z8w"])[:, :-1]}
    return out, used


def write_fork_input(path, inp, dt):
    ncol, nz = inp["p"].shape
    with open(path, "wb") as fh:
        np.array([ncol, nz], dtype="<i4").tofile(fh)
        np.array([dt], dtype="<f4").tofile(fh)
        for name in FORK_IN:
            a = inp["w"][:, :nz] if name == "w_lower" else inp[name]
            np.ascontiguousarray(np.asarray(a, f32).T, dtype="<f4").tofile(fh)
        for name in ("nwfa2d", "nifa2d"):
            np.ascontiguousarray(inp[name], dtype="<f4").tofile(fh)


def run_fork(build, inp, dt, work, itimestep=2):
    work.mkdir(parents=True, exist_ok=True)
    in_path, out_path = work / "fork-in.bin", work / "fork-out.bin"
    write_fork_input(in_path, inp, dt)
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    done = subprocess.run([str(build / "run_columns_fork"), str(in_path),
                           str(out_path), str(itimestep)], cwd=str(build),
                          env=env,
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"run_columns_fork failed:\n{done.stdout}\n"
                           f"{done.stderr}")
    ncol, nz = inp["p"].shape
    raw = np.fromfile(out_path, dtype="<f4")
    want = len(FORK_OUT3) * ncol * nz + len(FORK_OUT2) * ncol
    if raw.size != want:
        raise RuntimeError(f"{out_path}: {raw.size} words, expected {want}")
    out, off = {}, 0
    for name in FORK_OUT3:
        out[name] = raw[off:off + ncol * nz].reshape(nz, ncol).T.copy()
        off += ncol * nz
    for name in FORK_OUT2:
        out[name] = raw[off:off + ncol].copy()
        off += ncol
    in_path.unlink()
    out_path.unlink()
    return out, done.stdout


def ordered(a):
    """float32 bits on a monotone integer line (+0 and -0 both at 0)."""
    i = np.ascontiguousarray(a, f32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def compare(woof, fork):
    woof = np.ascontiguousarray(woof, f32)
    fork = np.ascontiguousarray(fork, f32)
    bits_differ = woof.view(np.uint32) != fork.view(np.uint32)
    both_nan = np.isnan(woof) & np.isnan(fork)
    bits_differ &= ~both_nan
    ulp = np.abs(ordered(woof) - ordered(fork))
    ulp = np.where(both_nan, 0, ulp)
    absdiff = np.abs(woof.astype(np.float64) - fork.astype(np.float64))
    absdiff = np.where(both_nan, 0.0, absdiff)
    res = {"cells": int(woof.size),
           "bit_mismatches": int(bits_differ.sum()),
           "max_ulp": int(ulp.max()) if ulp.size else 0,
           "cells_gt_1ulp": int((ulp > 1).sum()),
           "max_abs": float(np.nanmax(absdiff)) if absdiff.size else 0.0,
           "nan_woof": int(np.isnan(woof).sum()),
           "nan_fork": int(np.isnan(fork).sum())}
    if bits_differ.any():
        worst = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
        res["worst_at"] = [int(i) for i in worst]
        res["woof_at"] = float(woof[worst])
        res["fork_at"] = float(fork[worst])
        if woof.ndim == 2:
            res["columns_differing"] = int(bits_differ.any(axis=1).sum())
        else:
            res["columns_differing"] = int(bits_differ.sum())
    return res


def batch_columns(name, per_regime, seed):
    real, real_labels, _ = RC.real_columns()
    if name == "real":
        return {k: np.asarray(v, f32) for k, v in real.items()}, real_labels
    if name == "all":
        return RC.all_columns(per_regime, seed)
    if name == "synthetic":
        return RC.stack(RC.synthetic_columns(per_regime, seed))
    raise KeyError(name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("out")
    ap.add_argument("--batch", action="append", default=None,
                    help="SET:DT[:ITIMESTEP], SET in real/all/synthetic "
                    "(repeatable; ITIMESTEP defaults to 2)")
    ap.add_argument("--per-regime", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--snow-fall", default="wrf_39_noaa",
                    choices=("wrf_39_noaa", "blend"))
    ap.add_argument("--host", action="store_true",
                    help="run WOOF's kernels on the CPU through "
                    "tools/thompson_real_column_parity/host_backend.py "
                    "(g++, no FMA, the host libm) instead of the GPU: "
                    "separates libm words from formulation")
    args = ap.parse_args(argv)
    batches = args.batch or ["real:5", "all:20"]

    if args.host:
        sys.path.insert(0, str(ROOT / "tools" / "thompson_real_column_parity"))
        import host_backend
        host_backend.install()
    import gpuwm  # noqa: F401  (installs the strict hook when selected)
    from gpuwm import wrf_exact
    import cupy as cp

    build = Path(args.build).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "work"
    dev = ({"name": "host (g++ -O2 -ffp-contract=off)"} if args.host else
           cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id))
    summary = {
        "scheme": "Thompson mp_physics=28, thompson_version=wrf_39_noaa, "
                  f"thompson_fork_snow_fall={args.snow_fall}",
        "oracle": "NOAA-EMC/HRRR v4.1.21 module_mp_thompson.F "
                  "(sha256 4d600111...), run_columns_fork",
        "strict": bool(wrf_exact.ENABLED),
        "strict_options": list(wrf_exact.STRICT_OPTIONS)
        if wrf_exact.ENABLED else None,
        "gpu": dev["name"].decode() if isinstance(dev["name"], bytes)
        else str(dev["name"]),
        "cupy": None if args.host else cp.__version__,
        "host": platform.platform(),
        "batches": {},
    }
    for spec in batches:
        parts = spec.split(":")
        set_name, dt = parts[0], float(parts[1])
        itimestep = int(parts[2]) if len(parts) > 2 else 2
        cols, labels = batch_columns(set_name, args.per_regime, args.seed)
        woof, used = run_woof(cols, dt, args.snow_fall, itimestep)
        inp = {k: np.asarray(v, f32) for k, v in cols.items()}
        inp.update(used)
        fork, log = run_fork(build, inp, dt, work, itimestep)
        # WOOF publishes effective radii in microns (the radiation
        # contract): its kernel forms the fork driver's clamped metres and
        # multiplies by 1.0e6f (thompson_aerosol_state.cu).  The fork's
        # metres go through that same float32 multiply, so equal metres give
        # equal words; the raw metres are kept as fork_<name>_m.
        for name in ("re_cloud", "re_ice", "re_snow"):
            fork[name + "_m"] = fork[name]
            fork[name] = (fork[name] * f32(1.0e6)).astype(f32)
        fields = {name: compare(woof[name], fork[name]) for name in COMPARED}
        total = sum(v["bit_mismatches"] for v in fields.values())
        per_col = np.zeros(len(labels), np.int64)
        for name in COMPARED:
            a = np.ascontiguousarray(woof[name], f32).view(np.uint32)
            b = np.ascontiguousarray(fork[name], f32).view(np.uint32)
            d = a != b
            per_col += d.reshape(len(labels), -1).sum(axis=1)
        key = f"{set_name}@dt{dt:g}" + (f"@step{itimestep}"
                                            if itimestep != 2 else "")
        summary["batches"][key] = {
            "ncol": len(labels), "nz": int(cols["p"].shape[1]), "dt": dt,
            "itimestep": itimestep,
            "total_bit_mismatches": int(total),
            "max_ulp": max(v["max_ulp"] for v in fields.values()),
            "fields": fields,
            "columns_with_any_mismatch": int((per_col > 0).sum()),
            "mismatches_by_column": {labels[i]: int(per_col[i])
                                     for i in np.nonzero(per_col)[0]},
            "fork_log": log.strip(),
        }
        np.savez_compressed(
            out_dir / f"batch-{set_name}-dt{dt:g}-step{itimestep}.npz",
            labels=np.array(labels),
            **{f"woof_{k}": v for k, v in woof.items()},
            **{f"fork_{k}": v for k, v in fork.items()},
            **{f"in_{k}": v for k, v in inp.items()})
        print(f"{key}: {len(labels)} columns, {total} bit mismatches, "
              f"max ulp {summary['batches'][key]['max_ulp']}", flush=True)
    receipt = wrf_exact.compile_receipt()
    summary["nvrtc_compiles"] = sum(1 for r in receipt["records"]
                                    if r["kind"] == "nvrtc_compile")
    summary["compile_receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    worst = {k: {"total": v["total_bit_mismatches"], "max_ulp": v["max_ulp"]}
             for k, v in summary["batches"].items()}
    print(json.dumps({"strict": summary["strict"], "batches": worst}))


if __name__ == "__main__":
    main()
