#!/usr/bin/env python3
"""Random-column stress sets for the Morrison WRF oracle.

``make`` writes input files for ``run_morrison_columns`` (WRF's own
MP_MORR_TWO_MOMENT driver); ``compare`` runs gpuwm's Morrison over the same
inputs and grades every returned word against WRF's.

The columns are drawn to reach the scheme's branches, not to be
climatology: warm, mixed and glaciated layers, sub- and supersaturation,
every hydrometeor at threshold-straddling and active amounts, number
moments from zero to large, cumulus number seeding, and per-file time
steps from 10 s to 90 s under both morr_rimed_ice modes.

usage:
  stress_columns.py make OUTDIR [--seed S] [--ncol N] [--nz NZ]
  stress_columns.py compare IN.bin WRF_OUT.bin [--row37-off] [--json F]
  stress_columns.py npz OUT.npz IN.bin WRF_OUT.bin [IN.bin WRF_OUT.bin ...]
                    [--columns N]
  stress_columns.py compare-npz FIXTURE.npz [--row37-off] [--json F]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEVEL_IN = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
            "ng", "rho", "pii", "p", "dz", "qrcuten", "qscuten", "qicuten")
SURF = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
        "graupelncv", "sr")
LEVEL_OUT = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "ns", "nr",
             "ng", "refl_10cm")
#: (mode, dt) of each file ``make`` writes.
FILES = ((0, 10.0), (1, 20.0), (0, 30.0), (1, 45.0), (0, 60.0), (1, 90.0))


def _loguniform(rng, lo, hi, size):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size))


def make_columns(rng, ncol, nz):
    f32 = np.float32
    dz_col = rng.uniform(60.0, 600.0, ncol)
    z = (np.arange(nz)[:, None] + 0.5) * dz_col[None, :]
    psfc = rng.uniform(95000.0, 102500.0, ncol)
    p = psfc[None, :] * np.exp(-z / 8000.0)
    t0 = rng.uniform(225.0, 312.0, ncol)
    lapse = rng.uniform(-0.002, 0.0095, ncol)
    t = np.maximum(t0[None, :] - lapse[None, :] * z, 180.0)
    t += rng.normal(0.0, 0.3, t.shape)
    pii = (p / 100000.0) ** (287.0 / 1004.5)
    th = t / pii
    ew = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    qsat = 0.622 * ew / np.maximum(p - ew, 1.0)
    rh = np.clip(rng.uniform(0.15, 1.15, ncol)[None, :]
                 + rng.normal(0.0, 0.08, t.shape), 0.01, 1.3)
    qv = rh * qsat
    fields = {"th": th, "qv": qv, "p": p, "pii": pii,
              "dz": np.broadcast_to(dz_col[None, :], (nz, ncol)),
              "rho": p / (287.0 * t)}
    numbers = {"qc": None, "qr": ("nr", 1e2, 3e7), "qi": ("ni", 1e1, 3e7),
               "qs": ("ns", 1e1, 3e6), "qg": ("ng", 1e0, 3e6)}
    for q, nspec in numbers.items():
        present = rng.uniform(size=ncol) < 0.55
        top = rng.integers(0, nz, ncol)
        depth = rng.integers(1, nz, ncol)
        k = np.arange(nz)[:, None]
        layer = (k <= top[None, :]) & (k > top[None, :] - depth[None, :])
        mag = _loguniform(rng, 1e-9, 6e-3, (nz, ncol))
        tiny = rng.uniform(size=(nz, ncol)) < 0.08
        mag = np.where(tiny, _loguniform(rng, 1e-16, 1e-7, (nz, ncol)), mag)
        val = np.where(layer & present[None, :], mag, 0.0)
        fields[q] = val
        if nspec is not None:
            name, lo, hi = nspec
            nval = _loguniform(rng, lo, hi, (nz, ncol))
            zero = rng.uniform(size=(nz, ncol)) < 0.05
            fields[name] = np.where((val > 0) & ~zero, nval, 0.0)
    cu = rng.uniform(size=ncol) < 0.3
    for name in ("qrcuten", "qscuten", "qicuten"):
        val = _loguniform(rng, 1e-12, 2e-6, (nz, ncol))
        fields[name] = np.where(cu[None, :] & (rng.uniform(size=(nz, ncol))
                                               < 0.5), val, 0.0)
    surf = {name: rng.uniform(0.0, 50.0, ncol) for name in SURF}
    return ({k: np.ascontiguousarray(v, dtype=f32) for k, v in fields.items()},
            {k: np.ascontiguousarray(v, dtype=f32) for k, v in surf.items()})


def write_input(path, fields, surf, mode, dt):
    nz, ncol = fields["th"].shape
    with open(path, "wb") as out:
        np.asarray([ncol, nz, mode], dtype="<i4").tofile(out)
        np.asarray([dt], dtype="<f4").tofile(out)
        for name in LEVEL_IN:
            fields[name].astype("<f4").tofile(out)   # (nz, ncol) C = Fortran (ncol, nz)
        for name in SURF:
            surf[name].astype("<f4").tofile(out)


def read_input(path):
    raw = np.fromfile(path, dtype="<i4", count=3)
    ncol, nz, mode = (int(x) for x in raw)
    dt = float(np.fromfile(path, dtype="<f4", count=1, offset=12)[0])
    data = np.fromfile(path, dtype="<f4", offset=16)
    fields, pos = {}, 0
    for name in LEVEL_IN:
        fields[name] = data[pos:pos + nz * ncol].reshape(nz, ncol)
        pos += nz * ncol
    surf = {}
    for name in SURF:
        surf[name] = data[pos:pos + ncol]
        pos += ncol
    return ncol, nz, mode, np.float32(dt), fields, surf


def read_output(path, ncol, nz):
    data = np.fromfile(path, dtype="<f4")
    out, pos = {}, 0
    for name in LEVEL_OUT:
        out[name] = data[pos:pos + nz * ncol].reshape(nz, ncol)
        pos += nz * ncol
    for name in SURF:
        out[name] = data[pos:pos + ncol]
        pos += ncol
    return out


def _run_gpu(ncol, nz, mode, dt, fields, surf, steps, columns_per_call=None):
    import cupy as cp
    from gpuwm.core import morrison
    from gpuwm.core.refl import launch_refl10cm_morrison

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a).reshape(nz, 1, ncol))

    d = {name: dev(fields[name]) for name in LEVEL_IN}
    nc = cp.zeros((nz, 1, ncol), dtype=cp.float32)
    s = {name: cp.asarray(np.ascontiguousarray(surf[name]).reshape(1, ncol))
         for name in SURF}
    work = None
    chunked = columns_per_call is not None and columns_per_call < ncol
    t1d = cp.empty_like(d["th"]) if chunked else None
    extra = ({"columns_per_call": columns_per_call, "t1d_out": t1d}
             if chunked else {})
    for _ in range(steps):
        work = morrison.launch_morrison(
            d["th"], d["qv"], d["qc"], d["qr"], d["qi"], d["qs"], d["qg"],
            nc, d["nr"], d["ni"], d["ns"], d["ng"], d["rho"], d["pii"],
            d["p"], d["dz"], s["rainnc"], s["rainncv"], s["snownc"],
            s["snowncv"], s["graupelnc"], s["graupelncv"], s["sr"],
            float(dt), qrcuten=d["qrcuten"], qscuten=d["qscuten"],
            qicuten=d["qicuten"], morr_rimed_ice=mode, **extra)
    if chunked:
        temperature_source = "t1d_out of the column calls (WRF's t1d)"
    elif work is not None and hasattr(morrison, "WORK_T3D"):
        k = morrison.WORK_T3D
        t1d = cp.ascontiguousarray(work[k * nz:(k + 1) * nz])
        temperature_source = "process T3D (WRF's t1d)"
    else:
        # Pre-transcription API returns nothing; the closest stand-in for
        # WRF's t1d is theta * pii after the call.
        t1d = d["th"] * d["pii"]
        temperature_source = "theta*pii stand-in"
    refl = cp.empty_like(d["th"])
    launch_refl10cm_morrison(d["qv"], d["qr"], d["nr"], d["qs"], d["ns"],
                             d["qg"], d["ng"], t1d, d["p"], refl,
                             morr_rimed_ice=mode)
    out = {name: cp.asnumpy(d[name]).reshape(nz, ncol)
           for name in LEVEL_OUT if name != "refl_10cm"}
    out["refl_10cm"] = cp.asnumpy(refl).reshape(nz, ncol)
    for name in SURF:
        out[name] = cp.asnumpy(s[name]).reshape(ncol)
    return out, temperature_source


def run_gpu(ncol, nz, mode, dt, fields, surf):
    return _run_gpu(ncol, nz, mode, dt, fields, surf, 1)[0]


def compare(in_path, wrf_path, row37=True):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    if not row37:
        _row37_off()
    ncol, nz, mode, dt, fields, surf = read_input(in_path)
    want = read_output(wrf_path, ncol, nz)
    got = run_gpu(ncol, nz, mode, dt, fields, surf)
    report = {"file": str(in_path), "mode": mode, "dt": float(dt),
              "ncol": ncol, "nz": nz, "row37": row37, "fields": {}}
    total_bad = total = 0
    for name in LEVEL_OUT + SURF:
        g = np.ascontiguousarray(got[name], dtype=np.float32).ravel()
        w = np.ascontiguousarray(want[name], dtype=np.float32).ravel()
        bad = g.view(np.uint32) != w.view(np.uint32)
        # NaN payloads are not part of the claim.
        bad &= ~(np.isnan(g) & np.isnan(w))
        ulp = fp32_ulp_distance(g, w)
        cols = sorted({int(i) % ncol for i in np.nonzero(bad)[0]})
        report["fields"][name] = {
            "mismatches": int(bad.sum()), "values": int(g.size),
            "max_ulp": int(ulp[bad].max()) if bad.any() else 0,
            "columns": cols[:20]}
        total_bad += int(bad.sum())
        total += int(g.size)
    report["overall"] = {"mismatches": total_bad, "values": total,
                         "bitwise": total_bad == 0}
    return report


def export_npz(out_path, pairs, columns):
    """Keep the first ``columns`` columns of each (input, WRF output) pair."""
    arrays = {}
    for index, (in_path, wrf_path) in enumerate(pairs):
        ncol, nz, mode, dt, fields, surf = read_input(in_path)
        want = read_output(wrf_path, ncol, nz)
        keep = min(columns, ncol)
        arrays[f"s{index}_meta"] = np.asarray([keep, nz, mode], np.int32)
        arrays[f"s{index}_dt"] = np.asarray([dt], np.float32)
        for name in LEVEL_IN:
            arrays[f"s{index}_in_{name}"] = fields[name][:, :keep].copy()
        for name in SURF:
            arrays[f"s{index}_in_{name}"] = surf[name][:keep].copy()
        for name in LEVEL_OUT:
            arrays[f"s{index}_out_{name}"] = want[name][:, :keep].copy()
        for name in SURF:
            arrays[f"s{index}_out_{name}"] = want[name][:keep].copy()
    np.savez_compressed(out_path, **arrays)


def _row37_off():
    import cupy as cp
    from gpuwm.core import kernels
    original = kernels.load_module

    def load(name):
        if name != "morrison":
            return original(name)
        module = cp.RawModule(
            code="#define MORR_ROW37_OFF 1" + chr(10)
            + kernels.module_source("morrison"),
            options=kernels.module_options("morrison"))
        module.compile()
        return module

    kernels.load_module = load
    kernels.get_kernel.cache_clear()
    return original


def compare_npz(path, row37=True):
    """Grade every stress set in a committed fixture; one report per set."""
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    from gpuwm.core import kernels
    original = None if row37 else _row37_off()
    try:
        data = np.load(path)
        reports = []
        index = 0
        while f"s{index}_meta" in data:
            ncol, nz, mode = (int(x) for x in data[f"s{index}_meta"])
            dt = np.float32(data[f"s{index}_dt"][0])
            fields = {name: data[f"s{index}_in_{name}"] for name in LEVEL_IN}
            surf = {name: data[f"s{index}_in_{name}"] for name in SURF}
            got = run_gpu(ncol, nz, mode, dt, fields, surf)
            report = {"set": index, "mode": mode, "dt": float(dt),
                      "row37": row37, "fields": {}}
            bad_total = total = 0
            for name in LEVEL_OUT + SURF:
                g = np.ascontiguousarray(got[name], np.float32).ravel()
                w = np.ascontiguousarray(data[f"s{index}_out_{name}"],
                                         np.float32).ravel()
                bad = (g.view(np.uint32) != w.view(np.uint32)) & ~(
                    np.isnan(g) & np.isnan(w))
                ulp = fp32_ulp_distance(g, w)
                report["fields"][name] = {
                    "mismatches": int(bad.sum()), "values": int(g.size),
                    "max_ulp": int(ulp[bad].max()) if bad.any() else 0}
                bad_total += int(bad.sum())
                total += int(g.size)
            report["overall"] = {"mismatches": bad_total, "values": total,
                                 "bitwise": bad_total == 0}
            reports.append(report)
            index += 1
        return reports
    finally:
        if original is not None:
            kernels.load_module = original
            kernels.get_kernel.cache_clear()


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("outdir", type=Path)
    mk.add_argument("--seed", type=int, default=20261005)
    mk.add_argument("--ncol", type=int, default=4096)
    mk.add_argument("--nz", type=int, default=40)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("input", type=Path)
    cmp_.add_argument("wrf_output", type=Path)
    cmp_.add_argument("--row37-off", action="store_true")
    cmp_.add_argument("--json", type=Path)
    npz = sub.add_parser("npz")
    npz.add_argument("out", type=Path)
    npz.add_argument("files", type=Path, nargs="+")
    npz.add_argument("--columns", type=int, default=384)
    cnpz = sub.add_parser("compare-npz")
    cnpz.add_argument("fixture", type=Path)
    cnpz.add_argument("--row37-off", action="store_true")
    cnpz.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "npz":
        files = list(args.files)
        if len(files) % 2:
            parser.error("npz takes IN.bin WRF_OUT.bin pairs")
        export_npz(args.out, list(zip(files[0::2], files[1::2])),
                   args.columns)
        return 0
    if args.cmd == "compare-npz":
        reports = compare_npz(args.fixture, row37=not args.row37_off)
        if args.json:
            args.json.write_text(json.dumps(reports, indent=2))
        for report in reports:
            o = report["overall"]
            print(f"set {report['set']} mode {report['mode']} "
                  f"dt {report['dt']}: {o['mismatches']} of {o['values']} "
                  f"differ, bitwise={o['bitwise']}")
        return 0
    if args.cmd == "make":
        args.outdir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed)
        for i, (mode, dt) in enumerate(FILES):
            fields, surf = make_columns(rng, args.ncol, args.nz)
            write_input(args.outdir / f"stress_{i}.bin", fields, surf, mode,
                        dt)
        return 0
    report = compare(args.input, args.wrf_output, row37=not args.row37_off)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    for name, entry in report["fields"].items():
        if entry["mismatches"]:
            print(f"{name},{entry['max_ulp']},{entry['mismatches']},"
                  f"{entry['values']},cols={entry['columns'][:8]}")
    o = report["overall"]
    print(f"overall,{o['mismatches']},{o['values']},bitwise={o['bitwise']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
