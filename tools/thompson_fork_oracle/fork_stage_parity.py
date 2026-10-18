"""Locate where WOOF's fork Thompson first departs from the fork's Fortran:
process rates and stage states, bitwise, on the host.

usage:
    python fork_stage_parity.py BUILD_DIR RATES_DIR OUT_DIR [--batch SET:DT]

BUILD_DIR is build.sh's pristine oracle, RATES_DIR build_rates.sh's
instrumented one.  WOOF's kernels run through tools/
thompson_real_column_parity/host_backend.py (g++, no FMA, the host libm)
with port_rates.py's readbacks compiled in.  The instrumented Fortran must
return the pristine bytes, and the instrumented WOOF build the pristine
state, or the run is refused.

Per rate (cp1, or cp2 for the four late rates): the cells the fork wrote a
row for, how many differ in their float64 bits, and the worst relative
difference.  Per stage: WOOF's working state after its source networks
against the fork's ``X1d + Xten*DT`` at cp1, and after rain evaporation
against cp2, every species and the temperature, in float32 bits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE), str(HERE.parent / "thompson_real_column_parity")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import real_column_parity as R  # noqa: E402  (installs the host backend)
import instrument_fork_rates as IF  # noqa: E402
import gpu_column_oracle as G  # noqa: E402

f32 = np.float32


def read_streams(run_dir, ncol, nz):
    out = {}
    for cp, fields in IF.SCHEMA.items():
        raw = np.fromfile(Path(run_dir) / f"wrf-{cp}.bin", dtype="<f8")
        nf = len(fields)
        rows = raw.reshape(-1, nf)
        col = rows[:, 0].astype(np.int64) - 1
        lev = rows[:, 1].astype(np.int64) - 1
        keep = col < ncol          # the pad column is not read back
        table = {}
        for j, name in enumerate(fields[2:], start=2):
            a = np.full((ncol, nz), np.nan)
            a[col[keep], lev[keep]] = rows[keep, j]
            table[name] = a
        out[cp] = table
    return out


def run_fork_binary(binary, cwd, inp, dt):
    in_path, out_path = Path(cwd) / "stage-in.bin", Path(cwd) / "stage-out.bin"
    G.write_fork_input(in_path, inp, dt)
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    done = subprocess.run([str(binary), str(in_path), str(out_path)],
                          cwd=str(cwd), env=env, capture_output=True,
                          text=True, check=False)
    if done.returncode:
        raise RuntimeError(done.stdout + done.stderr)
    data = out_path.read_bytes()
    in_path.unlink()
    out_path.unlink()
    return data


def bits_differ(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    both = np.isnan(a) & np.isnan(b)
    return (a != b) & ~both


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("build")
    ap.add_argument("rates")
    ap.add_argument("out")
    ap.add_argument("--batch", default="all:20")
    ap.add_argument("--per-regime", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args(argv)

    set_name, dt = args.batch.split(":")
    dt = float(dt)
    cols, labels = G.batch_columns(set_name, args.per_regime, args.seed)
    inp = R.prepare(cols, 28)
    ncol, nz = inp["p"].shape
    build, rates_dir = Path(args.build).resolve(), Path(args.rates).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pristine = run_fork_binary(build / "run_columns_fork", build, inp, dt)
    instrumented = run_fork_binary(rates_dir / "run_columns_fork", rates_dir,
                                   inp, dt)
    if pristine != instrumented:
        raise RuntimeError("instrumented fork changed its outputs")
    cps = read_streams(rates_dir, ncol, nz)

    kw = dict(mp=28, thompson_version="wrf_39_noaa",
              thompson_fork_snow_fall="wrf_39_noaa")
    port = R.run_port(inp, dt, rates=False, **kw)
    # run_port binds the rate buffer to each module's plain host build; the
    # fork's kernels are that module's THOMPSON_AA_WRF39 build, so the
    # binding is pointed there for the instrumented run.
    import host_backend
    from gpuwm.core.thompson_aerosol_launch import WRF39_DEFINES
    plain = host_backend.build_module
    host_backend.build_module = (
        lambda name: host_backend._module_int_cached(name, WRF39_DEFINES))
    try:
        port_r = R.run_port(inp, dt, rates=True, **kw)
    finally:
        host_backend.build_module = plain
    for k in port["final"]:
        if bits_differ(port["final"][k], port_r["final"][k]).any():
            raise RuntimeError(f"instrumented WOOF changed {k}")

    res = {"batch": f"{set_name}@dt{dt:g}", "ncol": ncol, "nz": nz,
           "rates": {}, "stages": {}}
    first = {}
    for name in IF.RATES:
        cp = "cp2" if name in IF.LATE_RATES else "cp1"
        fork_v = cps[cp][name]
        have = np.isfinite(fork_v)
        woof_v = port_r["rates"][name]
        d = bits_differ(woof_v, fork_v) & have
        entry = {"cells": int(have.sum()), "fork_nonzero":
                 int((have & (fork_v != 0)).sum()), "differ": int(d.sum())}
        if d.any():
            scale = np.maximum(np.abs(woof_v), np.abs(fork_v))
            rel = np.where(d, np.abs(woof_v - fork_v)
                           / np.where(scale > 0, scale, 1), 0)
            w = np.unravel_index(int(np.argmax(rel)), rel.shape)
            entry.update(max_rel=float(rel.max()), at=[int(i) for i in w],
                         label=labels[w[0]], woof=float(woof_v[w]),
                         fork=float(fork_v[w]),
                         zero_vs_nonzero=int((d & ((woof_v == 0) |
                                                   (fork_v == 0))).sum()))
            for c, k in zip(*np.nonzero(d)):
                first.setdefault((int(c), int(k)), name)
        res["rates"][name] = entry

    # Stage states: X1d + Xten*DT in float32, as the fork forms them.
    def fork_stage(cp, var):
        c1 = cps["cp1"]
        ten = {"T": "tten"}.get(var, var + "ten")
        one = {"T": "t1d"}.get(var, var + "1d")
        x1d = c1[one].astype(f32)
        xt = cps[cp][ten].astype(f32)
        v = (x1d + xt * f32(dt)).astype(f32)
        if var == "qv":
            v = np.maximum(v, f32(1.0e-10))
        return np.where(np.isnan(c1[one]) | np.isnan(cps[cp][ten]),
                        np.nan, v)

    for stage, cp in (("sources", "cp1"), ("rain_evaporation", "cp2")):
        snap = port_r["stages"].get(stage)
        if snap is None:
            res["stages"][stage] = "missing"
            continue
        st = {}
        for var in ("T", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr",
                    "nc", "nwfa", "nifa"):
            if var not in snap:
                continue
            fv = fork_stage(cp, var)
            have = np.isfinite(fv)
            wv = np.asarray(snap[var], np.float64).astype(f32)
            d = (wv != fv.astype(f32)) & have
            e = {"cells": int(have.sum()), "differ": int(d.sum())}
            if d.any():
                ulp = np.abs(G.ordered(wv) - G.ordered(fv.astype(f32)))
                ulp = np.where(d, ulp, 0)
                w = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
                e.update(max_ulp=int(ulp.max()), at=[int(i) for i in w],
                         label=labels[w[0]], woof=float(wv[w]),
                         fork=float(fv[w]))
            st[var] = e
        res["stages"][stage] = st

    tally = {}
    for name in first.values():
        tally[name] = tally.get(name, 0) + 1
    res["first_differing_rate_in_cp1_order"] = dict(
        sorted(tally.items(), key=lambda kv: -kv[1]))
    (out_dir / "stage-parity.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        out_dir / f"stage-{set_name}-dt{dt:g}.npz",
        labels=np.array(labels),
        **{f"fork_{cp}_{k}": v for cp, t in cps.items() for k, v in t.items()},
        **{f"woof_rate_{k}": v for k, v in port_r["rates"].items()},
        **{f"woof_{st}_{k}": np.asarray(v) for st, s in
           port_r["stages"].items() for k, v in s.items()},
        **{f"in_{k}": v for k, v in inp.items()})
    short = {k: v["differ"] for k, v in res["rates"].items() if v["differ"]}
    print(json.dumps({"rates_differing": short,
                      "stages": {s: {k: v.get("differ") for k, v in
                                     t.items()} if isinstance(t, dict) else t
                                 for s, t in res["stages"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
