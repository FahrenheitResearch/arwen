"""TEMPO column oracle, adversarial WOOF side (refute pass).

Same comparison as run_woof_oracle.py (every output of every step, bitwise,
free-running and per-step replay), with the engine conditions the lane's
runner never exercised:

  --nz-max engine   compile with TEMPO_NZ_MAX = nz of each set, which is what
                    gpuwm.core.tempo._runtime does (the lane used 64 for all)
  --batch N         columns per kernel batch (the engine batches by
                    GPUWM_TEMPO_WORK_BYTES and reuses one work buffer across
                    batches; the lane's sets always fit one batch)
  --poison P        fill the per-column work area and column state with a bit
                    pattern before every batch (nan, big, negbig, denorm):
                    any read of a work value the kernel did not write this
                    call shows up as a difference
  --refl10cm 0      the engine adapter's configuration on non-output steps
  --skip F,...      outputs not compared (diagnostics a flag switched off)

Reuses the lane's readers, ULP distance and frozen_fraction rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_woof_oracle as rw  # noqa: E402

POISON = {"nan": 0x7FF8DEADBEEF0001, "big": 0x7FE1CCF385EBC8A0,     # ~1e308
          "negbig": 0xFFE1CCF385EBC8A0, "denorm": 0x0000000000000001}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("columns")
    ap.add_argument("catalog")
    ap.add_argument("reference")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tables")
    src.add_argument("--blob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="both", choices=("free", "replay", "both"))
    ap.add_argument("--nz-max", default="engine")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--poison", default="none", choices=["none"] + sorted(POISON))
    ap.add_argument("--refl10cm", type=int, default=1)
    ap.add_argument("--skip", default="")
    args = ap.parse_args(argv)

    import cupy as cp
    import gpuwm  # noqa: F401
    from gpuwm import wrf_exact
    from gpuwm.core.tempo_runtime import TempoConfig, TempoRuntime

    sets = rw.read_columns(args.columns)
    rw.read_reference(args.reference, sets)
    catalog = json.loads(Path(args.catalog).read_text())
    variant_of = {(r["set"], r["column"]): r["variant"] for r in catalog}
    skip = {s for s in args.skip.split(",") if s}
    cfg = TempoConfig(refl10cm=bool(args.refl10cm))

    def make_runtime(nz):
        nz_max = nz if args.nz_max == "engine" else int(args.nz_max)
        kw = dict(nz_max=nz_max)
        rt = (TempoRuntime(args.blob, **kw) if args.blob
              else TempoRuntime(table_dir=args.tables, **kw))
        if args.batch:
            rt.batch = args.batch
        if args.poison != "none":
            pat = np.uint64(POISON[args.poison])
            orig = rt._buffers

            def poisoned(n, orig=orig):
                cols, work = orig(n)
                work.view(cp.uint64).fill(pat)
                cols.view(cp.uint64).fill(pat)
                return cols, work
            rt._buffers = poisoned
        return rt

    result = dict(strict=os.environ.get("GPUWM_WRF_EXACT") == "1",
                  wrf_exact_enabled=bool(wrf_exact.ENABLED),
                  nz_max=args.nz_max, batch=args.batch, poison=args.poison,
                  refl10cm=args.refl10cm, skip=sorted(skip),
                  blob_source=("fortran dump " + str(args.blob)) if args.blob else
                  ("rust tempo_init from " + str(args.tables)),
                  device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                  sets=[dict(dt=s["dt"], ncol=s["ncol"], nz=s["nz"], nsteps=s["nsteps"])
                        for s in sets], modes={})
    modes = ["free", "replay"] if args.mode == "both" else [args.mode]
    for mode in modes:
        t0 = time.time()
        per_field = rw.Stats()
        undefined = 0
        miss_columns, first_miss, options = {}, None, set()
        # one runtime per set nz (as the engine does: one module per nz);
        # first_call follows the Fortran SAVE flag across the whole run
        runtimes = {}
        first_call = True
        for si, s in enumerate(sets, start=1):
            ncol, nz = s["ncol"], s["nz"]
            key = nz if args.nz_max == "engine" else "fixed"
            if key not in runtimes:
                runtimes[key] = make_runtime(nz)
            rt = runtimes[key]
            options.add(tuple(getattr(rt.module, "options", ()) or ()))
            f = {n: cp.ascontiguousarray(cp.asarray(s["fields"][n].T[:, None, :]))
                 for n in rw.STATE_IN}
            for it in range(s["nsteps"]):
                if mode == "replay" and it > 0:
                    prev = s["ref"][it - 1]
                    for n in rw.UPDATED:
                        f[n][...] = cp.asarray(prev[n].T[:, None, :])
                rt.first_call = first_call
                out = rt.step(f, s["dt"], cfg)
                first_call = False
                ref = s["ref"][it]
                got = {n: f[n][:, 0, :].get().T for n in rw.UPDATED}
                for n in rw.SURFACE:
                    got[n] = out["surface"][n][0, :].get().reshape(ncol, 1)
                for n in rw.COLUMN:
                    got[n] = out["column"][n][:, 0, :].get().T
                for n in rw.OUTPUTS:
                    if n in skip:
                        continue
                    if n == "frozen_fraction":
                        exc = rw.undefined_frozen_fraction(ref, got[n])
                        if exc.any():
                            undefined += int(exc.sum())
                            got[n] = np.where(exc, ref[n], got[n])
                    miss = per_field.add(n, got[n], ref[n])
                    if miss is None:
                        continue
                    cols = np.nonzero(miss.any(axis=1))[0]
                    for c in cols:
                        miss_columns.setdefault(f"{si}:{c + 1}:{variant_of[(si, c + 1)]}",
                                                set()).add(n)
                    if first_miss is None:
                        c = int(cols[0])
                        k = int(np.nonzero(miss[c])[0][0])
                        first_miss = dict(set=si, step=it + 1, column=c + 1,
                                          variant=variant_of[(si, c + 1)], field=n,
                                          level=k + 1, got=float(got[n][c, k]),
                                          want=float(ref[n][c, k]),
                                          got_hex=float(got[n][c, k]).hex(),
                                          want_hex=float(ref[n][c, k]).hex())
        fields = per_field.table()
        total = sum(v["n"] for v in fields.values())
        bitwise = sum(v["n_bitwise"] for v in fields.values())
        result["modes"][mode] = dict(
            values=total, bitwise_equal=bitwise, mismatched=total - bitwise,
            max_ulp=max(v["max_ulp"] for v in fields.values()),
            max_abs=max(v["max_abs"] for v in fields.values()),
            max_rel=max(v["max_rel"] for v in fields.values()),
            nan_mismatch=sum(v["nan_mismatch"] for v in fields.values()),
            per_field=fields, columns_with_misses=len(miss_columns),
            columns_with_misses_list={k: sorted(v) for k, v in sorted(miss_columns.items())},
            first_miss=first_miss, seconds=round(time.time() - t0, 1),
            tempo_undefined_frozen_fraction_cells=undefined,
            compile_options=sorted(list(o) for o in options))
        m = result["modes"][mode]
        print(f"[{mode}] strict={result['strict']} poison={args.poison} batch={args.batch} "
              f"nz_max={args.nz_max} refl10cm={args.refl10cm} values={total} bitwise={bitwise} "
              f"mismatched={total - bitwise} max_ulp={m['max_ulp']} max_abs={m['max_abs']:.3e} "
              f"max_rel={m['max_rel']:.3e} columns_with_misses={len(miss_columns)} "
              f"tempo_undefined_frozen_fraction={undefined}")
        if first_miss:
            print(f"[{mode}] first miss: {first_miss}")
    rec = wrf_exact.compile_receipt()
    result["compile_receipt"] = dict(
        enabled=rec["enabled"], strict_options=rec["strict_options"],
        nvrtc_compiles=[r for r in rec["records"] if r["kind"] == "nvrtc_compile"])
    Path(args.out).write_text(json.dumps(result, indent=1, default=str) + "\n")
    worst = max(m["mismatched"] for m in result["modes"].values())
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
