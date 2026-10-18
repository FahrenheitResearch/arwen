"""Grade the UW PBL port against oracle fixtures, every output word.

    python compare.py --engine gpu FIXTURE_DIR [FIXTURE_DIR ...] --json OUT.json
    python compare.py --engine cpu FIXTURE_DIR ... --json OUT.json [--workers N]

``gpu`` drives the PRODUCT launcher ``gpuwm.core.uwpbl.uwpbl_step`` (the
call ``PhysicsDriver._run_uwpbl`` makes, through
tests/test_uwpbl_launcher_wrf471_parity.py's ``_launch``) on every step of
every column of every fixture, with that step's exact recorded inputs, and
compares every float32 output word and the int32 KPBL for bitwise
equality.  Strict arithmetic is selected by the caller's environment
(GPUWM_WRF_EXACT=1 before Python starts); the JSON records which.

``cpu`` grades the binary64 CPU reference ``gpuwm.verify.uwpbl_ref`` (pure
Python, the host libm's cos/acos) the same way, unflushed.

For every field: mismatched words, max ULP distance (float32 values mapped
to a monotone integer line, so +0 and -0 are 0 ULP apart but still counted
as a mismatched word) and max absolute difference.  No tolerance enters the
verdict: ``exact`` is true only when every word of every field is equal.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from gpuwm.verify.uwpbl_oracle import (FULL_OUTPUTS, MASS_OUTPUTS,
                                       SURFACE_OUTPUTS, dims, load,
                                       step_arrays)

FIELDS = MASS_OUTPUTS + FULL_OUTPUTS + SURFACE_OUTPUTS


def _ordered(a):
    i = a.view(np.int32).astype(np.int64)
    mag = i & 0x7FFFFFFF
    return np.where(i < 0, -mag, mag)


def grade_field(got, want):
    got, want = np.asarray(got), np.asarray(want)
    if want.dtype == np.int32:
        diff = got.astype(np.int64) - want.astype(np.int64)
        bad = diff != 0
        return {"words": int(want.size), "mismatched": int(bad.sum()),
                "max_ulp": None,
                "max_abs": float(np.abs(diff).max()) if want.size else 0.0}, bad
    g = got.astype(np.float32)
    w = want.astype(np.float32)
    bad = g.view(np.uint32) != w.view(np.uint32)
    nan = np.isnan(g) | np.isnan(w)
    ulp = np.abs(_ordered(g) - _ordered(w))
    absd = np.abs(g.astype(np.float64) - w.astype(np.float64))
    finite = ~nan
    return {"words": int(w.size), "mismatched": int(bad.sum()),
            "nan_words": int(nan.sum()),
            "max_ulp": int(ulp[finite].max()) if finite.any() else 0,
            "max_abs": float(absd[finite].max()) if finite.any() else 0.0,
            }, bad


def _merge(acc, rec):
    acc["words"] += rec["words"]
    acc["mismatched"] += rec["mismatched"]
    acc["nan_words"] = acc.get("nan_words", 0) + rec.get("nan_words", 0)
    if rec["max_ulp"] is not None:
        acc["max_ulp"] = max(acc.get("max_ulp") or 0, rec["max_ulp"])
    acc["max_abs"] = max(acc["max_abs"], rec["max_abs"])


def _cpu_step(args):
    path, step = args
    from gpuwm.verify.uwpbl_ref.driver import camuwpbl_step
    fx = load(path)
    want = step_arrays(fx, step)
    return path, step, camuwpbl_step(want, ftz=False)


def _stems(dirs):
    out = []
    for d in dirs:
        for man in sorted(Path(d).glob("cases-*.manifest")):
            if man.stem.endswith("-spy"):
                continue
            out.append(man.with_suffix(""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--engine", choices=("gpu", "cpu"), required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma list of stems")
    args = ap.parse_args(argv)

    stems = _stems(args.dirs)
    if args.only:
        keep = set(args.only.split(","))
        stems = [s for s in stems if s.name in keep]
    started = time.time()
    meta = {"engine": args.engine, "host": platform.node(),
            "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
            "python": sys.version.split()[0], "numpy": np.__version__}
    results = {}
    cpu_out = {}
    if args.engine == "gpu":
        import cupy as cp
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
        from test_uwpbl_launcher_wrf471_parity import _launch
        from gpuwm import wrf_exact
        dev = cp.cuda.Device()
        meta["device"] = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
        meta["nvrtc"] = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
        meta["cupy"] = cp.__version__
    else:
        from concurrent.futures import ProcessPoolExecutor
        jobs = []
        for s in stems:
            _, _, nsteps = dims(load(s))
            jobs += [(str(s), k) for k in range(1, nsteps + 1)]
        with ProcessPoolExecutor(args.workers) as pool:
            for path, step, got in pool.map(_cpu_step, jobs, chunksize=1):
                cpu_out[(path, step)] = got

    for s in stems:
        fx = load(s)
        ncol, nk, nsteps = dims(fx)
        per_field = {f: {"words": 0, "mismatched": 0, "max_abs": 0.0}
                     for f in FIELDS}
        bad_cols = []
        for step in range(1, nsteps + 1):
            want = step_arrays(fx, step)
            if args.engine == "gpu":
                got = _launch(want, nk, ncol)
            else:
                got = cpu_out[(str(s), step)]
            colbad = np.zeros(ncol, bool)
            for f in FIELDS:
                rec, bad = grade_field(got[f], want[f])
                _merge(per_field[f], rec)
                colbad |= bad.reshape(ncol, -1).any(axis=1)
            bad_cols += [[int(c) + 1, step] for c in np.flatnonzero(colbad)]
        words = sum(v["words"] for v in per_field.values())
        mism = sum(v["mismatched"] for v in per_field.values())
        results[s.name] = {"dir": str(s.parent), "ncol": ncol, "nk": nk,
                           "nsteps": nsteps, "words": words,
                           "mismatched_words": mism,
                           "mismatched_column_steps_1based": bad_cols,
                           "fields": per_field}
        print(f"{s.parent.name}/{s.name}: {ncol} columns x {nsteps} steps, "
              f"{words} words, {mism} mismatched, "
              f"{len(bad_cols)} column-steps differ", flush=True)
    # Negative control: the same engine on the first fixture's step 1 with
    # ONE input word changed (the last bit of t at level 1 of column 1;
    # th is not read by camuwpbl, so it would be a vacuous probe)
    # must disagree with the recorded outputs; a grader that cannot see a
    # one-bit input change proves nothing.
    s = stems[0]
    fx = load(s)
    ncol, nk, _ = dims(fx)
    want = step_arrays(fx, 1)
    poked = dict(want)
    poked["t"] = want["t"].copy()
    poked["t"].view(np.uint32)[0, 0] ^= 1
    if args.engine == "gpu":
        got = _launch(poked, nk, ncol)
    else:
        from gpuwm.verify.uwpbl_ref.driver import camuwpbl_step
        got = camuwpbl_step(poked, ftz=False)
    seen = sum(grade_field(got[f], want[f])[0]["mismatched"] for f in FIELDS)
    negative = {"stem": s.name, "poked": "s1 t column 1 level 1, last bit",
                "mismatched_words": int(seen), "detected": seen > 0}
    print(f"negative control: {seen} output words changed", flush=True)

    tot_words = sum(r["words"] for r in results.values())
    tot_bad = sum(r["mismatched_words"] for r in results.values())
    max_ulp = max([f.get("max_ulp") or 0 for r in results.values()
                   for f in r["fields"].values()] or [0])
    max_abs = max([f["max_abs"] for r in results.values()
                   for f in r["fields"].values() if f.get("max_ulp") is not None]
                  or [0.0])
    summary = {"stems": len(results),
               "columns": sum(r["ncol"] for r in results.values()),
               "column_steps": sum(r["ncol"] * r["nsteps"] for r in results.values()),
               "words": tot_words, "mismatched_words": tot_bad,
               "max_ulp": max_ulp, "max_abs": max_abs,
               "exact": tot_bad == 0 and negative["detected"],
               "negative_control": negative,
               "seconds": round(time.time() - started, 1)}
    if args.engine == "gpu":
        meta["compile_receipt"] = wrf_exact.compile_receipt()
    Path(args.json).write_text(json.dumps(
        {"meta": meta, "summary": summary, "results": results}, indent=1)
        + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0 if summary["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
