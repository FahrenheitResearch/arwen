"""Grade WOOF's gsl_wrf39 MYNN surface-layer kernel against the fork oracle.

    python compare.py DECK.txt ORACLE.csv[.gz] [--json OUT] [--names]

Runs the production CUDA kernel through ``launch_mynn_surface_layer``
(variant "gsl_wrf39") over the deck's columns for the oracle's steps,
carrying UST, USTM, MOL, QSFC, ZNT, HFX and QFX between steps exactly as
run_columns.F90 does, and compares every output word against the
oracle's binary32 bit patterns.  The carried inputs are compared too, so a
drift in the carry is reported where it enters rather than one step later.

Whether the kernel compiled under the strict build is a property of the
process: set GPUWM_WRF_EXACT=1 before Python starts.  The report records
which build ran.  Exit status 0 means every word of every output matched.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsl_wrf39_columns import load_deck  # noqa: E402

OUTPUTS = ("regime", "zol", "rmol", "ust", "ustm", "mol", "psim", "psih",
           "chs", "chs2", "cqs2", "ch", "flhc", "flqc", "qgh", "qsfc", "hfx",
           "qfx", "lh", "u10", "v10", "th2", "t2", "q2", "gz1oz0", "wspd",
           "br", "ck", "cka", "cd", "cda", "wstar", "qstar", "cpm", "znt")
CARRIED = ("hfx", "qfx", "znt", "qsfc", "ust", "mol", "ustm")


def load_oracle(path) -> dict[int, dict[str, np.ndarray]]:
    """{step: {field: uint32 bit patterns}} from the hex CSV."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="ascii") as fh:
        header = fh.readline().strip().split(",")
        rows = [line.strip().split(",") for line in fh if line.strip()]
    steps: dict[int, dict[str, list]] = {}
    for row in rows:
        step = steps.setdefault(int(row[0]), {})
        for key, value in zip(header[2:], row[2:]):
            step.setdefault(key, []).append(int(value, 16))
    return {s: {k: np.asarray(v, dtype=np.uint32) for k, v in d.items()}
            for s, d in steps.items()}


def run_kernel(deck, nstep):
    """{step: {field: uint32}} from the production launcher."""
    import cupy as cp
    from gpuwm.core.mynn_sfclay import (
        MYNN_SURFACE_INPUTS, _allocate_result, launch_mynn_surface_layer)
    _, isftcflx, dx, cols = load_deck(deck)
    n = cols["u1"].size
    shape = (1, n)
    dev = {k: cp.asarray(cols[k].reshape(shape)) for k in MYNN_SURFACE_INPUTS}
    mol = cp.asarray(cols["mol"].reshape(shape))
    ustm = cp.asarray(cols["ust"].reshape(shape))
    steps = {}
    for step in range(1, nstep + 1):
        out = {f"{k}_input": cp.asnumpy(v).reshape(n).view(np.uint32)
               for k, v in (("hfx", dev["hfx"]), ("qfx", dev["qfx"]),
                            ("znt", dev["znt"]), ("qsfc", dev["qsfc"]),
                            ("ust", dev["ust"]), ("mol", mol),
                            ("ustm", ustm))}
        result = _allocate_result(shape)
        launch_mynn_surface_layer(dev, mol, ustm, result, dx=dx,
                                  itimestep=step, isfflx=1,
                                  isftcflx=isftcflx, variant="gsl_wrf39")
        for k in OUTPUTS:
            out[k] = cp.asnumpy(getattr(result, k)).reshape(n).view(np.uint32)
        steps[step] = out
        for name in ("hfx", "qfx", "znt", "qsfc", "ust"):
            dev[name] = cp.ascontiguousarray(getattr(result, name))
        mol = cp.ascontiguousarray(result.mol)
        ustm = cp.ascontiguousarray(result.ustm)
    return steps


def _ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """FP32 ULP distance on the bit patterns; NaN against non-NaN is huge."""
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    return np.abs(monotone_fp32_key(a.view(np.float32))
                  - monotone_fp32_key(b.view(np.float32)))


def grade(got, want, names=None) -> dict:
    fields = [f"{k}_input" for k in CARRIED] + list(OUTPUTS)
    table, worst_cols = {}, {}
    for name in fields:
        d = np.concatenate([_ulp(got[s][name], want[s][name])
                            for s in sorted(want)])
        absd = np.concatenate([
            np.abs(got[s][name].view(np.float32).astype(np.float64)
                   - want[s][name].view(np.float32).astype(np.float64))
            for s in sorted(want)])
        table[name] = {"max_ulp": int(d.max()), "mismatched": int((d > 0).sum()),
                       "n": int(d.size),
                       "max_abs": float(np.nanmax(absd)) if absd.size else 0.0}
        if d.max() > 0:
            ncol = got[min(want)][name].size
            idx = int(np.argmax(d))
            worst_cols[name] = {
                "step": sorted(want)[idx // ncol], "column": idx % ncol,
                "name": names[idx % ncol] if names else None,
                "ulp": int(d[idx])}
    first = None
    for s in sorted(want):
        for name in fields:
            bad = np.nonzero(got[s][name] != want[s][name])[0]
            if bad.size:
                first = {"step": s, "field": name, "columns": bad[:12].tolist()}
                break
        if first:
            break
    return {"fields": table, "worst": worst_cols, "first_mismatch": first,
            "words": int(sum(v["n"] for v in table.values())),
            "mismatched_words": int(sum(v["mismatched"]
                                        for v in table.values())),
            "max_ulp": int(max(v["max_ulp"] for v in table.values()))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("oracle")
    ap.add_argument("--json")
    ap.add_argument("--deck-name")
    args = ap.parse_args(argv)
    from gpuwm import wrf_exact
    oracle = load_oracle(args.oracle)
    nstep = max(oracle)
    names = None
    if args.deck_name:
        from gsl_wrf39_columns import deck_names
        names = deck_names(args.deck_name)
    got = run_kernel(args.deck, nstep)
    report = grade(got, oracle, names)
    import cupy as cp
    dev = cp.cuda.Device()
    report.update({
        "deck": str(args.deck), "oracle": str(args.oracle),
        "columns": int(next(iter(oracle[1].values())).size), "steps": nstep,
        "strict_build": bool(wrf_exact.ENABLED),
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "device": cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode(),
        "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
    })
    print(f"{Path(args.deck).stem}: strict={report['strict_build']} "
          f"columns={report['columns']} steps={nstep} "
          f"words={report['words']} mismatched={report['mismatched_words']} "
          f"max_ulp={report['max_ulp']}")
    for k, v in report["fields"].items():
        if v["max_ulp"]:
            print(f"  {k}: max_ulp={v['max_ulp']} mismatched={v['mismatched']}"
                  f"/{v['n']} max_abs={v['max_abs']:.3g} "
                  f"worst={report['worst'].get(k)}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1),
                                   encoding="utf-8")
    return 0 if report["mismatched_words"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
