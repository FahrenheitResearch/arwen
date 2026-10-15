#!/usr/bin/env python3
"""Run the port on the wide YSU oracle and print every field's distance.

    python tools/ysu_wrf461_oracle/compare_wide.py [--reference stock|O0]
        [--json OUT.json] [DIR]

Run once with GPUWM_WRF_EXACT=1 (the strict build: no FMA, no FTZ, IEEE
division and square root) and once without (default arithmetic).  Prints a
per-field table over every column where WRF's own output is finite, a
per-regime table, every kpbl disagreement, and the columns WRF fills with
NaN together with what the port writes there.  Asserts nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpuwm.verify.ysu_wide_oracle import (  # noqa: E402
    REGIMES, compare_wide, load_wide_oracle, run_port_wide)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--reference", default="stock", choices=("stock", "O0"))
    args = ap.parse_args(argv)

    import cupy as cp
    from gpuwm import wrf_exact

    cols = load_wide_oracle(args.directory, reference=args.reference)
    port = run_port_wide(cols)
    finite = [c for c in cols if c.wrf_finite]
    nan_cols = [c for c in cols if not c.wrf_finite]
    overall = compare_wide(cols, port)
    mode = "strict (GPUWM_WRF_EXACT=1)" if wrf_exact.ENABLED else "default"
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) \
        else props["name"]
    nvrtc = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
    print(f"reference: WRF bl_ysu_run, {args.reference} build")
    print(f"arithmetic: {mode}   device: {name} sm_{props['major']}"
          f"{props['minor']}   NVRTC {nvrtc}   cupy {cp.__version__}")
    print(f"columns: {len(cols)} total, {len(finite)} where WRF is finite,"
          f" {len(nan_cols)} where WRF writes NaN/Inf")
    print(f"{'field':<8} {'max_ulp':>12} {'max_abs':>12} {'differing':>14}")
    report = {"reference": args.reference, "mode": mode, "device": name, "nvrtc": nvrtc,
              "cupy": cp.__version__, "ncol": len(cols),
              "ncol_wrf_finite": len(finite), "fields": {}, "regimes": {}}
    for fname, st in overall["fields"].items():
        print(f"{fname:<8} {st.max_ulp:>12d} {st.max_abs:>12.4g} "
              f"{st.differing:>7d}/{st.total:<7d} worst={st.worst}")
        report["fields"][fname] = {
            "max_ulp": st.max_ulp, "max_abs": st.max_abs,
            "differing": st.differing, "total": st.total,
            "worst": st.worst}
    print(f"kpbl mismatches: {overall['kpbl_mismatch']}")
    report["kpbl_mismatch"] = overall["kpbl_mismatch"]

    print("\nper regime (max ULP over all fields; columns):")
    for rid, rname in REGIMES.items():
        sel = compare_wide(cols, port,
                           include=lambda c, r=rid: c.regime == r
                           and c.wrf_finite)
        n = sum(1 for c in cols if c.regime == rid and c.wrf_finite)
        if not n:
            continue
        m = max((s.max_ulp for s in sel["fields"].values()), default=0)
        dif = sum(s.differing for s in sel["fields"].values())
        print(f"  {rname:<24} cols={n:<3d} max_ulp={m:<10d} differing_words="
              f"{dif:<6d} kpbl_mismatch={len(sel['kpbl_mismatch'])}")
        report["regimes"][rname] = {"columns": n, "max_ulp": m,
                                    "differing_words": dif,
                                    "kpbl_mismatch": len(sel["kpbl_mismatch"])}

    worst_cols = sorted(overall["per_column"].items(),
                        key=lambda kv: -max(kv[1].values()))[:12]
    print("\nworst columns:")
    for idx, d in worst_cols:
        if max(d.values()) == 0:
            break
        c = cols[idx]
        nz = {k: v for k, v in d.items() if v}
        print(f"  col {idx:3d} {c.regime_name:<24} nz={c.nz} topo={c.topo} "
              f"topdown={c.topdown} {nz}")

    print("\ncolumns where WRF's output is non-finite (WRF undefined):")
    report["wrf_nonfinite"] = []
    for c in nan_cols:
        got = port[c.index]
        bad = [k for k, v in c.wrf.items() if k != "qmixtnp"
               and not np.all(np.isfinite(v))]
        port_finite = all(np.all(np.isfinite(got[k])) for k in
                          ("du", "dv", "dtheta", "dqv", "exch_h", "exch_m"))
        s = c.scalars
        print(f"  col {c.index:3d} ust={float(s['ust']):.3g} "
              f"hfx={float(s['hfx']):.3g} br={float(s['br']):.3g}: WRF "
              f"non-finite in {bad}; port finite={port_finite}, kpbl "
              f"{int(got['kpbl'][0])} vs WRF {c.wrf_kpbl}")
        report["wrf_nonfinite"].append({
            "column": c.index, "ust": float(s["ust"]), "hfx": float(s["hfx"]),
            "qfx": float(s["qfx"]), "br": float(s["br"]),
            "wrf_nonfinite_fields": bad, "port_finite": port_finite})

    if wrf_exact.ENABLED:
        rec = wrf_exact.compile_receipt()
        report["strict_options"] = rec["strict_options"]
        report["compiles"] = len(rec["records"])
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2,
                                              default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
