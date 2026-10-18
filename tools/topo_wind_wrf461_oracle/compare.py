#!/usr/bin/env python3
"""Run WOOF's topo_wind chain on the oracle and print every field's distance.

    python tools/topo_wind_wrf461_oracle/compare.py [--reference wrfflags|O0]
        [--json OUT.json] [DIR]

Run once with GPUWM_WRF_EXACT=1 (the strict build) and once without
(default arithmetic).  Prints the static planes per option, every YSU
output over the calls where WRF's output is finite, a per-option, per-family
and per-regime table, kpbl disagreements, and the calls WRF fills with
NaN.  Asserts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpuwm.verify.topo_wind_oracle import (  # noqa: E402
    FAMILIES, OPTIONS, REFERENCES, REGIMES, compare, load_topo_wind_oracle,
    nonfinite_calls, run_port, wrf_undefined_shallow)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--reference", default="wrfflags",
                    choices=tuple(REFERENCES))
    args = ap.parse_args(argv)

    import cupy as cp
    from gpuwm import wrf_exact

    patches = load_topo_wind_oracle(args.directory, reference=args.reference)
    port = run_port(patches)
    res = compare(patches, port)
    mode = "strict (GPUWM_WRF_EXACT=1)" if wrf_exact.ENABLED else "default"
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) \
        else props["name"]
    nvrtc = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
    ncell = sum(len(p.cells) for p in patches)
    nan = nonfinite_calls(patches)
    print(f"reference: WRF v4.6.1 start_em.F block + bl_ysu_run, "
          f"{args.reference} build")
    print(f"arithmetic: {mode}   device: {name} sm_{props['major']}"
          f"{props['minor']}   NVRTC {nvrtc}   cupy {cp.__version__}")
    print(f"patches: {len(patches)}   cells (columns): {ncell}   YSU calls: "
          f"{ncell * len(OPTIONS)} ({len(nan)} where WRF writes NaN/Inf)")
    report = {"reference": args.reference, "mode": mode, "device": name,
              "nvrtc": nvrtc, "cupy": cp.__version__,
              "npatch": len(patches), "ncell": ncell,
              "ncall": ncell * len(OPTIONS), "nonfinite_calls": nan,
              "static": {}, "ysu": {}, "options": {}, "families": {},
              "regimes": {}}
    print("\nstatic planes (start_em.F:1539-1626), per topo_wind option:")
    for fname, st in res["static"].items():
        print(f"  {fname:<12} max_ulp={st.max_ulp:<6d} differing="
              f"{st.differing}/{st.total}  worst={st.worst}")
        report["static"][fname] = vars(st)
    shallow = [(p.index, opt, n) for p in patches for opt in OPTIONS
               for n, c in enumerate(p.cells) if wrf_undefined_shallow(c)]
    report["wrf_undefined_shallow_calls"] = shallow
    print(f"calls on a column below 500 m, where WRF's get_pblh reads past "
          f"its arrays (reported apart): {len(shallow)}")
    print("\nYSU outputs (every call where WRF is finite and defined):")
    for fname, st in res["ysu"].items():
        print(f"  {fname:<8} max_ulp={st.max_ulp:<10d} max_abs="
              f"{st.max_abs:<11.4g} differing={st.differing}/{st.total}"
              f"  worst={st.worst}")
        report["ysu"][fname] = vars(st)
    print(f"kpbl mismatches: {res['kpbl_mismatch']}")
    report["kpbl_mismatch"] = res["kpbl_mismatch"]

    def table(title, key, labels, select):
        print(f"\n{title}:")
        for k, label in labels.items():
            calls = [v for (pi, opt, n), v in res["per_call"].items()
                     if select(pi, opt, n) == k]
            if not calls:
                continue
            m = max(c[0] for c in calls)
            d = sum(c[1] for c in calls)
            print(f"  {label:<24} calls={len(calls):<5d} max_ulp={m:<10d} "
                  f"differing_words={d}")
            report[key][label] = {"calls": len(calls), "max_ulp": m,
                                  "differing_words": d}

    by_index = {p.index: p for p in patches}
    table("per topo_wind option", "options",
          {o: f"topo_wind={o}" for o in OPTIONS}, lambda pi, opt, n: opt)
    table("per terrain family", "families", FAMILIES,
          lambda pi, opt, n: by_index[pi].family)
    table("per column regime", "regimes", REGIMES,
          lambda pi, opt, n: by_index[pi].cells[n].regime)

    if shallow:
        sres = compare(patches, port, include=lambda p, opt, c:
                       wrf_undefined_shallow(c) and c.wrf_finite(opt))
        print("\nthe below-500 m calls (WRF undefined), for the record:")
        for fname, st in sres["ysu"].items():
            if st.differing:
                print(f"  {fname:<8} max_ulp={st.max_ulp} differing="
                      f"{st.differing}/{st.total}")
        report["shallow"] = {k: vars(v) for k, v in sres["ysu"].items()}
    print("\nnon-finite WRF calls (WRF's own 0/0; WOOF keeps a defined value):")
    for pi, opt, n in nan:
        c = by_index[pi].cells[n]
        s = c.scalars
        g = port[(pi, opt)]["cells"][n]
        fin = all(np.all(np.isfinite(g[k])) for k in
                  ("du", "dv", "dtheta", "dqv", "exch_h", "exch_m"))
        bad = [k for k, v in c.wrf[opt].items() if k != "qmixtnp"
               and not np.all(np.isfinite(v))]
        print(f"  patch {pi} opt {opt} cell {n}: ust={float(s['ust']):.3g} "
              f"hfx={float(s['hfx']):.3g} br={float(s['br']):.3g}; WRF "
              f"non-finite in {bad}; WOOF finite={fin}")
    if wrf_exact.ENABLED:
        rec = wrf_exact.compile_receipt()
        report["strict_options"] = rec["strict_options"]
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2,
                                              default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
