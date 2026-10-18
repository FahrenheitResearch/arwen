#!/usr/bin/env python3
"""The topo_wind chain with Shin-Hong as the consumer: every field's distance.

    python tools/topo_wind_wrf461_oracle/compare_shinhong.py
        [--reference O2s|wrfflags|O0] [--json OUT.json] [DIR]

WRF: start_em.F:1539-1626 on each patch, then module_bl_shinhong.F's
``shinhong`` on every cell with the CTOPO/CTOPO2 it formed
(run_shinhong_topo_chain.F90).  WOOF: topo_wind_coefficients, then
launch_shinhong with WOOF's own coefficients (arm B for topo_wind 1 and 2).
Run once with GPUWM_WRF_EXACT=1 and once without.  Asserts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpuwm.verify.topo_wind_oracle import (  # noqa: E402
    FAMILIES, OPTIONS, SHINHONG_REFERENCES, compare_shinhong,
    load_shinhong_chain, load_topo_wind_oracle, run_port_shinhong)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--reference", default="O2s",
                    choices=tuple(SHINHONG_REFERENCES))
    args = ap.parse_args(argv)

    import cupy as cp
    from gpuwm import wrf_exact

    patches = load_topo_wind_oracle(args.directory)
    cols = load_shinhong_chain(patches, args.directory,
                               reference=args.reference)
    port = run_port_shinhong(patches, cols)
    mode = "strict (GPUWM_WRF_EXACT=1)" if wrf_exact.ENABLED else "default"
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"].decode() if isinstance(props["name"], bytes) \
        else props["name"]
    nvrtc = ".".join(map(str, cp.cuda.nvrtc.getVersion()))
    print(f"reference: WRF v4.6.1 start_em.F block + module_bl_shinhong.F, "
          f"{args.reference} build")
    print(f"arithmetic: {mode}   device: {name} sm_{props['major']}"
          f"{props['minor']}   NVRTC {nvrtc}   cupy {cp.__version__}")
    report = {"reference": args.reference, "mode": mode, "device": name,
              "nvrtc": nvrtc, "ncall": len(cols), "options": {},
              "families": {}}

    def show(title, res, key=None):
        print(f"\n{title}: {len(res['keys'])} calls")
        out = {}
        for fname, st in list(res["fields"].items()) + list(res["topo"].items()):
            d = st.as_dict() if hasattr(st, "as_dict") else vars(st)
            extra = (f" nan_lanes={d.get('nan_lanes', 0)} ftz_lanes="
                     f"{d.get('ftz_lanes', 0)} max_ulp_outside_ftz="
                     f"{d.get('max_ulp_outside_ftz', '-')}"
                     if "nan_lanes" in d else "")
            print(f"  {fname:<8} max_ulp={d['max_ulp']:<10d} differing="
                  f"{d['differing']}/{d['total']}{extra}  worst={d['worst']}")
            out[fname] = d
        print(f"  kpbl mismatches: {res['kpbl_mismatch']}")
        oob = res.get("oob_excluded")
        if oob is not None:
            print(f"  tke/el words excluded where WRF read q2xk(kte+1): {oob}")
        out["kpbl_mismatch"] = res["kpbl_mismatch"]
        return out

    report["all"] = show("all calls", compare_shinhong(cols, port))
    for opt in OPTIONS:
        report["options"][opt] = show(
            f"topo_wind = {opt}",
            compare_shinhong(cols, port,
                             include=lambda k, c, o=opt: k[1] == o))
    by_patch = {p.index: p.family for p in patches}
    for fid, fname in FAMILIES.items():
        report["families"][fname] = show(
            f"family {fname}",
            compare_shinhong(cols, port,
                             include=lambda k, c, f=fid: by_patch[k[0]] == f))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2,
                                              default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
