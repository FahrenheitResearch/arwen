#!/usr/bin/env python3
"""Run WOOF's Shin-Hong on the wide column oracle and report every word.

    python tools/shinhong_column_oracle/compare.py ORACLE_DIR OUT.json \
        [--authority] [--no-gpu]

ORACLE_DIR holds columns-in.bin and wrf-out.bin (or their .gz) from
build.sh.  The GPU arm launches gpuwm.core.shinhong.launch_shinhong; set
GPUWM_WRF_EXACT=1 in the environment for the strict build (no FMA, no FTZ,
IEEE division and square root), leave it unset for default arithmetic.
--authority also runs the float32 NumPy authority (CPU).  Prints a table
and writes the full report as JSON.  Asserts nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from gpuwm.verify.shinhong_column_oracle import (
    LEVEL_FIELDS, SURFACE_FIELDS, compare, load_columns, run_authority,
    run_port, wrf_passthrough_violations)

FIELDS = tuple(LEVEL_FIELDS) + tuple(SURFACE_FIELDS)


def _summary(cols, result, label):
    rep = {"label": label,
           "fields": {k: v.as_dict() for k, v in result["fields"].items()},
           "kpbl_mismatch": result["kpbl_mismatch"],
           "oob_excluded_field_columns": result["oob_excluded"]}
    fam = defaultdict(lambda: defaultdict(int))
    fam_cols = defaultdict(int)
    for c in cols:
        fam_cols[c.family_name] += 1
        for name, u in result["per_column"].get(c.index, {}).items():
            fam[c.family_name][name] = max(fam[c.family_name][name], u)
    rep["per_family_max_ulp"] = {f: dict(v) for f, v in fam.items()}
    rep["per_family_columns"] = dict(fam_cols)
    worst = sorted(((max(v.values()) if v else 0, i)
                    for i, v in result["per_column"].items()), reverse=True)
    rep["worst_columns"] = [
        {"column": i, "max_ulp": u, "family": cols[i].family_name,
         "dx": float(cols[i].scalars["dx"]), "tke_diag": cols[i].tke_diag,
         "per_field": result["per_column"][i]}
        for u, i in worst[:15] if u > 0]
    return rep


def _print(rep):
    print(f"== {rep['label']}")
    print(f"{'field':8s} {'max_ulp':>10s} {'max_abs':>12s} {'differing':>10s}"
          f" {'lanes':>8s} {'nan':>5s} {'nan_geo':>7s} {'nan_bits':>8s}"
          f" {'ftz':>5s} {'ulp_noftz':>9s}")
    for name in FIELDS:
        s = rep["fields"].get(name)
        if s is None:
            continue
        print(f"{name:8s} {s['max_ulp']:10d} {s['max_abs']:12.4e} "
              f"{s['differing']:10d} {s['total']:8d} {s['nan_lanes']:5d} "
              f"{s['nan_geometry_mismatch']:7d} {s['nan_payload_mismatch']:8d}"
              f" {s['ftz_lanes']:5d} {s['max_ulp_outside_ftz']:9d}")
    print(f"kpbl mismatches: {len(rep['kpbl_mismatch'])}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle_dir", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--authority", action="store_true")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--refs", default="wrf-out.bin,wrf-out-O0.bin,"
                    "wrf-out-O2vec.bin")
    args = ap.parse_args(argv)

    refs = {}
    for name in args.refs.split(","):
        if ((args.oracle_dir / name).exists()
                or (args.oracle_dir / (name + ".gz")).exists()):
            refs[name] = load_columns(args.oracle_dir, outputs=name)
    cols = refs["wrf-out.bin"]
    report = {
        "columns": len(cols),
        "base_columns": len({c.base for c in cols}),
        "levels": int(sum(c.nz for c in cols)),
        "GPUWM_WRF_EXACT": os.environ.get("GPUWM_WRF_EXACT", ""),
        "host": platform.node(),
        "wrf_kpbl_eq_nz_tke_diag1": [c.index for c in cols if c.oob_tke],
        "wrf_passthrough_violations": wrf_passthrough_violations(cols),
        "wrf_nan_columns": sorted({c.index for c in cols
                                   if any(np.isnan(v).any()
                                          for v in c.wrf.values())}),
        "arms": {},
    }
    if not args.no_gpu:
        import cupy as cp

        from gpuwm.certify.compile_platform import (
            compile_platform_fingerprint)

        port = run_port(cols)
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        report["device"] = name.decode() if isinstance(name, bytes) else name
        report["compile_platform"] = compile_platform_fingerprint()
        try:
            from gpuwm import wrf_exact

            rec = wrf_exact.compile_receipt()
            report["strict_compile"] = {
                "enabled": rec["enabled"],
                "strict_options": rec["strict_options"],
                "shinhong_compiles": [r for r in rec["records"]
                                      if r["kind"] == "nvrtc_compile"]}
        except Exception as exc:  # noqa: BLE001
            report["strict_compile"] = repr(exc)
        for ref, rcols in refs.items():
            label = f"woof-gpu-vs-{ref.removesuffix('.bin')}"
            rep = _summary(rcols, compare(rcols, port), label)
            report["arms"][label] = rep
            _print(rep)
    if args.authority:
        auth = run_authority(cols)
        for ref, rcols in refs.items():
            label = f"numpy-authority-vs-{ref.removesuffix('.bin')}"
            rep = _summary(rcols, compare(rcols, auth), label)
            report["arms"][label] = rep
            _print(rep)
    args.out_json.write_text(json.dumps(report, indent=1, default=str) + "\n",
                             encoding="utf-8")
    print(f"columns {report['columns']} (base {report['base_columns']}), "
          f"WRF NaN columns {report['wrf_nan_columns']}, "
          f"kpbl==nz with tke_diag=1: {report['wrf_kpbl_eq_nz_tke_diag1']}, "
          f"passthrough violations {len(report['wrf_passthrough_violations'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
