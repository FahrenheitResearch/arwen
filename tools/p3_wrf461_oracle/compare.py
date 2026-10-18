#!/usr/bin/env python3
"""Bitwise comparison of WOOF P3 dumps against the WRF Fortran oracle.

    python compare.py <fixdir> <ref_dir> <cand_dir> [--json out.json] [--quiet]

Every frame (1..nsteps), every one of the 22 output fields, every element.
"Match" means the float32 bit patterns are equal (NaN matches NaN only when
both are NaN).  ULP distance is the monotonic int32 ordering, exact across
zero (gpuwm.core.fp32_ulp); a NaN on one side only is reported as a NaN
mismatch, never folded into a ULP number.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key as ordered

OUT3D = ["th", "qv", "qc", "qr", "nr", "qi", "ni", "qir", "qib",
         "th_old", "qv_old", "zdbz", "effc", "effi", "vmi", "di", "rhopo"]
OUT2D = ["rainnc", "rainncv", "sr", "snownc", "snowncv"]


def read(path, ncol, nk, nsteps):
    buf = np.fromfile(path, dtype="<f4")
    per = len(OUT3D) * ncol * nk + len(OUT2D) * ncol
    if buf.size != per * (nsteps + 1):
        raise SystemExit(f"{path}: {buf.size} floats, expected "
                         f"{per * (nsteps + 1)}")
    frames = buf.reshape(nsteps + 1, per)
    out = {}
    o = 0
    for f in OUT3D:     # -> (nsteps+1, nk, ncol)
        out[f] = frames[:, o:o + ncol * nk].reshape(nsteps + 1, nk, ncol)
        o += ncol * nk
    for f in OUT2D:     # -> (nsteps+1, 1, ncol)
        out[f] = frames[:, o:o + ncol].reshape(nsteps + 1, 1, ncol)
        o += ncol
    return out


def compare_case(case, ref_dir, cand_dir):
    name = case["name"]
    ncol, nk, nsteps = case["ncol"], case["nk"], case["nsteps"]
    R = read(pathlib.Path(ref_dir) / (name + ".ref"), ncol, nk, nsteps)
    C = read(pathlib.Path(cand_dir) / (name + ".ref"), ncol, nk, nsteps)
    res = {"name": name, "ncol": ncol, "nk": nk, "nsteps": nsteps,
           "fields": {}, "n_elements": 0, "n_differing": 0, "max_ulp": 0,
           "max_abs": 0.0, "max_rel": 0.0, "nan_mismatch": 0,
           "first_differing_step": None, "columns_differing": [],
           "in_claim": case.get("in_claim", True)}
    col_bad = np.zeros(ncol, dtype=bool)
    for f in OUT3D + OUT2D:
        r, c = R[f][1:], C[f][1:]
        rn, cn = np.isnan(r), np.isnan(c)
        same = (r.view(np.uint32) == c.view(np.uint32)) | (rn & cn)
        nanmis = int(np.count_nonzero(rn ^ cn))
        diff = ~same
        nd = int(np.count_nonzero(diff))
        fin = ~(rn | cn)
        u = np.where(fin, np.abs(ordered(r) - ordered(c)), 0)
        a = np.where(fin, np.abs(r.astype(np.float64) - c.astype(np.float64)),
                     0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(fin & (r != 0), a / np.abs(r.astype(np.float64)),
                           0.0)
        entry = {"n_differing": nd, "max_ulp": int(u.max()),
                 "max_abs": float(a.max()), "max_rel": float(rel.max()),
                 "nan_mismatch": nanmis}
        if nd:
            steps = np.nonzero(diff.reshape(nsteps, -1).any(axis=1))[0]
            entry["first_step"] = int(steps[0]) + 1
            s, k, i = np.unravel_index(int(np.argmax(np.where(diff, u, -1))),
                                       u.shape)
            entry["worst_at"] = {"step": int(s) + 1, "k": int(k), "col": int(i),
                                 "ref": float(r[s, k, i]),
                                 "cand": float(c[s, k, i])}
            col_bad |= diff.any(axis=(0, 1))
            fs = entry["first_step"]
            if (res["first_differing_step"] is None
                    or fs < res["first_differing_step"]):
                res["first_differing_step"] = fs
        res["fields"][f] = entry
        res["n_elements"] += int(r.size)
        res["n_differing"] += nd
        res["max_ulp"] = max(res["max_ulp"], entry["max_ulp"])
        res["max_abs"] = max(res["max_abs"], entry["max_abs"])
        res["max_rel"] = max(res["max_rel"], entry["max_rel"])
        res["nan_mismatch"] += nanmis
    res["columns_differing"] = [int(i) for i in np.nonzero(col_bad)[0]]
    labels = [c["label"] for c in case.get("columns", [])]
    if labels:
        res["column_labels_differing"] = [labels[i]
                                          for i in res["columns_differing"]]
    # step-1 only numbers (before any feedback through later steps)
    s1 = 0
    for f in OUT3D + OUT2D:
        r, c = R[f][1], C[f][1]
        fin = ~(np.isnan(r) | np.isnan(c))
        s1 = max(s1, int(np.where(fin, np.abs(ordered(r) - ordered(c)),
                                  0).max()))
    res["step1_max_ulp"] = s1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixdir")
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    index = json.loads((pathlib.Path(a.fixdir) / "index.json").read_text())
    rows = []
    for case in index["cases"]:
        if not (pathlib.Path(a.cand_dir) / (case["name"] + ".ref")).exists():
            continue
        r = compare_case(case, a.ref_dir, a.cand_dir)
        rows.append(r)
        bad = {f: e for f, e in r["fields"].items() if e["n_differing"]}
        print(f"{r['name']:24s} {'' if r['in_claim'] else '(outside claim) '}"
              f"elements={r['n_elements']:8d} differing={r['n_differing']:7d}"
              f" max_ulp={r['max_ulp']} step1_max_ulp={r['step1_max_ulp']}"
              f" nan_mismatch={r['nan_mismatch']}"
              f" first_step={r['first_differing_step']}"
              f" columns={len(r['columns_differing'])}")
        if not a.quiet:
            for f, e in bad.items():
                print(f"    {f:8s} n={e['n_differing']:6d} ulp={e['max_ulp']}"
                      f" abs={e['max_abs']:.3e} rel={e['max_rel']:.3e}"
                      f" nan={e['nan_mismatch']} first={e.get('first_step')}"
                      f" worst={e.get('worst_at')}")
    claim = [r for r in rows if r["in_claim"]]
    summary = {
        "cases": len(rows),
        "claim_cases": len(claim),
        "claim_bit_identical": all(r["n_differing"] == 0 for r in claim),
        "claim_max_ulp": max((r["max_ulp"] for r in claim), default=0),
        "claim_max_abs": max((r["max_abs"] for r in claim), default=0.0),
        "claim_max_rel": max((r["max_rel"] for r in claim), default=0.0),
        "claim_nan_mismatch": sum(r["nan_mismatch"] for r in claim),
        "claim_elements": sum(r["n_elements"] for r in claim),
        "claim_differing": sum(r["n_differing"] for r in claim),
    }
    print("SUMMARY", json.dumps(summary))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"summary": summary, "cases": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
