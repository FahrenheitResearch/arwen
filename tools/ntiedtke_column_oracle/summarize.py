"""One table over every run of run_matrix.sh, plus the attribution check.

    python summarize.py RUNS_DIR > summary.md

For each run it reads the compare.py receipts and adds three things the
receipts cannot say on their own:

* zcbase columns: the columns where the pinned WRF build and the build
  with cutypen's zcbase declared real disagree.  These are the columns the
  WRF declaration bug decides.
* attribution: every column where WOOF differs from the pinned WRF build
  must be one of those columns.  If WOOF differs anywhere else, the
  difference is not the WRF bug and the row says so.
* stock: whether WRF's own -O2 build gives the same words as the -O0
  reference.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

LEV = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten", "rvcuten")
SFC = ("raincv", "pratec")


def columns_differing(a_dir, b_dir, ny, nz, nx, rep_b=1):
    """Columns (of the unreplicated case) where any graded word differs."""
    out = np.zeros(ny * nx, dtype=bool)
    for n in LEV + SFC:
        pa, pb = a_dir / f"{n}.f32", b_dir / f"{n}.f32"
        if not (pa.exists() and pb.exists()):
            continue
        k = nz if n in LEV else 1
        a = np.fromfile(pa, "<f4").reshape(ny, k, nx).view(np.int32)
        b = np.fromfile(pb, "<f4").reshape(ny, k, nx * rep_b).view(np.int32)
        for r in range(rep_b):
            out |= (a != b[:, :, r * nx:(r + 1) * nx]).any(axis=1).ravel()
    return out


def main() -> int:
    runs = pathlib.Path(sys.argv[1])
    rows = []
    totals = {"columns": 0, "words": 0}
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        cmp = {}
        for f in d.glob("cmp-*.json"):
            cmp[f.stem[4:]] = json.loads(f.read_text())
        if "strict-zfix" not in cmp:
            continue
        s = cmp["strict-zfix"]
        nz = s["nz"]
        info = s["woof"]
        rep = int(info["replicate"])
        ncol = s["columns"]
        # ny*nx from the case
        meta = {}
        for line in open(pathlib.Path(s["case"]) / "meta.txt"):
            k, _, v = line.strip().partition(" ")
            meta[k] = v
        ny, nx = int(meta["ny"]), int(meta["nx"])
        zcols = columns_differing(d / "wrf-ref", d / "wrf-zfix", ny, nz, nx)
        wref = columns_differing(d / "wrf-ref", d / "woof-strict", ny, nz,
                                 nx, rep)
        stock_same = not columns_differing(d / "wrf-ref", d / "wrf-stock",
                                           ny, nz, nx).any()
        attributed = bool(np.all(zcols[wref])) if wref.any() else True
        dz = cmp["default-zfix"]
        maxabs = max(f.get("max_abs_diff", 0.0) for f in dz["fields"].values())
        reg = s["regimes"]
        rows.append({
            "run": d.name, "columns": ncol * rep, "unique": ncol, "nz": nz,
            "strict_zfix_words": s["total_differing_words"],
            "strict_zfix_ulp": s["max_ulp"],
            "default_zfix_words": dz["total_differing_words"],
            "default_zfix_ulp": dz["max_ulp"], "default_zfix_abs": maxabs,
            "strict_ref_cols": int(wref.sum()),
            "zcbase_cols": int(zcols.sum()), "attributed": attributed,
            "stock_same": stock_same,
            "active": reg["active"]["columns"], "deep": reg["deep"]["columns"],
            "raining": reg["raining"]["columns"],
            "chunks": info["chunks"], "fused": info["fused_walk"],
            "padded": info["padded_columns"],
            "pratec": info["pratec_graded"],
        })
        totals["columns"] += ncol * rep
        totals["words"] += s["total_words"]
    print("| run | columns graded | nz | strict vs WRF (zcbase real): words differing, max ULP "
          "| default vs same: words, max ULP, max abs | strict vs pinned WRF: columns differing "
          "| columns the zcbase bug decides | all differences attributed | stock -O2 == -O0 "
          "| active / deep / raining | walk |")
    print("|---|---:|---:|---|---|---:|---:|---|---|---|---|")
    for r in rows:
        walk = (f"{r['chunks']} chunk(s), {'fused' if r['fused'] else 'unfused'}"
                + (f", {r['padded']} pad" if r["padded"] else ""))
        print(f"| {r['run']} | {r['columns']:,} ({r['unique']:,} unique) | {r['nz']} "
              f"| {r['strict_zfix_words']}, {r['strict_zfix_ulp']} "
              f"| {r['default_zfix_words']}, {r['default_zfix_ulp']}, {r['default_zfix_abs']:.3g} "
              f"| {r['strict_ref_cols']} | {r['zcbase_cols']} "
              f"| {'yes' if r['attributed'] else 'NO'} | {'yes' if r['stock_same'] else 'NO'} "
              f"| {r['active']:,} / {r['deep']:,} / {r['raining']:,} | {walk} |")
    print()
    print(f"Total: {totals['columns']:,} column evaluations, "
          f"{totals['words']:,} output words per arithmetic mode.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
