"""Bitwise comparison of WOOF's legacy RRTMG outputs with the WRF oracle.

A field matches when every float32 bit pattern is identical (max ULP 0 and
no signed-zero or NaN-payload difference).  The ULP distance is measured on
the ordered integer line of float32 bit patterns.

usage: python compare.py COLUMNS_DIR WRF_DIR WOOF_DIR ARM [WRF_ARM] > summary.json
  WRF_DIR/<WRF_ARM>/out_<regime>.bin  (default WRF_ARM = stock)
  WOOF_DIR/woof_<ARM>_<regime>.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

#: WRF record -> WOOF output.  rthratensw_wrapper is RRTMG_SWRAD's own
#: heating rate; the driver's published RTHRATENSW is RTHRATEN - RTHRATENLW
#: and is compared separately (PUBLISHED).
PAIRS = {"rthratenlw": "rthratenlw", "rthratensw_wrapper": "rthratensw",
         "rthraten": "rthraten", "glw": "glw", "olr": "olr", "gsw": "gsw",
         "swdown": "swdown", "coszen": "coszen", "cldfra": "cldfra",
         "swddir": "swddir", "swddif": "swddif", "o3rad": "o3rad"}
PUBLISHED = {"rthratensw": "rthratensw"}


def read_records(path: Path) -> dict:
    raw = path.read_bytes()
    out, pos = {}, 0
    while pos < len(raw):
        name = raw[pos:pos + 32].decode().strip()
        n = int(np.frombuffer(raw, "<i4", 1, pos + 32)[0])
        out[name] = np.frombuffer(raw, "<f4", n, pos + 36).copy()
        pos += 36 + 4 * n
    return out


def ordered(a):
    i = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def diff(wrf, woof):
    wrf = np.asarray(wrf, np.float32).ravel()
    woof = np.asarray(woof, np.float32).ravel()
    if wrf.shape != woof.shape:
        return {"error": f"shape {wrf.shape} vs {woof.shape}"}
    bits = wrf.view(np.uint32) != woof.view(np.uint32)
    both_nan = np.isnan(wrf) & np.isnan(woof)
    ulp = np.abs(ordered(wrf) - ordered(woof))
    ulp[both_nan] = 0
    absd = np.abs(wrf.astype(np.float64) - woof.astype(np.float64))
    worst = int(np.argmax(ulp)) if ulp.size else 0
    return {"n": int(wrf.size), "differ": int(bits.sum()),
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "max_abs": float(np.nanmax(absd)) if absd.size else 0.0,
            "worst_index": worst,
            "worst": [float(wrf[worst]), float(woof[worst])] if ulp.size else None}


def to_cols(vec, ncol):
    return vec.reshape(-1, ncol).T if vec.size != ncol else vec   # Fortran (ncol, nk)


def main():
    cols_dir, wrf_dir, woof_dir, arm = (Path(sys.argv[1]), Path(sys.argv[2]),
                                        Path(sys.argv[3]), sys.argv[4])
    wrf_arm = sys.argv[5] if len(sys.argv) > 5 else "stock"
    meta = json.loads((cols_dir / "regimes.json").read_text())
    ncol = meta["ncol"]
    summary = {"arm": arm, "wrf_arm": wrf_arm, "regimes": {}}
    total_n = total_diff = 0
    worst_ulp = 0
    for name in meta["regimes"]:
        wpath = wrf_dir / wrf_arm / f"out_{name}.bin"
        opath = woof_dir / f"woof_{arm}_{name}.npz"
        if not (wpath.exists() and opath.exists()):
            continue
        wrf = read_records(wpath)
        woof = dict(np.load(opath))
        fields = {}
        for wk, ok in PAIRS.items():
            if ok not in woof or wk not in wrf:
                continue
            a = to_cols(wrf[wk], ncol)
            b = woof[ok]
            fields[wk] = diff(a, b)
            if "error" not in fields[wk]:
                col = int(np.unravel_index(fields[wk]["worst_index"], b.shape)[0]) \
                    if b.ndim == 2 else fields[wk]["worst_index"]
                fields[wk]["worst_column"] = col
                total_n += fields[wk]["n"]; total_diff += fields[wk]["differ"]
                worst_ulp = max(worst_ulp, fields[wk]["max_ulp"])
        published = {wk: diff(to_cols(wrf[wk], ncol), woof[ok])
                     for wk, ok in PUBLISHED.items() if wk in wrf and ok in woof}
        cz = to_cols(wrf["coszen"], ncol)
        summary["regimes"][name] = {"fields": fields, "published_diagnostics": published,
                                    "day_columns": int((cz > 0).sum())}
    summary["total_values"] = total_n
    summary["total_differ"] = total_diff
    summary["max_ulp"] = worst_ulp
    json.dump(summary, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
