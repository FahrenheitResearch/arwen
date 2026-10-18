"""Bitwise comparison of the WRF and WOOF NSSL option-18 column outputs.

A field matches only at 0 ULP: identical float32 bit patterns (+0 and -0 are
counted as a 1-ULP-free sign difference and reported separately).  For each
field the report gives the count of differing values, the maximum ULP
distance, the maximum absolute difference, and the worst column/level with
its regime, overall and per regime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_gpu_columns import OUT2, OUT3


def read_fortran(path: Path, ncol: int, nz: int) -> dict:
    raw = np.fromfile(path, dtype="<f4")
    expected = len(OUT3) * ncol * nz + len(OUT2) * ncol
    if raw.size != expected:
        raise ValueError(f"{path}: {raw.size} values, expected {expected}")
    out, off = {}, 0
    for name in OUT3:
        out[name] = raw[off:off + ncol * nz].reshape(ncol, nz)
        off += ncol * nz
    for name in OUT2:
        out[name] = raw[off:off + ncol]
        off += ncol
    return out


def ordered(a: np.ndarray) -> np.ndarray:
    i = a.view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(ordered(a) - ordered(b))
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    d = np.where(both_nan, 0, d)
    return np.where(one_nan, np.iinfo(np.int64).max, d)


def field_stats(w: np.ndarray, g: np.ndarray, regime: np.ndarray) -> dict:
    bits_differ = w.view(np.uint32) != g.view(np.uint32)
    u = ulp(w, g)
    with np.errstate(invalid="ignore"):
        absd = np.abs(w.astype(np.float64) - g.astype(np.float64))
    absd = np.where(np.isnan(absd), 0.0, absd)
    stats = {
        "values": int(w.size),
        "bit_mismatches": int(bits_differ.sum()),
        "max_ulp": int(u.max()) if u.size else 0,
        "max_abs_diff": float(absd.max()) if absd.size else 0.0,
        "signed_zero_only": int((bits_differ & (u == 0)).sum()),
    }
    if stats["bit_mismatches"]:
        flat = int(np.argmax(np.where(bits_differ, u, -1)))
        idx = np.unravel_index(flat, w.shape)
        stats["worst"] = {
            "column": int(idx[0]), "level": (int(idx[1]) + 1 if w.ndim == 2 else None),
            "regime": str(regime[idx[0]]), "wrf": float(w[idx]), "woof": float(g[idx]),
        }
        cols = np.unique(np.nonzero(bits_differ.reshape(w.shape[0], -1).any(axis=1))[0])
        stats["columns_with_mismatch"] = int(cols.size)
    return stats


def compare(wrf_bin: Path, gpu_npz: Path, columns_npz: Path, report: Path,
            b_is_fortran: bool = False) -> dict:
    cols = np.load(columns_npz, allow_pickle=False)
    regime = cols["regime"]
    ncol, nz = cols["theta"].shape
    if b_is_fortran:
        gpu = read_fortran(gpu_npz, ncol, nz)
    else:
        gpu = dict(np.load(gpu_npz, allow_pickle=False))
    wrf = read_fortran(wrf_bin, ncol, nz)
    fields, per_regime = {}, {}
    total = 0
    for name in OUT3 + OUT2:
        w, g = wrf[name], gpu[name].astype(np.float32)
        st = field_stats(w, g, regime)
        fields[name] = st
        total += st["bit_mismatches"]
        for r in np.unique(regime):
            sel = regime == r
            sub = field_stats(w[sel], g[sel], regime[sel])
            entry = per_regime.setdefault(str(r), {"columns": int(sel.sum()),
                                                   "bit_mismatches": 0, "max_ulp": 0,
                                                   "fields_off": []})
            entry["bit_mismatches"] += sub["bit_mismatches"]
            entry["max_ulp"] = max(entry["max_ulp"], sub["max_ulp"])
            if sub["bit_mismatches"]:
                entry["fields_off"].append(name)
    result = {
        "schema": "woof.nssl2.column-oracle-comparison/v1",
        "wrf": str(wrf_bin), "woof": str(gpu_npz), "ncol": int(ncol), "nz": int(nz),
        "fields_compared": len(OUT3) + len(OUT2),
        "values_compared": int(sum(f["values"] for f in fields.values())),
        "total_bit_mismatches": int(total),
        "max_ulp": int(max(f["max_ulp"] for f in fields.values())),
        "max_abs_diff_by_field": {k: v["max_abs_diff"] for k, v in fields.items()},
        "verdict": "EXACT_0_ULP" if total == 0 else "MISMATCH",
        "fields": fields, "regimes": per_regime,
    }
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("verdict", "ncol", "values_compared",
                                              "total_bit_mismatches", "max_ulp")}))
    for name, st in fields.items():
        if st["bit_mismatches"]:
            print(f"  {name:11s} mismatches={st['bit_mismatches']:6d} "
                  f"max_ulp={st['max_ulp']:<12d} max_abs={st['max_abs_diff']:.3e} "
                  f"cols={st.get('columns_with_mismatch')} worst={st['worst']}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wrf_bin", type=Path)
    p.add_argument("gpu_npz", type=Path)
    p.add_argument("columns_npz", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--b-is-fortran", action="store_true",
                   help="second input is another Fortran output stream")
    a = p.parse_args()
    r = compare(a.wrf_bin, a.gpu_npz, a.columns_npz, a.report, a.b_is_fortran)
    raise SystemExit(0 if r["verdict"] == "EXACT_0_ULP" else 1)


if __name__ == "__main__":
    main()
