"""Bitwise comparison of WOOF's km_opt=2 outputs against compiled WRF.

Every word of every output field is compared as a uint32 bit pattern (signed
zeros included).  The ULP distance is a reported measurement, never an
acceptance tolerance.

Usage: python compare.py CASES_DIR WRF_DIR WOOF_DIR OUT_JSON [--label TEXT]
       [--got-prefix wrf]   (second directory holds another WRF build)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from arms import ARMS  # noqa: E402

FIELDS = ("kmh", "kmv", "khh", "khv", "bn2", "rtke", "d11", "d22", "d12",
          "d33", "d13", "d23")


def _keys(a):
    """Monotone integer keys for float32 words (gpuwm.core.fp32_ulp's map)."""
    bits = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, np.int64(-0x80000000) - bits, bits)


def word_stats(got, ref):
    got = np.ascontiguousarray(got, dtype=np.float32)
    ref = np.ascontiguousarray(ref, dtype=np.float32)
    if got.shape != ref.shape:
        raise ValueError(f"shape {got.shape} vs {ref.shape}")
    diff = got.view(np.uint32) != ref.view(np.uint32)
    ulp = np.abs(_keys(got) - _keys(ref))
    nan = ~np.isfinite(got) | ~np.isfinite(ref)
    ulp = np.where(nan & diff, np.int64(2) ** 40, ulp)
    stats = {"words": int(got.size), "different": int(diff.sum()),
             "max_ulp": int(ulp.max(initial=0)),
             "max_abs": float(np.max(np.abs(got.astype(np.float64)
                                            - ref.astype(np.float64)),
                                     initial=0.0)),
             "nonfinite": int(nan.sum())}
    if stats["different"]:
        at = np.unravel_index(int(np.argmax(ulp)), ulp.shape)
        stats["worst_at"] = [int(x) for x in at]
        stats["worst_words"] = [float(got[at]), float(ref[at])]
        levels = sorted({int(x) for x in np.nonzero(diff)[0]})
        stats["levels"] = levels[:8] + (["..."] if len(levels) > 8 else [])
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", type=Path)
    ap.add_argument("wrf", type=Path)
    ap.add_argument("woof", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--got-prefix", default="woof",
                    help="file prefix in the second directory ('wrf' to "
                         "compare two WRF builds)")
    args = ap.parse_args()
    index = json.loads((args.cases / "cases.json").read_text())
    detail, total = {}, {f: {"words": 0, "different": 0, "max_ulp": 0,
                             "max_abs": 0.0} for f in FIELDS}
    for case in index["cases"]:
        for arm in ARMS:
            tag = f"{case['name']}/{arm[0]}"
            with np.load(args.wrf / f"wrf-{case['name']}-{arm[0]}.npz") as w, \
                    np.load(args.woof / f"{args.got_prefix}-{case['name']}-{arm[0]}.npz") as g:
                detail[tag] = {f: word_stats(g[f], w[f]) for f in FIELDS}
            for f, s in detail[tag].items():
                t = total[f]
                t["words"] += s["words"]; t["different"] += s["different"]
                t["max_ulp"] = max(t["max_ulp"], s["max_ulp"])
                t["max_abs"] = max(t["max_abs"], s["max_abs"])
    columns = sum(c["columns"] for c in index["cases"])
    words = sum(t["words"] for t in total.values())
    different = sum(t["different"] for t in total.values())
    summary = {"label": args.label, "columns": columns,
               "cases": len(index["cases"]), "arms": [a[0] for a in ARMS],
               "words": words, "different": different,
               "max_ulp": max(t["max_ulp"] for t in total.values()),
               "fields": total}
    receipt = args.woof / "woof-receipt.json"
    if receipt.exists():
        summary["woof_build"] = json.loads(receipt.read_text())
    args.out.write_text(json.dumps({"summary": summary, "detail": detail},
                                   indent=1) + "\n", encoding="utf-8")
    print(f"{args.label}: {different} of {words} words differ, max ULP "
          f"{summary['max_ulp']} ({columns} columns x {len(ARMS)} arms)")
    for f, t in total.items():
        print(f"  {f:5s} {t['different']:7d} / {t['words']:8d}  max_ulp "
              f"{t['max_ulp']:>10d}  max_abs {t['max_abs']:.3e}")


if __name__ == "__main__":
    main()
