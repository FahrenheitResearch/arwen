"""Record each GPU operator's word and ULP comparison against native Fortran.

Run only under the GPU ownership protocol. This replays the same chained
fixture as the focused test and writes a compact, auditable JSON receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def comparison(actual, expected):
    a = np.ascontiguousarray(actual, dtype=np.float32)
    b = np.ascontiguousarray(expected, dtype=np.float32)
    au, bu = a.view(np.uint32), b.view(np.uint32)
    different = au != bu
    zeros = different & (a == 0) & (b == 0)
    # IEEE total-order distance, with signed zero counted separately.
    ao = np.where(au & 0x80000000, (~au).astype(np.uint64),
                  au.astype(np.uint64) | 0x80000000).astype(np.int64)
    bo = np.where(bu & 0x80000000, (~bu).astype(np.uint64),
                  bu.astype(np.uint64) | 0x80000000).astype(np.int64)
    ulp = np.abs(ao - bo)
    ulp[zeros] = 0
    return {"words": int(a.size), "different_words": int(different.sum()),
            "signed_zeros": int(zeros.sum()),
            "different_nonzero_words": int((different & ~zeros).sum()),
            "max_ulp": int(ulp.max(initial=0)),
            "max_abs": float(np.abs(a.astype(np.float64) - b).max(initial=0)),
            "actual_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(b.tobytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tests"))
    sys.path.insert(0, str(root))
    import cupy as cp
    from test_zadvect_implicit import _kernel_results

    fixture = root / "tests/data/ieva_wrf_legacy.npz"
    result = _kernel_results(cp, variant="wrf_legacy", fixture=fixture)
    receipt = {"variant": "wrf_legacy", "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
               "fields": {}, "scope": "native operator chain on specified domain; forced outer w ring excluded"}
    with np.load(fixture) as packed:
        receipt["source"] = json.loads(str(packed["provenance"]))
        for name, (actual, shape, region) in result.items():
            receipt["fields"][name] = comparison(actual[region], packed[f"wrf_{name}"].reshape(shape)[region])
        actual, shape, region = result["rw_t"]
        receipt["unmodified_w"] = comparison(actual[region], packed["original_rw_t"].reshape(shape)[region])
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"fields": {k: {p: v[p] for p in ("words", "different_nonzero_words", "max_ulp")}
                                  for k, v in receipt["fields"].items()},
                      "unmodified_w_different_words": receipt["unmodified_w"]["different_words"]}))


if __name__ == "__main__":
    main()
