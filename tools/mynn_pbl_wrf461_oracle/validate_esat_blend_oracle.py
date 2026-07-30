#!/usr/bin/env python3
"""Compare the CPU MYNN phase-blend helpers with the official WRF CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from gpuwm.core.fp32_ulp import monotone_fp32_key
from gpuwm.core.mynn_pbl import (
    mynn_esat_blend, mynn_qsat_blend, mynn_xl_blend,
)


HELPERS = {
    "esat_blend": lambda t, p: mynn_esat_blend(t),
    "qsat_blend": mynn_qsat_blend,
    "xl_blend": lambda t, p: mynn_xl_blend(t),
}
CASES = ("surface_pressure", "midlevel_pressure", "upper_pressure")


#: The total-order FP32 key is shared rather than re-derived here.
#: Thirteen local copies of this two-line bit trick carried the same
#: sign error, which reported -0.0 as 2**32 ULP from +0.0; see
#: gpuwm/core/fp32_ulp.py for the measurement and the live case.
_ordered_bits = monotone_fp32_key


def main(path: str) -> None:
    with Path(path).open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    assert cases == CASES, cases
    t = np.asarray([np.float32(row["t"]) for row in rows], dtype=np.float32)
    p = np.asarray([np.float32(row["p"]) for row in rows], dtype=np.float32)
    assert np.isfinite(t).all() and np.isfinite(p).all()
    # Branch coverage over esat_blend / qsat_blend: XC clamp, pure ice,
    # blended band, pure liquid.  xl_blend switches at t0c instead of t0c-6.
    assert (t <= np.float32(193.15)).sum() >= 3, "XC clamp uncovered"
    assert (t <= np.float32(240.0)).sum() >= 8, "ice branch uncovered"
    assert ((t > np.float32(240.0)) & (t < np.float32(267.15))).sum() >= 6, \
        "blended band uncovered"
    assert (t >= np.float32(267.15)).sum() >= 10, "liquid branch uncovered"
    assert ((t > np.float32(240.0)) & (t < np.float32(273.15))).sum() >= 8, \
        "xl_blend blended band uncovered"

    summary: dict[str, dict[str, float | int]] = {}
    ceiling_hits = 0
    for name, helper in HELPERS.items():
        expected = np.asarray(
            [np.float32(row[name]) for row in rows], dtype=np.float32
        )
        got = np.asarray(
            [helper(ti, pi) for ti, pi in zip(t, p)], dtype=np.float32
        )
        ulp = np.abs(_ordered_bits(got) - _ordered_bits(expected))
        summary[name] = {
            "max_abs": float(np.abs(
                got.astype(np.float64) - expected.astype(np.float64)
            ).max(initial=0.0)),
            "max_ulp": int(ulp.max(initial=0)),
            "rows": int(expected.size),
        }
        np.testing.assert_array_equal(got, expected, err_msg=name)
        if name == "esat_blend":
            ceiling_hits = int(np.count_nonzero(got > np.float32(0.15) * p))
    # qsat_blend clamps the vapour pressure at 0.15*p; the upper-pressure
    # rows must actually reach that ceiling or the clamp stays unpinned.
    assert ceiling_hits > 0, "qsat_blend 0.15*p ceiling uncovered"
    summary["qsat_ceiling_rows"] = ceiling_hits
    print(json.dumps({"status": "PASS", "helpers": summary}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: validate_esat_blend_oracle.py esat-blend.csv"
        )
    main(sys.argv[1])
