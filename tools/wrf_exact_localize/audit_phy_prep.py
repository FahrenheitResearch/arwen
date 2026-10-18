"""Test two operation-level hypotheses against real step-1 dump words on CPU.

The scalar reference's powf is evaluated on the saved input quotient. This
is an analysis receipt, not a replacement scheme or a qualifying run.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np

from gpuwm.core import constants as c
from tools.wrf_exact_localize.cmpphys import compare, wrf_array


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wrf_dump", type=Path)
    p.add_argument("woof_dump", type=Path, nargs="?")
    p.add_argument("--wrf-only", action="store_true",
                   help="audit the density association on saved WRF primitives")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.wrf_only:
        ids, ide, jds, jde, kds, kde, *_ = map(
            int, (a.wrf_dump / "dims.txt").read_text().split())
        shape = (kde-kds, jde-jds, ide-ids)
        alt = wrf_array(a.wrf_dump, "b_prep", "alt", shape)
        qv = wrf_array(a.wrf_dump, "b_prep", "qv", shape)
        reference = wrf_array(a.wrf_dump, "p00_phy_prep", "rho", shape)
        f = np.float32
        result = dict(schema="gpuwm.density-source-audit/v1",
                      source_expression=compare(reference, (f(1)+qv)/alt),
                      wrf_order=compare(reference, (f(1)/alt)*(f(1)+qv)),
                      note="CPU evaluation on WRF primitives, not an actual WOOF replay")
        a.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result["wrf_order"]["different_words"] == 0 else 1
    if a.woof_dump is None:
        p.error("woof_dump is required without --wrf-only")
    paths = sorted(a.woof_dump.glob("*_s001_*_p00_phy_prep.npz"))
    if len(paths) != 1:
        raise ValueError(f"expected one phy_prep dump: {paths}")
    with np.load(paths[0], allow_pickle=False) as z:
        pressure, alt = z["p"], z["alt"]
        qv, current_exner, current_rho = z["x_qv"], z["x_exner"], z["x_rho"]
    f = np.float32
    libm = ctypes.CDLL("libm.so.6")
    powf = libm.powf
    powf.argtypes, powf.restype = [ctypes.c_float, ctypes.c_float], ctypes.c_float
    quotient = np.asarray(pressure / f(c.P0), dtype=np.float32)
    predicted_exner = np.fromiter((powf(float(x), float(f(c.RCP)))
                                  for x in quotient.flat), dtype=np.float32,
                                 count=pressure.size).reshape(pressure.shape)
    predicted_rho = (f(1.0) / alt) * (f(1.0) + qv)
    result = {"schema": "gpuwm.phy-prep-operation-audit/v1", "step": 1,
              "reference_powf": "libm.so.6 on the scalar WRF build host",
              "points": []}
    for name, current, predicted in (("exner", current_exner, predicted_exner),
                                     ("rho", current_rho, predicted_rho)):
        reference = wrf_array(a.wrf_dump, "p00_phy_prep", name, current.shape)
        result["points"].append(dict(field=name,
            current=compare(reference, current), hypothesis=compare(reference, predicted)))
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if all(p["hypothesis"]["different_words"] == 0 for p in result["points"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
