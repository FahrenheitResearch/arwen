#!/usr/bin/env python3
"""Run WRF v4.6.1's Fortran on the inputs one GPU run used, and compare
every output bitwise.

The WRF driver is handed the raw columns plus the Exner function, layer
depths and lower-face heights the adapter formed on the device
(``in_pii``, ``in_dz``, ``in_hgt`` in the GPU output), so both codes run
the scheme on identical float32 inputs.  Every one of the 23 driver outputs
is compared bit for bit: the eleven moments, theta, reflectivity, the three
effective radii and the seven surface accumulations.

usage: compare.py COLUMNS.npz GPU_OUT.npz WRF_BUILD_DIR SUMMARY.json
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oracle_io as io  # noqa: E402

f32 = np.float32


def wrf_answers(cols, gpu, wrf_build, dt):
    inp = {"p": cols["p"], "th": cols["th"], "w": cols["w"],
           "pii": gpu["in_pii"], "dz": gpu["in_dz"], "hgt": gpu["in_hgt"],
           "nwfa2d": cols["nwfa2d"], "nifa2d": cols["nifa2d"]}
    for s in io.SPECIES:
        inp[s] = cols[s]
    binary = Path(wrf_build) / "pristine" / "run_columns_aero"
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.bin", Path(tmp) / "out.bin"
        io.write_wrf_input(fin, inp, dt)
        io.run_wrf(binary, Path(wrf_build) / "run", fin, fout)
        out = io.read_wrf_output(fout, *cols["p"].shape)
        return {k: io.wrf_view(k, v) for k, v in out.items()}


def summarize(cols, gpu, wrf):
    regimes = cols["regime"]
    per_field, per_regime = {}, {}
    col_bad = np.zeros(len(regimes), bool)
    for name in io.FIELDS:
        res, bad = io.bitwise(gpu[name], wrf[name])
        per_field[name] = res
        colmask = bad if bad.ndim == 1 else bad.any(axis=1)
        col_bad |= colmask
        if res["differ"]:
            i = res["worst"]["index"]
            res["worst"]["column"] = str(cols["labels"][i[0]])
    for r in np.unique(regimes):
        m = regimes == r
        worst = 0
        for name in io.FIELDS:
            d = io.ulp_distance(gpu[name][m], wrf[name][m])
            same = (np.asarray(gpu[name][m], f32).view(np.int32)
                    == np.asarray(wrf[name][m], f32).view(np.int32))
            worst = max(worst, int(np.where(same, 0, d).max()))
        per_regime[str(r)] = {"columns": int(m.sum()),
                              "columns_bit_identical":
                                  int((~col_bad[m]).sum()),
                              "max_ulp": worst}
    total = sum(v["cells"] for v in per_field.values())
    differ = sum(v["differ"] for v in per_field.values())
    return {"columns": int(len(regimes)), "levels": int(cols["p"].shape[1]),
            "fields": len(io.FIELDS), "cells": total, "cells_differ": differ,
            "columns_bit_identical": int((~col_bad).sum()),
            "max_ulp": max(v["max_ulp"] for v in per_field.values()),
            "max_abs_diff": {k: v["max_abs_diff"] for k, v in
                             per_field.items()},
            "per_field": per_field, "per_regime": per_regime,
            "differing_columns": [str(x) for x in cols["labels"][col_bad]]}


def tendency_share(cols, gpu, wrf):
    """Largest |GPU - WRF| over the column-level largest |WRF one-step
    change| per moment: the measure the whole-model exactness report used
    for QICE and QNICE ("difference as a share of WRF's own one-step
    tendency")."""
    out = {}
    for s in io.SPECIES + ("th",):
        d = np.abs(np.asarray(gpu[s], np.float64) - np.asarray(wrf[s],
                                                              np.float64))
        ten = np.abs(np.asarray(wrf[s], np.float64)
                     - np.asarray(cols[s], np.float64))
        scale = ten.max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(scale > 0, d / scale, np.where(d > 0, np.inf,
                                                            0.0))
        out[s] = {"max_share_of_column_tendency": float(share.max()),
                  "domain_share": float(d.max() / ten.max())
                  if ten.max() > 0 else 0.0}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 4:
        raise SystemExit(__doc__)
    cols = io.load(argv[0])
    gpu = io.load(argv[1])
    meta = json.loads(str(gpu["meta"]))
    # The adapter's own th sum must reproduce the raw input bit for bit.
    wrf = wrf_answers(cols, gpu, argv[2], meta["dt"])
    np.savez(Path(argv[3]).with_suffix(".wrf.npz"), **wrf)
    s = summarize(cols, gpu, wrf)
    s["tendency_share"] = tendency_share(cols, gpu, wrf)
    s["gpu"] = meta
    Path(argv[3]).write_text(json.dumps(s, indent=1) + "\n")
    mode = "STRICT" if meta["strict"] else "DEFAULT"
    print(f"{mode} dt={meta['dt']}: {s['columns']} columns x {s['levels']} "
          f"levels, {s['fields']} fields: {s['cells_differ']} of "
          f"{s['cells']} cells differ; columns bit-identical "
          f"{s['columns_bit_identical']}/{s['columns']}; max ULP "
          f"{s['max_ulp']}")
    for name, r in s["per_field"].items():
        w = r.get("worst", {})
        print(f"  {name:11s} differ {r['differ']:6d}/{r['cells']:6d}  "
              f"max_ulp {r['max_ulp']:>11d}  max_abs {r['max_abs_diff']:.3e}"
              + (f"  worst col '{w.get('column')}' lev "
                 f"{w['index'][1] if len(w['index']) > 1 else '-'} "
                 f"gpu {w['port']:.9g} wrf {w['wrf']:.9g}" if w else ""))
    for r, v in s["per_regime"].items():
        print(f"  [{r}] {v['columns_bit_identical']}/{v['columns']} columns "
              f"bit-identical, max ULP {v['max_ulp']}")


if __name__ == "__main__":
    main()
