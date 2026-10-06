"""The committed fork fixture check: the port's ``wrf_39_noaa`` generation
against the operational WRF 3.9 fork's own answers, on the host.

usage:
    python fork_fixture_check.py [--snow-fall blend|wrf_39_noaa]

Inputs are the 42 saved real-data columns of
tests/data/thompson_real_columns_wrf461.npz (``col_*``, dt = 5 s).  The
answers are tests/data/thompson_real_columns_wrf39_fork.npz: what the
unmodified fork module (NOAA-EMC/HRRR v4.1.21, built and run by
tools/thompson_fork_oracle/build.sh and fork_column_parity.py) returned for
those columns.  The port runs through tools/thompson_real_column_parity's
host backend, so the fork's process tables must be staged where
gpuwm.physics_compat.thompson_fork_table_root finds them.

Prints one JSON document: per field the compared cells and the cells beyond
1e-2 relative, and for each beyond cell its column, level and input
temperature, plus the largest reflectivity difference in dB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "thompson_real_column_parity"))
sys.path.insert(0, str(ROOT))

import real_column_parity as R  # noqa: E402  (installs the host backend)

INPUTS = ROOT / "tests" / "data" / "thompson_real_columns_wrf461.npz"
ANSWERS = ROOT / "tests" / "data" / "thompson_real_columns_wrf39_fork.npz"

FLOOR = {
    "qv": 0.0, "qc": 1.0e-12, "qr": 1.0e-12, "qi": 1.0e-12, "qs": 1.0e-12,
    "qg": 1.0e-12, "ni": 1.0, "nr": 1.0, "nc": 1.0, "nwfa": 1.0,
    "nifa": 1.0, "th": 0.0, "re_cloud": 0.0, "re_ice": 0.0, "re_snow": 0.0,
    "rainnc": 0.0, "snownc": 0.0, "graupelnc": 0.0,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snow-fall", default="wrf_39_noaa",
                    choices=("blend", "wrf_39_noaa"))
    args = ap.parse_args(argv)

    z = np.load(INPUTS)
    cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
    dt = float(z["dt"])
    answers = np.load(ANSWERS)
    inp = R.prepare(cols, 28)
    port = R.run_port(inp, dt, mp=28, thompson_version="wrf_39_noaa",
                      thompson_fork_snow_fall=args.snow_fall)["final"]
    result = {"snow_fall": args.snow_fall, "fields": {}}
    temperature = inp["T"]
    for name, floor in FLOOR.items():
        fork = np.asarray(answers[f"fork_{name}"], np.float64)
        mine = np.asarray(port[name], np.float64)
        if name.startswith("re_"):
            fork = fork * 1.0e6      # the port's radii are microns
        mask = (np.abs(mine) > floor) | (np.abs(fork) > floor)
        scale = np.maximum(np.abs(mine), np.abs(fork))
        rel = np.where(mask, np.abs(mine - fork)
                       / np.where(scale > 0, scale, 1.0), 0.0)
        beyond = np.argwhere(rel > 1.0e-2)
        cells = []
        for where in beyond[:200]:
            where = tuple(int(i) for i in where)
            cells.append({
                "at": list(where),
                "t_in": (float(temperature[where]) if len(where) == 2
                         else None)})
        result["fields"][name] = {"n": int(mask.sum()),
                                  "n_beyond_1e-2": int(len(beyond)),
                                  "cells": cells}
    pr = np.asarray(port["refl"], np.float64)
    fr = np.asarray(answers["fork_refl"], np.float64)
    echo = (pr > -10.0) | (fr > -10.0)
    result["refl"] = {"n_echo": int(echo.sum()),
                      "max_abs_db": float(np.abs(pr - fr)[echo].max())}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
