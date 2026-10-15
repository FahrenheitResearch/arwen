#!/usr/bin/env python3
"""Run WOOF's Ferrier-Aligo kernel over a column-oracle input set.

usage: run_woof_ferrier.py COLUMNS_INPUTS_CSV OUT_DIR

Reads ``columns_inputs.csv`` (make_columns.py / make_columns_refute.py
format) and does, through the engine's own launcher
(gpuwm.core.fer_hires.device_tables / launch_fer_hires), exactly what
``run_ferrier.F90`` does with WRF's Fortran: per column, fer_hires_init at
that column's time step, then FER_HIRES once per step for the column's
step count with th, qv, qc, qr, qi, F_ICE, F_RAIN, F_RIMEF and RAINNC
carried, writing ``levels.csv`` and ``surface.csv`` for step 1 and the last
step.  ``compare_ferrier.py`` grades the two directories.

Columns sharing (dt, domain id, step count) run as one launch, one column per
thread; no column reads another.  Strict arithmetic is the caller's
``GPUWM_WRF_EXACT=1`` (it must be set before gpuwm is imported).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FIELDS_IN = ("p", "pi", "th", "qv", "qc", "qr", "qi", "dz8w", "rho",
             "f_ice", "f_rain", "f_rimef")
FIELDS_OUT = ("th", "qv", "qt", "qc", "qr", "qi", "f_ice", "f_rain",
              "f_rimef")


def _f(word: str) -> np.float32:
    return np.uint32(int(word, 16)).view(np.float32)


def _hex(x) -> str:
    return f"{int(np.float32(x).view(np.uint32)):08X}"


def read_columns(path: Path):
    cols: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.startswith("C,"):
            _, idx, regime, gid, nsteps, dt, dx, rainnc = line.split(",")
            cols[int(idx)] = {"regime": regime, "gid": int(gid),
                              "nsteps": int(nsteps), "dt": _f(dt),
                              "rainnc": _f(rainnc), "levels": {}}
        elif line.startswith("L,"):
            parts = line.split(",")
            col, k = int(parts[1]), int(parts[2])
            cols[col]["levels"][k] = [_f(w) for w in parts[3:]]
    return cols


def main(inputs: Path, out: Path) -> None:
    import cupy as cp

    from gpuwm.core.fer_hires import device_tables, launch_fer_hires

    cols = read_columns(inputs)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for idx, c in cols.items():
        key = (int(np.float32(c["dt"]).view(np.uint32)), c["gid"],
               c["nsteps"])
        groups[key].append(idx)
    levels_rows: dict[tuple, str] = {}
    surface_rows: dict[tuple, str] = {}
    for (_dtbits, gid, nsteps), members in sorted(groups.items()):
        dt = float(cols[members[0]]["dt"])
        nz = len(cols[members[0]]["levels"])
        ncol = len(members)
        host = {f: np.zeros((nz, 1, ncol), np.float32) for f in FIELDS_IN}
        for jcol, idx in enumerate(members):
            lv = cols[idx]["levels"]
            for k in range(nz):
                for f, v in zip(FIELDS_IN, lv[k + 1]):
                    host[f][k, 0, jcol] = v
        d = {f: cp.asarray(host[f]) for f in FIELDS_IN}
        qt = cp.zeros((nz, 1, ncol), cp.float32)
        rainnc = cp.asarray(np.array(
            [[cols[i]["rainnc"] for i in members]], np.float32))
        rainncv = cp.zeros((1, ncol), cp.float32)
        sr = cp.zeros((1, ncol), cp.float32)
        tab = device_tables(dt)
        for step in range(1, nsteps + 1):
            launch_fer_hires(d["th"], d["qv"], d["qc"], d["qr"], d["qi"], qt,
                             d["f_ice"], d["f_rain"], d["f_rimef"], d["p"],
                             d["pi"], d["rho"], d["dz8w"], rainnc, rainncv,
                             sr, tab, dt, gid)
            if step in (1, nsteps):
                out_h = {f: cp.asnumpy(qt if f == "qt" else d[f])
                         for f in FIELDS_OUT}
                rn, rv, s = (cp.asnumpy(rainnc), cp.asnumpy(rainncv),
                             cp.asnumpy(sr))
                for jcol, idx in enumerate(members):
                    for k in range(nz):
                        words = [_hex(out_h[f][k, 0, jcol]) for f in FIELDS_OUT]
                        levels_rows[(idx, step, k + 1)] = (
                            f"{idx},{step},{k + 1}," + ",".join(words))
                    surface_rows[(idx, step)] = (
                        f"{idx},{step},{_hex(rn[0, jcol])},"
                        f"{_hex(rv[0, jcol])},{_hex(s[0, jcol])}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "levels.csv", "w", newline="\n") as fl:
        fl.write("col,step,k," + ",".join(FIELDS_OUT) + "\n")
        for key in sorted(levels_rows):
            fl.write(levels_rows[key] + "\n")
    with open(out / "surface.csv", "w", newline="\n") as fs:
        fs.write("col,step,rainnc,rainncv,sr\n")
        for key in sorted(surface_rows):
            fs.write(surface_rows[key] + "\n")
    print(f"wrote {len(cols)} columns in {len(groups)} launches to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
