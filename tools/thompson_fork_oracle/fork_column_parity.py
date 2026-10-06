"""The port's Thompson (mp_physics=28) against the operational WRF 3.9 fork's
own Fortran, on saved model columns, on the CPU.

usage:
    python fork_column_parity.py COLUMNS.npz FORK_BUILD_DIR OUT.json
        [--version wrf_39_noaa] [--dt SECONDS] [--max-cols N]

COLUMNS.npz is either a committed fixture (``col_*`` arrays plus ``dt``,
tests/data/thompson_real_columns_wrf461.npz) or a column file written by
tools/thompson_real_column_parity/extract_columns.py.  FORK_BUILD_DIR is
the output of tools/thompson_fork_oracle/build.sh (run_columns_fork, the
fork's generated tables and CCN_ACTIVATE.BIN).

The port runs through tools/thompson_real_column_parity's host backend: the
exact kernel source nvrtc receives, compiled for the host, behind the
production adapter.  ``--version`` picks RunConfig.thompson_version for the
port side; ``wrf_461`` measures how far the WRF v4.6.1 transcription sits
from the fork (the "before"), ``wrf_39_noaa`` the fork version (the
"after").  The fork oracle side is always the unmodified fork module.

The comparison is the final state of one call, field by field: a cell is
compared where either code's value passes the field's floor; reported are
the count of compared cells, the counts beyond 1e-4 and 1e-2 relative, the
largest relative difference and where it sits, and for reflectivity the
largest absolute difference in dB over cells where either side is above
-10 dBZ.  Surface precipitation is compared per column.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARITY = HERE.parent / "thompson_real_column_parity"
sys.path.insert(0, str(PARITY))
sys.path.insert(0, str(HERE.parents[1]))

import real_column_parity as R  # noqa: E402  (installs the host backend)

f32 = np.float32

#: run_columns_fork.F90's output streams.
FORK_OUT3 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
             "nifa", "th", "refl", "re_cloud", "re_ice", "re_snow")
FORK_OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
             "graupelncv", "sr", "frain")

#: Below these a value counts as absent (masses kg/kg, numbers per kg,
#: radii metres, theta K is always compared).
FLOOR = {
    "qv": 0.0, "qc": 1.0e-12, "qr": 1.0e-12, "qi": 1.0e-12, "qs": 1.0e-12,
    "qg": 1.0e-12, "ni": 1.0, "nr": 1.0, "nc": 1.0, "nwfa": 1.0,
    "nifa": 1.0, "th": 0.0, "re_cloud": 0.0, "re_ice": 0.0, "re_snow": 0.0,
}


def load(path, max_cols=None):
    z = np.load(path)
    if "col_p" in z.files:
        cols = {k[4:]: z[k] for k in z.files if k.startswith("col_")}
        dt = float(z["dt"]) if "dt" in z.files else None
    else:
        cols = {k: z[k] for k in z.files}
        dt = float(z["dt"]) if "dt" in z.files else None
    if max_cols and cols["p"].shape[0] > max_cols:
        rng = np.random.default_rng(0)
        ncol = cols["p"].shape[0]
        keep = np.sort(rng.choice(ncol, size=max_cols, replace=False))
        cols = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim >= 1
                    and v.shape[0] == ncol else v) for k, v in cols.items()}
    return cols, dt


def run_fork(build, inp, dt, work):
    work.mkdir(parents=True, exist_ok=True)
    in_path = work / "fork-in.bin"
    out_path = work / "fork-out.bin"
    R.write_wrf_input(in_path, inp, dt, mp=28)
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    done = subprocess.run([str(build / "run_columns_fork"), str(in_path),
                           str(out_path)], cwd=str(build), env=env,
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"run_columns_fork failed:\n{done.stdout}\n"
                           f"{done.stderr}")
    ncol, nz = inp["p"].shape
    raw = np.fromfile(out_path, dtype="<f4")
    want = len(FORK_OUT3) * ncol * nz + len(FORK_OUT2) * ncol
    if raw.size != want:
        raise RuntimeError(f"{out_path}: {raw.size} words, expected {want}")
    out, off = {}, 0
    for name in FORK_OUT3:
        out[name] = raw[off:off + ncol * nz].reshape(nz, ncol).T.astype(
            np.float64)
        off += ncol * nz
    for name in FORK_OUT2:
        out[name] = raw[off:off + ncol].astype(np.float64)
        off += ncol
    in_path.unlink()
    out_path.unlink()
    return out


def compare_field(port, fork, floor):
    port = np.asarray(port, np.float64)
    fork = np.asarray(fork, np.float64)
    mask = (np.abs(port) > floor) | (np.abs(fork) > floor)
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    scale = np.maximum(np.abs(port), np.abs(fork))
    rel = np.where(mask, np.abs(port - fork) / np.where(scale > 0, scale, 1),
                   0.0)
    worst = np.unravel_index(int(np.argmax(rel)), rel.shape)
    return {
        "n": n,
        "n_gt_1e-4": int((rel > 1.0e-4).sum()),
        "n_gt_1e-2": int((rel > 1.0e-2).sum()),
        "max_rel": float(rel.max()),
        "at": [int(i) for i in worst],
        "port_at": float(port[worst]), "fork_at": float(fork[worst]),
        "mean_port": float(port[mask].mean()),
        "mean_fork": float(fork[mask].mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("columns")
    ap.add_argument("build")
    ap.add_argument("out")
    ap.add_argument("--version", default="wrf_39_noaa")
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--max-cols", type=int, default=None)
    ap.add_argument("--snow-fall", default="blend",
                    choices=("blend", "wrf_39_noaa"),
                    help="RunConfig.thompson_fork_snow_fall for the port")
    ap.add_argument("--dump", default=None,
                    help="write both sides' final fields to this .npz")
    args = ap.parse_args(argv)

    cols, dt = load(args.columns, args.max_cols)
    dt = args.dt if args.dt is not None else (dt if dt is not None else 20.0)
    inp = R.prepare(cols, 28)
    work = Path(os.environ.get("TMPDIR", ".")) / "fork-parity"
    fork = run_fork(Path(args.build).resolve(), inp, dt, work)
    port = R.run_port(inp, dt, mp=28, thompson_version=args.version,
                      thompson_fork_snow_fall=args.snow_fall)["final"]
    # The port publishes effective radii in microns (gpuwm's radiation
    # contract), the fork's driver in metres.
    for name in ("re_cloud", "re_ice", "re_snow"):
        fork[name] = fork[name] * 1.0e6
    if args.dump:
        np.savez_compressed(args.dump, **{f"port_{k}": np.asarray(v)
                                          for k, v in port.items()},
                            **{f"fork_{k}": v for k, v in fork.items()},
                            **{f"in_{k}": v for k, v in inp.items()})

    result = {"columns": args.columns, "ncol": int(inp["p"].shape[0]),
              "nz": int(inp["p"].shape[1]), "dt": dt,
              "port_version": args.version,
              "port_snow_fall": args.snow_fall, "fields": {}}
    for name, floor in FLOOR.items():
        result["fields"][name] = compare_field(port[name], fork[name], floor)
    # Reflectivity in dB where either side shows an echo.
    pr, fr = port["refl"], fork["refl"]
    echo = (pr > -10.0) | (fr > -10.0)
    diff = np.abs(pr - fr)
    result["refl"] = {
        "n_echo": int(echo.sum()),
        "max_abs_db": float(diff[echo].max()) if echo.any() else 0.0,
        "n_gt_0p1_db": int((diff[echo] > 0.1).sum()),
        "n_gt_1_db": int((diff[echo] > 1.0).sum()),
        "mean_port_dbz": float(pr[echo].mean()) if echo.any() else None,
        "mean_fork_dbz": float(fr[echo].mean()) if echo.any() else None,
    }
    for name in ("rainnc", "snownc", "graupelnc", "sr"):
        result["fields"][name] = compare_field(port[name], fork[name], 0.0)
    Path(args.out).write_text(json.dumps(result, indent=1))
    bad = {k: v.get("n_gt_1e-2") for k, v in result["fields"].items()
           if v.get("n_gt_1e-2")}
    print(json.dumps({"beyond_1e-2": bad, "refl": result["refl"]}, indent=1))


if __name__ == "__main__":
    main()
