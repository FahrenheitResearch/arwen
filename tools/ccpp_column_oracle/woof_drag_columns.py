#!/usr/bin/env python3
"""WOOF's own gwd_opt = 3 tendencies on the oracle columns (GPU).

Runs the engine's production GSL drag kernel (gpuwm.core.terrain_drag
launch_gwdo_gsl) on the columns of gpuwm/data/terrain_drag/oracle/gwdo/inputs
at the five grid lengths the WRF oracle uses, and writes the momentum
tendency increments and the eight drag components it returns, one npz per
grid length.  The call is the one tests/test_terrain_drag_wrf471_parity.py
holds to WRF v4.7.1 at 0 ULP; nothing is computed here but the launch.

    python tools/ccpp_column_oracle/woof_drag_columns.py OUT_DIR [--dt SECONDS]

Without --dt each grid length runs at its WRF fixture's own step, which is
the step the parity test holds to WRF.  With --dt every grid length runs at
that step (RRFS v1 runs 36 s); the npz records the step it ran.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ORACLE = ROOT / "gpuwm" / "data" / "terrain_drag" / "oracle"
GRID_LENGTHS = (1000, 3000, 5000, 9000, 15000)


def _kij(a):
    return np.ascontiguousarray(np.asarray(a, np.float32).T[:, None, :])


def _row(a):
    return np.ascontiguousarray(np.asarray(a)[None, :])


def _planes(a):
    return [np.ascontiguousarray(np.asarray(a, np.float32)[:, m][None, :])
            for m in range(4)]


def main(argv=None) -> int:
    import cupy as cp

    from gpuwm.core.terrain_drag import (GSL_DIAGNOSTIC_NAMES, gsl_kpblmax,
                                         launch_gwdo_gsl)
    from gpuwm.verify.urban_oracle import load_case

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    inputs = load_case(ORACLE / "gwdo" / "inputs")
    for dx in GRID_LENGTHS:
        fixture_dt = float(load_case(ORACLE / "gwdo_gsl" / f"dx{dx}")["dt"])
        dt = fixture_dt if args.dt is None else float(args.dt)
        atm = {k: cp.asarray(v) for k, v in {
            "u": _kij(inputs["u3d"]), "v": _kij(inputs["v3d"]),
            "temperature": _kij(inputs["t3d"]), "qv": _kij(inputs["qv3d"]),
            "pressure": _kij(inputs["p3d"]), "exner": _kij(inputs["pi3d"]),
            "p_interface": _kij(inputs["p3di"]), "z": _kij(inputs["z"]),
            "dz": _kij(inputs["dz"])}.items()}
        du0 = _kij(inputs["rublten0"])
        dv0 = _kij(inputs["rvblten0"])
        du = cp.asarray(du0)
        dv = cp.asarray(dv0)
        diag = {}
        launch_gwdo_gsl(
            atm, du, dv,
            ls={"var": _row(inputs["var2d"]), "con": _row(inputs["oc12d"]),
                "oa": _planes(inputs["oa"]), "ol": _planes(inputs["ol"])},
            ss={"var": _row(inputs["var2dss"]), "con": _row(inputs["oc12dss"]),
                "oa": _planes(inputs["oass"]), "ol": _planes(inputs["olss"])},
            sina=_row(inputs["sina"]), cosa=_row(inputs["cosa"]),
            xland=_row(inputs["xland"]), br=_row(inputs["br"]),
            pblh=_row(inputs["pblh"]), kpbl=_row(inputs["kpbl"]),
            kpblmax=gsl_kpblmax(inputs["znu"]), dx=float(dx), dt=dt,
            diagnostics=diag)
        # (nz, 1, ncol) -> (ncol, nz), the oracle's order.  The increment is
        # what the scheme added: the kernel accumulates into RUBLTEN.
        record = {
            "dt": np.float64(dt),
            "fixture_dt": np.float64(fixture_dt),
            "du": (cp.asnumpy(du) - du0)[:, 0, :].T,
            "dv": (cp.asnumpy(dv) - dv0)[:, 0, :].T,
        }
        for comp in GSL_DIAGNOSTIC_NAMES:
            for axis in ("x", "y"):
                record[f"dtau{axis}_{comp}"] = cp.asnumpy(
                    diag[f"DTAU{axis.upper()}3D_{comp}"])[:, 0, :].T
        np.savez(out / f"woof_dx{dx}.npz", **record)
    print(f"WOOF gwd_opt 3 tendencies for {len(GRID_LENGTHS)} grid lengths in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
