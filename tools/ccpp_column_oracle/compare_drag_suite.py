#!/usr/bin/env python3
"""Compare CCPP's GSL drag tendencies with WOOF's and WRF's on the same columns.

    python tools/ccpp_column_oracle/compare_drag_suite.py \
        CCPP_BUILD_DIR CASE WOOF_DIR [--json OUT]

CCPP_BUILD_DIR is build.sh's output (r8/fixtures/CASE and r4/fixtures/CASE);
CASE names one IMX:DELTIM run, for example ``imx15800_dt36``.  WOOF_DIR
holds woof_drag_columns.py's npz files.  The WRF v4.7.1 fixture the engine
is held to is read from the tree as a third column whenever WOOF ran at the
fixture's own step.  For every grid length, both cdmbgwd settings and both
CCPP precisions it reports, per momentum tendency and per drag component:
the largest absolute difference, the relative L2 difference (difference
norm over the WOOF norm) and how many of the 64 columns differ by more than
1e-6 of WOOF's largest magnitude.  For form drag it also reports the ratio
of the two largest form-drag tendencies and, per column, the ratio of the
vertically integrated form stress (sum over levels of the tendency times
the layer's pressure depth over g, in N m-2), over the columns where WOOF's
is non-zero.  This is a measurement tool: it computes no tendency itself.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NCOL, NZ = 64, 40
G = 9.81
GRID_LENGTHS = (1000, 3000, 5000, 9000, 15000)
#: WOOF and WRF name the mesoscale component "ls", CCPP "ms".
COMPONENTS = (("ls", "ms"), ("bl", "bl"), ("ss", "ss"), ("fd", "fd"))
CDMB = {1: "cdmbgwd (1, 1)", 2: "cdmbgwd (3.5, 1.0), RRFS v1's value"}


def _ccpp(build: pathlib.Path, prec: str, case: str, tag: str, name: str) -> np.ndarray:
    raw = np.fromfile(build / prec / "fixtures" / case / f"{tag}_{name}.bin", "<f8")
    return raw.reshape((NCOL, NZ), order="F")


def _del(build: pathlib.Path, prec: str, case: str) -> np.ndarray:
    raw = np.fromfile(build / prec / "fixtures" / case / "del.bin", "<f8")
    return raw.reshape((NCOL, NZ), order="F")


def _stats(ref: np.ndarray, got: np.ndarray) -> dict:
    ref = np.asarray(ref, np.float64)
    got = np.asarray(got, np.float64)
    diff = got - ref
    scale = float(np.abs(ref).max())
    norm = float(np.linalg.norm(ref))
    floor = 1e-6 * scale if scale > 0 else 0.0
    return {
        "max_abs_ref": scale,
        "max_abs_diff": float(np.abs(diff).max()),
        "rel_l2": (float(np.linalg.norm(diff)) / norm) if norm > 0 else
                  (0.0 if not np.any(diff) else None),
        "columns_differing": int(np.any(np.abs(diff) > floor, axis=1).sum()),
    }


def _form_drag(woof: dict, ccpp_x: np.ndarray, ccpp_y: np.ndarray,
               dp: np.ndarray) -> dict:
    wx = np.asarray(woof["dtaux_fd"], np.float64)
    wy = np.asarray(woof["dtauy_fd"], np.float64)
    w_max = float(np.abs(wx).max())
    c_max = float(np.abs(ccpp_x).max())
    w_tau = np.hypot((wx * dp).sum(axis=1), (wy * dp).sum(axis=1)) / G
    c_tau = np.hypot((ccpp_x * dp).sum(axis=1), (ccpp_y * dp).sum(axis=1)) / G
    have = w_tau > 0
    ratio = c_tau[have] / w_tau[have]
    return {
        "largest_dtaux_woof": w_max,
        "largest_dtaux_ccpp": c_max,
        "largest_ratio": (c_max / w_max) if w_max > 0 else None,
        "columns_with_woof_form_stress": int(have.sum()),
        "columns_with_ccpp_form_stress": int((c_tau > 0).sum()),
        "column_stress_ratio_min": float(ratio.min()) if ratio.size else None,
        "column_stress_ratio_median": float(np.median(ratio)) if ratio.size else None,
        "column_stress_ratio_max": float(ratio.max()) if ratio.size else None,
        "all_columns_stress_ratio": (float(c_tau[have].sum() / w_tau[have].sum())
                                     if ratio.size else None),
    }


def main(argv=None) -> int:
    from gpuwm.verify.urban_oracle import load_case

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ccpp_build", type=pathlib.Path)
    parser.add_argument("case")
    parser.add_argument("woof_dir", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args(argv)
    match = re.fullmatch(r"imx(\d+)_dt([0-9.]+)", args.case)
    if not match:
        parser.error("CASE is imx<IMX>_dt<DELTIM>, as build.sh names it")
    imx, deltim = int(match.group(1)), float(match.group(2))

    oracle = ROOT / "gpuwm" / "data" / "terrain_drag" / "oracle"
    inputs = load_case(oracle / "gwdo" / "inputs")
    report = {"columns": NCOL, "levels": NZ, "case": args.case,
              "ccpp_imx": imx, "ccpp_deltim_s": deltim, "rows": []}
    for dx in GRID_LENGTHS:
        woof = dict(np.load(args.woof_dir / f"woof_dx{dx}.npz"))
        woof_dt = float(woof["dt"])
        row = {"dx_m": dx, "woof_dt_s": woof_dt}
        if woof_dt == float(woof["fixture_dt"]):
            wrf = load_case(oracle / "gwdo_gsl" / f"dx{dx}")
            wrf_du = np.asarray(wrf["rublten"], np.float64) - inputs["rublten0"]
            wrf_dv = np.asarray(wrf["rvblten"], np.float64) - inputs["rvblten0"]
            row["woof_vs_wrf"] = {"du": _stats(wrf_du, woof["du"]),
                                  "dv": _stats(wrf_dv, woof["dv"])}
        for m, label in CDMB.items():
            tag = f"dx{dx}_cdmb{m}"
            for prec in ("r8", "r4"):
                key = f"ccpp_{prec}_cdmb{m}_vs_woof"
                get = lambda name: _ccpp(args.ccpp_build, prec, args.case, tag, name)  # noqa: E731
                entry = {"cdmbgwd": label,
                         "du": _stats(woof["du"], get("du")),
                         "dv": _stats(woof["dv"], get("dv"))}
                for woof_name, ccpp_name in COMPONENTS:
                    for axis in ("x", "y"):
                        entry[f"dtau{axis}_{woof_name}"] = _stats(
                            woof[f"dtau{axis}_{woof_name}"], get(f"dtau{axis}_{ccpp_name}"))
                entry["form_drag"] = _form_drag(
                    woof, get("dtaux_fd"), get("dtauy_fd"),
                    _del(args.ccpp_build, prec, args.case))
                row[key] = entry
        report["rows"].append(row)

    print(f"case {args.case}: CCPP imx {imx}, deltim {deltim:g} s")
    for row in report["rows"]:
        print(f"dx {row['dx_m']} m (WOOF at {row['woof_dt_s']:g} s)")
        if "woof_vs_wrf" in row:
            w = row["woof_vs_wrf"]
            print(f"  WOOF vs WRF v4.7.1 fixture: du max|diff| {w['du']['max_abs_diff']:.3e}"
                  f"  dv max|diff| {w['dv']['max_abs_diff']:.3e}")
        for key in sorted(k for k in row if k.startswith("ccpp_")):
            e = row[key]
            parts = []
            for name in ("du", "dv") + tuple(
                    f"dtau{a}_{c}" for c, _ in COMPONENTS for a in ("x",)):
                s = e[name]
                relative = ("undefined (zero reference)" if s["rel_l2"] is None
                            else f"{s['rel_l2']:.2e}")
                parts.append(f"{name} rel {relative} max {s['max_abs_diff']:.2e}"
                             f" cols {s['columns_differing']}")
            print(f"  {key}: " + "; ".join(parts))
            f = e["form_drag"]
            if f["largest_ratio"] is not None:
                print(f"    form drag: largest {f['largest_dtaux_ccpp']:.4g} / "
                      f"{f['largest_dtaux_woof']:.4g} = {f['largest_ratio']:.2f}; "
                      f"column stress ratio {f['column_stress_ratio_min']:.2f} to "
                      f"{f['column_stress_ratio_max']:.2f}, median "
                      f"{f['column_stress_ratio_median']:.2f}, all columns "
                      f"{f['all_columns_stress_ratio']:.2f} over "
                      f"{f['columns_with_woof_form_stress']} columns")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
