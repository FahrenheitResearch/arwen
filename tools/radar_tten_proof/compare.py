"""Grade the device builder against NOAA's compiled answers.

    python compare.py --cases <dir from cases.py> --receipt <json>

For every case: the boundary-layer top and the cone-filled reflectivity
must be the same bits; every point of the tendency slot must be in the same
class (no coverage, or a tendency, below the top; the flag value on top)
and within one float32 unit in the last place.  The count of points that
are not bit-identical is reported, never hidden behind the tolerance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ulp_distance(a, b):
    """Units in the last place between two float32 arrays, counted through
    zero when the signs differ."""
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2147483648) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2147483648) - ib, ib)
    return np.abs(ia - ib)


def _classes(slot):
    """Below the top: 0 = no coverage (-20), 1 = a tendency."""
    return np.where(slot[:-1] == np.float32(-20.0), 0, 1)


def _load(folder, name, shape):
    return np.fromfile(folder / f"{name}.bin", dtype=np.float32).reshape(shape)


def grade_case(folder: Path):
    import cupy as cp
    from gpuwm.da import radar_tten

    meta = json.loads((folder / "case.json").read_text())
    shape = (meta["nz"], meta["ny"], meta["nx"])
    inputs = {name: _load(folder, name, shape)
              for name in ("ref", "t", "q", "p", "h")}
    want_pblh = _load(folder, "pblh", shape[1:])
    want_cone = _load(folder, "refcone", shape)
    want = _load(folder, "tten", shape)
    config = radar_tten.RadarTtenConfig(
        convection_only=bool(meta["convection_only"]))
    keep = {}
    slot, receipt = radar_tten.build_tendency(
        cp.asarray(inputs["ref"]), cp.asarray(inputs["t"]),
        cp.asarray(inputs["p"]), cp.asarray(inputs["q"]),
        cp.asarray(inputs["h"]), config, intermediates=keep)
    got = cp.asnumpy(slot)
    got_pblh = cp.asnumpy(keep["pblh"])
    got_cone = cp.asnumpy(keep["ref_cone"])

    ulp = _ulp_distance(got, want)
    class_mismatch = int(np.count_nonzero(_classes(got) != _classes(want)))
    flag_mismatch = int(np.count_nonzero(got[-1] != want[-1]))
    below = want[:-1]
    result = {
        "case": folder.name,
        "shape": list(shape),
        "convection_only": bool(meta["convection_only"]),
        "points": int(got.size),
        "pblh_not_bit_identical": int(np.count_nonzero(
            got_pblh.view(np.int32) != want_pblh.view(np.int32))),
        "cone_not_bit_identical": int(np.count_nonzero(
            got_cone.view(np.int32) != want_cone.view(np.int32))),
        "class_mismatches": class_mismatch,
        "flag_mismatches": flag_mismatch,
        "tendency_not_bit_identical": int(np.count_nonzero(ulp)),
        "tendency_max_ulp": int(ulp.max()),
        "oracle_census": {
            "points_no_coverage": int(np.count_nonzero(
                below == np.float32(-20.0))),
            "points_zero": int(np.count_nonzero(below == 0.0)),
            "points_heated": int(np.count_nonzero(below > 0.0)),
            "points_cooled": int(np.count_nonzero(
                (below < 0.0) & (below > -1.0))),
            "points_at_cap": int(np.count_nonzero(
                below == np.float32(0.01))),
            "points_within_1e6_of_cap": int(np.count_nonzero(
                np.abs(below.astype(np.float64) - 0.01) < 1e-6)),
            "max_tendency": float(below.max()),
            "flag_no_information": int(np.count_nonzero(want[-1] == -10.0)),
            "flag_no_convection": int(np.count_nonzero(want[-1] == 0.0)),
            "flag_convection_nearby": int(np.count_nonzero(want[-1] == 1.0)),
            "cone_changed_points": int(np.count_nonzero(
                want_cone != inputs["ref"])),
            "pblh_below_level_7": int(np.count_nonzero(want_pblh < 6.5)),
            "pblh_at_or_above_level_7": int(np.count_nonzero(
                want_pblh >= 6.5)),
        },
        "device_receipt": receipt,
    }
    result["pass"] = (result["pblh_not_bit_identical"] == 0
                      and result["cone_not_bit_identical"] == 0
                      and class_mismatch == 0 and flag_mismatch == 0
                      and result["tendency_max_ulp"] <= 1)
    return result


def grade_smooth(folder: Path):
    import cupy as cp
    from gpuwm.da import radar_tten

    meta = json.loads((folder / "case.json").read_text())
    shape = (meta["ny"], meta["nx"])
    field = np.fromfile(folder / "smooth_in.bin",
                        dtype=np.float64).reshape(shape)
    out = {"case": "smooth", "shape": list(shape), "passes": {}}
    ok = True
    for passes, name in meta["passes"].items():
        want = np.fromfile(folder / name, dtype=np.float64).reshape(shape)
        got = cp.asnumpy(radar_tten.smooth(
            cp.asarray(field[None]), passes=int(passes)))[0]
        differing = int(np.count_nonzero(
            got.view(np.int64) != want.view(np.int64)))
        out["passes"][passes] = {"not_bit_identical": differing,
                                 "points": int(got.size)}
        ok = ok and differing == 0
    out["pass"] = ok
    return out


def grade_vinterp(folder: Path):
    import cupy as cp
    from gpuwm.da import radar_tten

    meta = json.loads((folder / "case.json").read_text())
    shape = (meta["nz"], meta["ny"], meta["nx"])
    mosaic = np.fromfile(folder / "mosaic.bin", dtype=np.float32).reshape(
        (meta["levels"],) + shape[1:])
    h = np.fromfile(folder / "h.bin", dtype=np.float32).reshape(shape)
    zh = np.fromfile(folder / "zh.bin", dtype=np.float32).reshape(shape[1:])
    want = np.fromfile(folder / "vinterp.bin", dtype=np.float32).reshape(shape)
    got = cp.asnumpy(radar_tten.vinterp_mosaic(
        cp.asarray(mosaic), cp.asarray(h), cp.asarray(zh)))
    differing = int(np.count_nonzero(got.view(np.int32) != want.view(np.int32)))
    return {"case": folder.name, "shape": list(shape),
            "levels": meta["levels"], "points": int(got.size),
            "not_bit_identical": differing,
            "oracle_census": {
                "valid": int(np.count_nonzero(np.abs(want) < 90.0)),
                "no_echo": int(np.count_nonzero(want == -99.0)),
                "no_coverage": int(np.count_nonzero(want == -99999.0))},
            "pass": differing == 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--build", type=Path, default=None,
                        help="the oracle's BUILD.txt, copied into the receipt")
    args = parser.parse_args()
    import cupy as cp

    results = []
    for folder in sorted(p for p in args.cases.iterdir() if p.is_dir()):
        if folder.name == "smooth":
            results.append(grade_smooth(folder))
        elif folder.name.startswith("vinterp-"):
            results.append(grade_vinterp(folder))
        else:
            results.append(grade_case(folder))
        print(json.dumps({k: v for k, v in results[-1].items()
                          if k not in ("device_receipt", "oracle_census")}),
              flush=True)
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    receipt = {
        "schema": "gpuwm-da.radar-tten-oracle.v1",
        "device": props["name"].decode(),
        "cupy": cp.__version__,
        "build": (args.build.read_text() if args.build is not None
                  and args.build.exists() else None),
        "cases": results,
        "pass": all(r["pass"] for r in results),
    }
    args.receipt.write_text(json.dumps(receipt, indent=1))
    print("PASS" if receipt["pass"] else "FAIL")
    raise SystemExit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
