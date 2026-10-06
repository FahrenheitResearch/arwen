"""Device half of the oracle proof: NOAA's outputs against the CUDA kernels.

Reads what ``run_oracles.sh`` wrote, runs :mod:`gpuwm.da.hydrometeor_analysis`
on the same inputs on the card, and compares point by point.  The pass rule:
the same sentinel pattern everywhere, and every value within one float32 unit
in the last place; the count of points that are not bit-identical is
reported beside it.  NumPy is used for buffers and for the comparison only.

Usage: python compare.py <oracle-work-dir> <receipt.json>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SENTINELS = (-999.0, -99999.0)


def _ordered(bits: np.ndarray) -> np.ndarray:
    """float32 bit patterns mapped to a line where adjacent floats differ by 1."""
    bits = bits.astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def compare(noaa: np.ndarray, ours: np.ndarray) -> dict:
    noaa = np.ascontiguousarray(noaa, dtype=np.float32)
    ours = np.ascontiguousarray(ours, dtype=np.float32)
    sentinel = np.isin(noaa, np.float32(SENTINELS))
    ours_sentinel = np.isin(ours, np.float32(SENTINELS))
    pattern_ok = bool(np.array_equal(sentinel, ours_sentinel)
                      and np.array_equal(noaa[sentinel], ours[sentinel]))
    a = noaa.view(np.int32)
    b = ours.view(np.int32)
    differ = a != b
    ulp = np.abs(_ordered(a) - _ordered(b))
    values = ~sentinel
    return {
        "points": int(noaa.size),
        "sentinel_points": int(sentinel.sum()),
        "sentinel_pattern_identical": pattern_ok,
        "not_bit_identical": int(differ.sum()),
        "signed_zero_only": int((differ & (ulp == 0)).sum()),
        "max_ulp": int(ulp[values].max()) if values.any() else 0,
        "over_one_ulp": int((ulp[values] > 1).sum()),
        "nonfinite_ours": int((~np.isfinite(ours)).sum()),
    }


def passed(row: dict) -> bool:
    return (row["sentinel_pattern_identical"] and row["max_ulp"] <= 1
            and row["nonfinite_ours"] == 0)


def oracle_a(work: Path, ha, cp) -> dict:
    raw = (work / "case_a.bin").read_bytes()
    nx, ny, nz = np.frombuffer(raw, np.int32, 3)
    n = int(nx) * int(ny) * int(nz)
    shape = (int(nz), int(ny), int(nx))
    offset = 12
    t = np.frombuffer(raw, np.float32, n, offset).reshape(shape)
    offset += 4 * n
    p = np.frombuffer(raw, np.float32, n, offset).reshape(shape)
    offset += 4 * n
    ref = np.frombuffer(raw, np.float64, n, offset).reshape(shape)
    out = (work / "out_a.bin").read_bytes()
    head = np.frombuffer(out, np.int32, 4)
    names = ("qr_gkg", "qnr_per_kg", "qs_gkg")
    noaa = {name: np.frombuffer(out, np.float32, n, 16 + 4 * n * slot)
            .reshape(shape) for slot, name in enumerate(names)}
    started = time.perf_counter()
    got = ha.thompson_retrieval(cp.asarray(t), cp.asarray(p), cp.asarray(ref))
    cp.cuda.runtime.deviceSynchronize()
    seconds = time.perf_counter() - started
    ours = dict(zip(names, (cp.asnumpy(v) for v in got)))
    visited = (slice(1, -1), slice(1, -1), slice(1, -1))
    edges = (np.array([0, shape[0] - 1]), slice(1, -1), slice(1, -1))
    rows = {}
    for name in names:
        rows[name] = compare(noaa[name][visited], ours[name][visited])
        rows[name]["unvisited_levels_identical"] = bool(np.array_equal(
            noaa[name][edges].view(np.int32), ours[name][edges].view(np.int32)))
    return {"istatus": int(head[3]), "grid": [int(nz), int(ny), int(nx)],
            "visited_points": int(np.prod([s - 2 for s in shape])),
            "echo_points_visited": int((ref[visited] >= 0.0).sum()),
            "kernel_seconds_first_call": round(seconds, 3),
            "fields": rows,
            "pass": all(passed(r) and r["unvisited_levels_identical"]
                        for r in rows.values())}


def _case_b(path: Path):
    raw = path.read_bytes()
    lon2, lat2, nsig = (int(v) for v in np.frombuffer(raw, np.int32, 3))
    shape = (nsig, lat2, lon2)
    n = nsig * lat2 * lon2
    offset = 12
    out = {}
    for name, dtype, count in (("theta", np.float32, n), ("p_hpa", np.float32, n),
                               ("ref", np.float64, n), ("ctp", np.float32, lat2 * lon2),
                               ("qr", np.float64, n), ("nr", np.float64, n),
                               ("qs", np.float64, n), ("qg", np.float64, n)):
        array = np.frombuffer(raw, dtype, count, offset)
        offset += array.nbytes
        out[name] = array.reshape(shape if count == n else (lat2, lon2))
    return shape, out


def _out_b(path: Path, shape):
    raw = path.read_bytes()
    n = int(np.prod(shape))
    names = ("qr", "nr", "qs", "qg", "t", "p")
    return {name: np.frombuffer(raw, np.float32, n, 12 + 4 * n * slot)
            .reshape(shape) for slot, name in enumerate(names)}


def oracle_b(work: Path, ha, cp) -> dict:
    runs = [line.split() for line in
            (work / "runs.txt").read_text(encoding="utf-8").splitlines() if line]
    results = {}
    interior = (slice(None), slice(1, -1), slice(1, -1))
    for name, source, mode, light, clean, allcol, threshold in runs:
        shape, case = _case_b(work / source)
        noaa = _out_b(work / f"out_b_{name}.bin", shape)
        # The boundary columns of t and p are never written by the Fortran
        # (PrecipMxr_radar.f90:103-110 loops the interior); give the kernel
        # finite values there.  Only the interior is compared.
        t = np.array(noaa["t"], copy=True)
        p = np.array(noaa["p"], copy=True)
        edge = np.ones(shape, bool)
        edge[interior] = False
        t[edge] = 280.0
        p[edge] = 50000.0
        cfg = ha.HydrometeorAnalysisConfig(
            mode={"all": "retrieve-all"}.get(mode, mode),
            cold_surface_threshold_c=float(threshold),
            light_precipitation=light == "1",
            clear_with_reflectivity=clean == "1",
            clear_column_when_satellite_clear=allcol == "1",
            scope="noaa")
        fields = {key: cp.asarray(case[key].astype(np.float32))
                  for key in ha.THOMPSON_FIELDS}
        result = ha.hydrometeor_analysis(
            fields, cp.asarray(case["ref"]), cfg,
            temperature=cp.asarray(t), pressure=cp.asarray(p),
            cloud_top_pressure=cp.asarray(case["ctp"]))
        rows = {}
        for key in ha.THOMPSON_FIELDS:
            ours = cp.asnumpy(result.analysed[key])
            rows[key] = compare(noaa[key][interior], ours[interior])
        results[name] = {
            "mode": cfg.mode, "settings": {
                "i_lightpcp": int(light), "iclean_hydro_withRef": int(clean),
                "iclean_hydro_withRef_allcol": int(allcol),
                "r_cleanSnow_WarmTs_threshold": float(threshold),
                "satellite_cloud_top_pressure": source != "case_b_noctp.bin"},
            "fields": rows,
            "columns": result.receipt.get("columns"),
            "cells": result.receipt["cells"],
            "pass": all(passed(r) for r in rows.values()),
        }
    return results


def main() -> int:
    work = Path(sys.argv[1])
    receipt_path = Path(sys.argv[2])
    import cupy as cp

    from gpuwm.da import hydrometeor_analysis as ha

    device = cp.cuda.runtime.getDeviceProperties(0)
    receipt = {
        "card": device["name"].decode() if isinstance(device["name"], bytes)
        else str(device["name"]),
        "nvrtc": ".".join(str(v) for v in cp.cuda.nvrtc.getVersion()),
        "cupy": cp.__version__,
        "build": (work / "build-info.txt").read_text(encoding="utf-8"),
        "cases": json.loads((work / "cases.json").read_text(encoding="utf-8")),
        "oracle_a": oracle_a(work, ha, cp),
        "oracle_b": oracle_b(work, ha, cp),
    }
    receipt["pass"] = bool(receipt["oracle_a"]["pass"] and all(
        run["pass"] for run in receipt["oracle_b"].values()))
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"pass": receipt["pass"],
                      "oracle_a": {k: v for k, v in receipt["oracle_a"].items()
                                   if k != "fields"},
                      "oracle_a_fields": receipt["oracle_a"]["fields"],
                      "oracle_b": {k: {f: (r["max_ulp"], r["not_bit_identical"])
                                       for f, r in v["fields"].items()}
                                   for k, v in receipt["oracle_b"].items()}},
                     indent=1))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
