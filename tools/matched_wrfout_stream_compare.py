#!/usr/bin/env python3
"""Streaming matched-run comparison of gpuwm wrfouts against CPU WRF wrfouts.

Watches a gpuwm run directory; whenever a wrfout frame has a same-domain,
same-valid-time counterpart in the (read-only) CPU reference directory, it
computes the matched-comparison metrics for that lead and appends one CSV
row.  Designed to overlap a live GPU integration: strictly CPU-only (no
CuPy import; set CUDA_VISIBLE_DEVICES="" in the launching shell for belt
and suspenders).

Metrics per (domain, valid time), computed on the common mass grid with
the outer ``--exclude-rows`` frame removed (default 5 = spec_bdy_width,
mirroring the interior-grid convention of the earlier matched
comparisons):

- T2/PSFC: MAE + Pearson correlation (+ means)
- 10 m wind: mean of U10/V10 correlations
- QVAPOR, W: full-3D Pearson correlation
- RAINNC: correlation + CPU/GPU domain means (mm)
- composite reflectivity (column max of the model-native REFL_10CM both
  sides -- a direct field comparison, no derived formula): correlation,
  MAE, and CSI at 20 dBZ

Completion: exits when a ``--done-file`` exists AND no unprocessed pair
remains; otherwise polls.  Frames are only consumed once they are older
than a settling delay and open cleanly with a readable Times entry, so
half-written files are never scored.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_FRAME_RE = re.compile(
    r"wrfout_d(?P<dom>\d{2})_(?P<time>\d{4}-\d{2}-\d{2}[_:]\d{2}[_:]\d{2}[_:]\d{2})$")


def _frame_key(path: Path) -> tuple[str, str] | None:
    m = _FRAME_RE.match(path.name)
    if not m:
        return None
    stamp = m.group("time").replace(":", "_")
    return (f"d{m.group('dom')}", stamp)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0.0:
        return float("nan")
    return float((a * b).sum() / denom)


def _csi(gpu: np.ndarray, cpu: np.ndarray, threshold: float) -> float:
    g = gpu >= threshold
    c = cpu >= threshold
    hits = int(np.sum(g & c))
    misses = int(np.sum(~g & c))
    false_alarms = int(np.sum(g & ~c))
    denom = hits + misses + false_alarms
    return float("nan") if denom == 0 else hits / denom


def _interior(field: np.ndarray, rows: int) -> np.ndarray:
    """Trim the outer boundary frame from the LAST TWO axes."""
    if rows <= 0:
        return field
    return field[..., rows:-rows, rows:-rows]


def _read_fields(path: Path, names: tuple[str, ...], rows: int):
    import netCDF4

    out = {}
    with netCDF4.Dataset(path) as ds:
        times = ds.variables["Times"][:]
        stamp = b"".join(times[0].data).decode() if hasattr(
            times[0], "data") else "".join(
                x.decode() for x in times[0].astype(str))
        for name in names:
            if name not in ds.variables:
                out[name] = None
                continue
            arr = np.asarray(ds.variables[name][0], dtype=np.float32)
            out[name] = _interior(arr, rows)
    return stamp, out


_FIELDS = ("T2", "PSFC", "U10", "V10", "QVAPOR", "W", "RAINNC", "REFL_10CM")


def compare_pair(gpu_path: Path, cpu_path: Path, rows: int) -> dict:
    gpu_stamp, gpu = _read_fields(gpu_path, _FIELDS, rows)
    cpu_stamp, cpu = _read_fields(cpu_path, _FIELDS, rows)
    if gpu_stamp != cpu_stamp:
        raise ValueError(
            f"Times mismatch: gpu {gpu_stamp!r} vs cpu {cpu_stamp!r}")
    row: dict[str, object] = {}

    def _stat2(name: str, key: str):
        g, c = gpu[name], cpu[name]
        if g is None or c is None:
            row[f"{key}_mae"] = row[f"{key}_corr"] = float("nan")
            return
        # U10/V10 on mass points both sides; guard staggering drift anyway.
        if g.shape != c.shape:
            raise ValueError(f"{name} shape {g.shape} != {c.shape}")
        row[f"{key}_mae"] = float(np.abs(
            g.astype(np.float64) - c.astype(np.float64)).mean())
        row[f"{key}_corr"] = _pearson(g, c)

    _stat2("T2", "t2")
    _stat2("PSFC", "psfc")
    u = _pearson(gpu["U10"], cpu["U10"]) if gpu["U10"] is not None else float("nan")
    v = _pearson(gpu["V10"], cpu["V10"]) if gpu["V10"] is not None else float("nan")
    row["wind10_corr"] = (u + v) / 2.0
    row["qvapor_corr"] = (_pearson(gpu["QVAPOR"], cpu["QVAPOR"])
                          if gpu["QVAPOR"] is not None else float("nan"))
    row["w_corr"] = (_pearson(gpu["W"], cpu["W"])
                     if gpu["W"] is not None else float("nan"))
    if gpu["RAINNC"] is not None and cpu["RAINNC"] is not None:
        row["rainnc_corr"] = _pearson(gpu["RAINNC"], cpu["RAINNC"])
        row["cpu_rainnc_mean_mm"] = float(cpu["RAINNC"].mean())
        row["gpu_rainnc_mean_mm"] = float(gpu["RAINNC"].mean())
    else:
        row["rainnc_corr"] = float("nan")
        row["cpu_rainnc_mean_mm"] = row["gpu_rainnc_mean_mm"] = float("nan")
    if gpu["REFL_10CM"] is not None and cpu["REFL_10CM"] is not None:
        gcomp = gpu["REFL_10CM"].max(axis=0)
        ccomp = cpu["REFL_10CM"].max(axis=0)
        row["refl_comp_corr"] = _pearson(gcomp, ccomp)
        row["refl_comp_mae"] = float(np.abs(
            gcomp.astype(np.float64) - ccomp.astype(np.float64)).mean())
        row["refl_csi20"] = _csi(gcomp, ccomp, 20.0)
        row["cpu_refl_max"] = float(ccomp.max())
        row["gpu_refl_max"] = float(gcomp.max())
    else:
        for key in ("refl_comp_corr", "refl_comp_mae", "refl_csi20",
                    "cpu_refl_max", "gpu_refl_max"):
            row[key] = float("nan")
    return row


_COLUMNS = (
    "domain", "valid_time", "forecast_hour",
    "t2_mae", "t2_corr", "psfc_mae", "psfc_corr", "wind10_corr",
    "qvapor_corr", "w_corr", "rainnc_corr",
    "cpu_rainnc_mean_mm", "gpu_rainnc_mean_mm",
    "refl_comp_corr", "refl_comp_mae", "refl_csi20",
    "cpu_refl_max", "gpu_refl_max",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-dir", type=Path, required=True)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--log-md", type=Path, default=None,
                        help="markdown ledger to append one line per frame")
    parser.add_argument("--done-file", type=Path, default=None,
                        help="exit once this exists and no pair is pending")
    parser.add_argument("--start-time", default=None,
                        help="run start (YYYY-MM-DD_HH:MM:SS) for lead hours")
    parser.add_argument("--exclude-rows", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    args = parser.parse_args()

    start_dt = (datetime.strptime(args.start_time, "%Y-%m-%d_%H:%M:%S")
                if args.start_time else None)
    done: set[tuple[str, str]] = set()
    if args.out_csv.exists():
        with args.out_csv.open() as fh:
            for existing in csv.DictReader(fh):
                done.add((existing["domain"],
                          existing["valid_time"].replace(":", "_")))
    else:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as fh:
            csv.writer(fh).writerow(_COLUMNS)

    cpu_index: dict[tuple[str, str], Path] = {}
    for path in sorted(args.cpu_dir.iterdir()):
        key = _frame_key(path)
        if key:
            cpu_index[key] = path

    while True:
        pending = []
        for path in sorted(args.gpu_dir.glob("wrfout_d0*")):
            key = _frame_key(path)
            if key is None or key in done or key not in cpu_index:
                continue
            age = time.time() - path.stat().st_mtime
            if age < args.settle_seconds:
                pending.append(key)
                continue
            try:
                row = compare_pair(path, cpu_index[key], args.exclude_rows)
            except Exception as exc:  # noqa: BLE001 - retry next poll
                print(f"[stream-compare] {path.name}: not ready ({exc})",
                      flush=True)
                pending.append(key)
                continue
            domain, stamp = key
            valid = stamp.replace("_", ":", 2) if False else stamp
            lead = ""
            if start_dt is not None:
                vt = datetime.strptime(stamp, "%Y-%m-%d_%H_%M_%S")
                lead = f"{(vt - start_dt).total_seconds() / 3600.0:.1f}"
            record = {"domain": domain, "valid_time": stamp,
                      "forecast_hour": lead}
            record.update({k: row.get(k, float("nan"))
                           for k in _COLUMNS if k not in record})
            with args.out_csv.open("a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_COLUMNS).writerow(record)
            done.add(key)
            line = (f"- {domain} {stamp} (F{lead or '?'}): "
                    f"T2 MAE {record['t2_mae']:.3f} K corr "
                    f"{record['t2_corr']:.3f}; PSFC MAE "
                    f"{record['psfc_mae']:.1f} Pa; refl-comp corr "
                    f"{record['refl_comp_corr']:.3f} MAE "
                    f"{record['refl_comp_mae']:.2f} dBZ CSI20 "
                    f"{record['refl_csi20']:.3f}; wind10 corr "
                    f"{record['wind10_corr']:.3f}; RAINNC cpu/gpu "
                    f"{record['cpu_rainnc_mean_mm']:.3f}/"
                    f"{record['gpu_rainnc_mean_mm']:.3f} mm")
            print(f"[stream-compare] {line}", flush=True)
            if args.log_md is not None:
                with args.log_md.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        run_done = args.done_file is not None and args.done_file.exists()
        if run_done and not pending:
            remaining = [k for k in cpu_index
                         if k not in done
                         and (args.gpu_dir / f"wrfout_{k[0]}_{k[1]}").exists()]
            if not remaining:
                print("[stream-compare] complete", flush=True)
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
