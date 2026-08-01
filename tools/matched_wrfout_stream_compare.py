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


def boundary_distance(ny: int, nx: int) -> np.ndarray:
    """Cell distance to the nearest lateral boundary, ``min`` over sides.

    Identical convention to the runtime's boundary-row attribution
    (``distance = min(j, ny - 1 - j, i, nx - 1 - i)``): d = 0 is the
    specified row, d = 1.. are relaxation rows.  Re-deriving it here keeps
    the comparator CPU-only, and a test asserts the two agree cell for
    cell rather than by inspection.
    """
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    return np.minimum(np.minimum(j, ny - 1 - j),
                      np.minimum(i, nx - 1 - i))


def strata_masks(ny: int, nx: int, spec_bdy_width: int):
    """``(label, mask)`` pairs for the specified / relaxation / interior
    bands, plus one mask per individual relaxation row.

    The interior band starts at ``d >= spec_bdy_width``, so the aggregate
    relaxation band is exactly ``d = 1..spec_bdy_width - 1``.  Bands with
    no cells are still emitted, with an empty mask, so a small frame
    reports a band as unpopulated instead of silently dropping it.
    """
    if spec_bdy_width < 1:
        raise ValueError(
            f"spec_bdy_width must be >= 1, got {spec_bdy_width}")
    d = boundary_distance(ny, nx)
    bands = [("d0_specified", d == 0)]
    for row in range(1, spec_bdy_width):
        bands.append((f"d{row}_relaxation", d == row))
    bands.append(("relaxation", (d >= 1) & (d <= spec_bdy_width - 1)))
    bands.append(("interior", d >= spec_bdy_width))
    return bands


def _masked(field, mask: np.ndarray):
    """Select the mask's cells from the LAST TWO axes of ``field``."""
    if field is None:
        return None
    return field[..., mask]


def compare_masked(gpu: dict, cpu: dict, mask: np.ndarray) -> dict:
    """Matched metrics over one boundary band of an untrimmed frame pair."""
    return _compare_fields({k: _masked(v, mask) for k, v in gpu.items()},
                           {k: _masked(v, mask) for k, v in cpu.items()},
                           composite_axis=0)


def compare_strata(gpu_path: Path, cpu_path: Path,
                   spec_bdy_width: int) -> list[dict]:
    """One metric row per boundary band for a matched frame pair."""
    gpu_stamp, gpu = _read_fields(gpu_path, _FIELDS, 0)
    cpu_stamp, cpu = _read_fields(cpu_path, _FIELDS, 0)
    if gpu_stamp != cpu_stamp:
        raise ValueError(
            f"Times mismatch: gpu {gpu_stamp!r} vs cpu {cpu_stamp!r}")
    shape = None
    for name in _FIELDS:
        if gpu.get(name) is not None:
            shape = gpu[name].shape[-2:]
            break
    if shape is None:
        raise ValueError(f"{gpu_path.name} carries none of {_FIELDS}")
    ny, nx = shape
    rows = []
    for label, mask in strata_masks(ny, nx, spec_bdy_width):
        row = {"band": label, "cells": int(mask.sum())}
        if row["cells"] == 0:
            row.update({k: float("nan") for k in _METRIC_COLUMNS})
        else:
            row.update(compare_masked(gpu, cpu, mask))
        rows.append(row)
    return rows


def compare_pair(gpu_path: Path, cpu_path: Path, rows: int) -> dict:
    gpu_stamp, gpu = _read_fields(gpu_path, _FIELDS, rows)
    cpu_stamp, cpu = _read_fields(cpu_path, _FIELDS, rows)
    if gpu_stamp != cpu_stamp:
        raise ValueError(
            f"Times mismatch: gpu {gpu_stamp!r} vs cpu {cpu_stamp!r}")
    return _compare_fields(gpu, cpu, composite_axis=0)


def _compare_fields(gpu: dict, cpu: dict, *, composite_axis: int) -> dict:
    """The metric set, over whatever selection of cells it is handed.

    Shared verbatim by the ``--exclude-rows`` interior grid and by the
    boundary bands, so one field is never scored by two formulas.
    """
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
        gcomp = gpu["REFL_10CM"].max(axis=composite_axis)
        ccomp = cpu["REFL_10CM"].max(axis=composite_axis)
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


_METRIC_COLUMNS = (
    "t2_mae", "t2_corr", "psfc_mae", "psfc_corr", "wind10_corr",
    "qvapor_corr", "w_corr", "rainnc_corr",
    "cpu_rainnc_mean_mm", "gpu_rainnc_mean_mm",
    "refl_comp_corr", "refl_comp_mae", "refl_csi20",
    "cpu_refl_max", "gpu_refl_max",
)

_COLUMNS = ("domain", "valid_time", "forecast_hour") + _METRIC_COLUMNS

_STRATA_COLUMNS = (("domain", "valid_time", "forecast_hour", "band", "cells")
                   + _METRIC_COLUMNS)


def build_parser() -> argparse.ArgumentParser:
    """The comparator's whole flag surface, assembled.

    Split out of :func:`main` so a test can hold the published reproduce
    commands against the parser that actually runs them, instead of against a
    transcription of the help text.
    """
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
    parser.add_argument("--boundary-strata", action="store_true",
                        help="also score per boundary-distance band "
                             "(d=0 specified, d=1..relaxation, interior)")
    parser.add_argument("--strata-csv", type=Path, default=None,
                        help="band table destination "
                             "(default: --out-csv with a -strata suffix)")
    parser.add_argument("--spec-bdy-width", type=int, default=5,
                        help="first interior distance; the relaxation band "
                             "is d = 1..spec_bdy_width-1")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()

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

    strata_csv = None
    if args.boundary_strata:
        strata_csv = args.strata_csv or args.out_csv.with_name(
            args.out_csv.stem + "-strata" + args.out_csv.suffix)
        if not strata_csv.exists():
            strata_csv.parent.mkdir(parents=True, exist_ok=True)
            with strata_csv.open("w", newline="") as fh:
                csv.writer(fh).writerow(_STRATA_COLUMNS)

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
                bands = (compare_strata(path, cpu_index[key],
                                        args.spec_bdy_width)
                         if strata_csv is not None else ())
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
            if strata_csv is not None:
                with strata_csv.open("a", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=_STRATA_COLUMNS)
                    for band in bands:
                        band_record = {"domain": domain, "valid_time": stamp,
                                       "forecast_hour": lead}
                        band_record.update(
                            {k: band.get(k, float("nan"))
                             for k in _STRATA_COLUMNS
                             if k not in band_record})
                        writer.writerow(band_record)
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
