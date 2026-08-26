#!/usr/bin/env python3
"""Positive evidence and reductions for a p_top A/B: what did each arm run?

A/B arms prove treatment: an exact-zero delta between two runs means one of
them never ran, and a nonzero delta proves nothing about *what* ran.  This
tool reads every history frame of each named arm and writes, per arm, the
positive evidence of its configuration -- the P_TOP the output carries, the
geometric depth of its column, where its damp_opt=3 sponge base sits -- and
the physical reductions the comparison argues from: per-frame max updraft
profiles, composite-reflectivity maxima, histograms, object counts, echo-top
and cloud-top height distributions.

Analysis instrument only: it renders nothing and decides nothing.  The
composite-reflectivity plane of every frame is also saved as ``.npy`` so the
charts can be drawn without hauling the wrfout files anywhere.

Usage::

    python tools/ptop_ab_probes.py --arm control=RUNDIR --arm treatment=RUNDIR \
        --zdamp 5000.0 --out OUTDIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError as error:  # pragma: no cover - environment probe
    raise SystemExit(
        "ptop_ab_probes needs the netCDF4 package in this environment "
        f"({error}); `pip install netCDF4` and rerun") from error

G = 9.81
ECHO_TOP_DBZ = 18.3
CLOUD_MIXING_RATIO_FLOOR = 1e-6  # kg/kg of qc+qi+qs
OBJECT_THRESHOLDS_DBZ = (30.0, 40.0, 50.0)
HISTOGRAM_EDGES_DBZ = tuple(float(v) for v in range(-30, 80, 5))


def _label_count(mask: np.ndarray) -> int:
    """4-connected component count, plain numpy (grids here are tiny)."""
    remaining = np.array(mask, dtype=bool)
    count = 0
    while True:
        seeds = np.argwhere(remaining)
        if seeds.size == 0:
            return count
        count += 1
        stack = [tuple(seeds[0])]
        remaining[tuple(seeds[0])] = False
        while stack:
            j, i = stack.pop()
            for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nj, ni = j + dj, i + di
                if (0 <= nj < remaining.shape[0]
                        and 0 <= ni < remaining.shape[1]
                        and remaining[nj, ni]):
                    remaining[nj, ni] = False
                    stack.append((nj, ni))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def probe_frame(path: Path, *, zdamp: float, npy_dir: Path) -> dict:
    with Dataset(path) as nc:
        record: dict[str, object] = {"frame": path.name}
        if "P_TOP" in nc.variables:
            record["p_top_pa"] = float(np.asarray(nc["P_TOP"][:]).ravel()[0])
        ph = np.asarray(nc["PH"][0], dtype=np.float64)
        phb = np.asarray(nc["PHB"][0], dtype=np.float64)
        z_w = (ph + phb) / G                       # (nz+1, ny, nx), MSL
        terrain = z_w[0]
        z_top = z_w[-1]
        z_mass = 0.5 * (z_w[1:] + z_w[:-1])        # (nz, ny, nx)
        record["column_top_m_msl"] = {
            "mean": float(z_top.mean()), "min": float(z_top.min()),
            "max": float(z_top.max())}
        record["column_depth_m"] = {
            "mean": float((z_top - terrain).mean()),
            "min": float((z_top - terrain).min()),
            "max": float((z_top - terrain).max())}
        record["sponge_base_m_agl"] = {
            "mean": float((z_top - zdamp - terrain).mean()),
            "min": float((z_top - zdamp - terrain).min()),
            "max": float((z_top - zdamp - terrain).max())}

        if "W" in nc.variables:
            w = np.asarray(nc["W"][0], dtype=np.float64)
            record["w_max_ms"] = float(w.max())
            record["w_min_ms"] = float(w.min())
            record["w_max_profile"] = {
                "z_mean_m_msl": [float(v) for v in
                                 z_w.reshape(z_w.shape[0], -1).mean(axis=1)],
                "w_max_ms": [float(v) for v in
                             w.reshape(w.shape[0], -1).max(axis=1)],
            }

        cloud = None
        for name in ("QCLOUD", "QICE", "QSNOW"):
            if name in nc.variables:
                part = np.asarray(nc[name][0], dtype=np.float64)
                cloud = part if cloud is None else cloud + part
        if cloud is not None:
            cloudy = cloud >= CLOUD_MIXING_RATIO_FLOOR
            any_cloud = cloudy.any(axis=0)
            top_z = np.where(
                any_cloud,
                np.max(np.where(cloudy, z_mass, -np.inf), axis=0),
                np.nan)
            agl = (top_z - terrain)[any_cloud]
            record["cloud_top_m_agl"] = _percentiles(agl[np.isfinite(agl)])
            record["cloudy_column_fraction"] = float(any_cloud.mean())

        if "REFL_10CM" in nc.variables:
            refl = np.asarray(nc["REFL_10CM"][0], dtype=np.float64)
            comp = refl.max(axis=0)
            np.save(npy_dir / (path.name + ".comprefl.npy"),
                    comp.astype(np.float32))
            record["comp_refl_max_dbz"] = float(comp.max())
            record["comp_refl_histogram"] = {
                "edges_dbz": list(HISTOGRAM_EDGES_DBZ),
                "counts": [int(v) for v in np.histogram(
                    comp, bins=HISTOGRAM_EDGES_DBZ)[0]],
            }
            record["comp_refl_coverage"] = {
                f">={int(t)}dbz": float((comp >= t).mean())
                for t in OBJECT_THRESHOLDS_DBZ}
            record["comp_refl_objects"] = {
                f">={int(t)}dbz": _label_count(comp >= t)
                for t in OBJECT_THRESHOLDS_DBZ}
            echoing = refl >= ECHO_TOP_DBZ
            any_echo = echoing.any(axis=0)
            etop = np.where(
                any_echo,
                np.max(np.where(echoing, z_mass, -np.inf), axis=0),
                np.nan)
            agl = (etop - terrain)[any_echo]
            record["echo_top_m_agl"] = _percentiles(agl[np.isfinite(agl)])
        return record


def probe_arm(name: str, run_dir: Path, *, zdamp: float, out: Path) -> dict:
    frames = sorted(run_dir.glob("wrfout_d01_*"))
    if not frames:
        raise SystemExit(f"arm {name}: no wrfout_d01_* frames in {run_dir}")
    npy_dir = out / f"comprefl_{name}"
    npy_dir.mkdir(parents=True, exist_ok=True)
    return {
        "arm": name,
        "run_directory": str(run_dir),
        "frame_count": len(frames),
        "zdamp_m": float(zdamp),
        "frames": [probe_frame(frame, zdamp=zdamp, npy_dir=npy_dir)
                   for frame in frames],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True,
                        metavar="NAME=RUNDIR",
                        help="an arm to probe; repeatable")
    parser.add_argument("--zdamp", type=float, required=True,
                        help="the configured damp_opt=3 layer depth (m)")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"arms": {}}
    for spec in arguments.arm:
        name, _, directory = spec.partition("=")
        if not directory:
            raise SystemExit(f"--arm takes NAME=RUNDIR, got {spec!r}")
        record = probe_arm(name, Path(directory), zdamp=arguments.zdamp,
                           out=arguments.out)
        path = arguments.out / f"probes_{name}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        first = record["frames"][0]
        summary["arms"][name] = {
            "p_top_pa": first.get("p_top_pa"),
            "column_depth_m_mean_t0": first["column_depth_m"]["mean"],
            "sponge_base_m_agl_mean_t0": first["sponge_base_m_agl"]["mean"],
            "frame_count": record["frame_count"],
            "probes_file": str(path),
        }
        print(f"{name}: p_top={summary['arms'][name]['p_top_pa']} Pa, "
              f"column depth {first['column_depth_m']['mean']:.0f} m, "
              f"sponge base {first['sponge_base_m_agl']['mean']:.0f} m AGL, "
              f"{record['frame_count']} frames")
    (arguments.out / "ab_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
