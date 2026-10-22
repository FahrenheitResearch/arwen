"""Readers and word comparison shared by the WSM5 oracle tools (CPU only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.wsm6_wrf461_column_oracle.common import (
    micron, _ordered, ulp, digest, accumulate, read_columns as _read_columns)

LEVEL_IN = ("th", "qv", "qc", "qr", "qi", "qs", "den", "pii", "p", "dz")
SURFACE_IN = ("rain", "snow")
LEVEL_OUT = ("th", "qv", "qc", "qi", "qr", "qs",
             "re_cloud", "re_ice", "re_snow", "refl_10cm", "t")
SURFACE_OUT = ("rain", "rainncv", "snow", "snowncv", "sr")
RADII = ("re_cloud", "re_ice", "re_snow")

#: (set, dt, nsteps) runs build.sh makes.  main: dt 30 s one minor loop,
#: 150 s one loop longer than dtcldcr, 300 s three loops, 600 s five, 1800 s
#: fifteen.  lev2: the remap's last arrival interface at every CFL scale.
RUNS = (("main", "30", 12), ("main", "150", 3), ("main", "300", 4),
        ("main", "600", 2), ("main", "1800", 1),
        ("lev2", "0.001", 3), ("lev2", "1", 3), ("lev2", "20", 4),
        ("lev2", "300", 2))
ARMS = ("scalar", "strict", "stock", "remap", "psevp", "defined")


def tag_of(set_name: str, dt: str, nsteps: int) -> str:
    return f"{set_name}-dt{dt}-n{nsteps}"


def read_set(path: Path):
    return _read_columns(path, LEVEL_IN, SURFACE_IN)


def read_wrf(path: Path):
    """run_wsm5_columns.F90's output; radii in microns, probes if present."""
    with open(path, "rb") as fh:
        ncol, nz, nsteps, probe = (int(v) for v in np.fromfile(fh, "<i4", 4))
        dt = float(np.fromfile(fh, "<f4", 1)[0])
        steps, probes = [], []
        for _ in range(nsteps):
            lv = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in LEVEL_OUT}
            for n in RADII:
                lv[n] = micron(lv[n])
            sf = {n: np.fromfile(fh, "<f4", ncol) for n in SURFACE_OUT}
            steps.append((lv, sf))
            if probe:
                probes.append((np.fromfile(fh, "<i4", ncol), np.fromfile(fh, "<i4", ncol)))
    return dict(ncol=ncol, nz=nz, nsteps=nsteps, dt=dt, steps=steps, probes=probes)


def read_probe(path: Path):
    """remap_probe.F90's output: [(nz, dt, rq, precip)]."""
    out = []
    raw = Path(path).read_bytes()
    pos = 0
    while pos < len(raw):
        nz = int(np.frombuffer(raw, "<i4", 1, pos)[0]); pos += 4
        dt = float(np.frombuffer(raw, "<f4", 1, pos)[0]); pos += 4
        rq = np.frombuffer(raw, "<f4", nz, pos).copy(); pos += 4 * nz
        precip = float(np.frombuffer(raw, "<f4", 1, pos)[0]); pos += 4
        out.append((nz, dt, rq, precip))
    return out


def pairs(a_step, b_step):
    (al, as_), (bl, bs) = a_step, b_step
    out = {n: (al[n], bl[n]) for n in LEVEL_OUT}
    out.update({n: (as_[n][None, :], bs[n][None, :]) for n in SURFACE_OUT})
    return out


def compare_runs(a, b, names, keep=None):
    fields = {}
    for istep, (sa, sb) in enumerate(zip(a["steps"], b["steps"])):
        accumulate(fields, istep, pairs(sa, sb), names, keep)
    return fields


def mismatch_columns(fields) -> set:
    out = set()
    for row in fields.values():
        out.update(row["columns"])
    return out


def column_names(oracle_dir: Path, set_name: str) -> list[str]:
    meta = json.loads((Path(oracle_dir) / f"{set_name}.json").read_text())
    return [c["name"] for c in meta["columns"]]
