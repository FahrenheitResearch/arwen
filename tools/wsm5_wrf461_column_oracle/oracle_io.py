"""Readers and word comparison shared by the WSM5 oracle tools (CPU only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

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
    with open(path, "rb") as fh:
        ncol, nz = (int(v) for v in np.fromfile(fh, "<i4", 2))
        level = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in LEVEL_IN}
        surf = {n: np.fromfile(fh, "<f4", ncol) for n in SURFACE_IN}
    return ncol, nz, level, surf


def micron(re_metres):
    """WRF's radius word through gpuwm's one float32 multiply into microns."""
    return (np.asarray(re_metres, np.float32) * np.float32(1.0e6)).astype(np.float32)


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


def _ordered(a):
    bits = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulp(a, b):
    return np.abs(_ordered(a) - _ordered(b))


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, "<f4").tobytes()).hexdigest()


def pairs(a_step, b_step):
    (al, as_), (bl, bs) = a_step, b_step
    out = {n: (al[n], bl[n]) for n in LEVEL_OUT}
    out.update({n: (as_[n][None, :], bs[n][None, :]) for n in SURFACE_OUT})
    return out


def accumulate(fields, istep, step_pairs, names, keep=None):
    """Word comparison of one step into ``fields`` (bits, ULP, columns)."""
    for n, (w, g) in step_pairs.items():
        w = np.asarray(w, np.float32)
        g = np.asarray(g, np.float32)
        cols = list(names)
        if keep is not None:
            w, g = w[:, keep], g[:, keep]
            cols = [names[i] for i in np.nonzero(keep)[0]]
        diff_bits = w.view(np.uint32) != g.view(np.uint32)
        u = np.where(np.isnan(w) & np.isnan(g), 0, ulp(w, g))
        gap = np.abs(w.astype(np.float64) - g.astype(np.float64))
        gap = np.where(np.isnan(gap), 0.0, gap)
        row = fields.setdefault(n, {"words": 0, "bit_mismatch": 0, "max_ulp": 0,
                                    "max_abs": 0.0, "worst": None, "columns": {},
                                    "first_step_with_a_mismatch": None})
        row["words"] += int(w.size)
        row["bit_mismatch"] += int(diff_bits.sum())
        row["max_abs"] = max(row["max_abs"], float(gap.max()) if gap.size else 0.0)
        if diff_bits.any() and row["first_step_with_a_mismatch"] is None:
            row["first_step_with_a_mismatch"] = istep + 1
        for i in np.nonzero(diff_bits.any(axis=0))[0]:
            row["columns"][cols[i]] = row["columns"].get(cols[i], 0) + int(diff_bits[:, i].sum())
        m = int(u.max()) if u.size else 0
        if diff_bits.any() and (row["worst"] is None or m > row["max_ulp"]):
            k, i = np.unravel_index(int(np.argmax(u)), u.shape)
            row["max_ulp"] = max(row["max_ulp"], m)
            row["worst"] = {"step": istep + 1, "level": int(k), "column": cols[i],
                            "a": float(w[k, i]), "b": float(g[k, i])}


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
