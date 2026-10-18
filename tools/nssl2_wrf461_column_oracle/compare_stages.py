"""Attribute a WRF/WOOF miss to a driver stage (diagnostic, not a grade).

Stage 1 = after gather + sedimentation + precipitation sums (before GS),
stage 2 = after the GS process step, stage 3 = after NUCOND.  WRF's slab
``an`` is mapped onto WOOF's 16-field driver workspace through the species
indices the dump carries; theta and temperature are compared too.  The
report lists, per stage and field, the bit-mismatch count, max ULP and the
first (lowest-stage) column/level that differs, which is where to look.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_bitwise import ulp

WOOF_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr", "qni",
               "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
# WRF index names in the dump order (instrument.py)
IDX = ("lt", "lv", "lc", "lr", "li", "ls", "lh", "lhl", "lccn", "lnc", "lnr",
       "lni", "lns", "lnh", "lnhl", "lvh", "lvhl")
WRF_OF = {"qv": "lv", "qc": "lc", "qr": "lr", "qi": "li", "qs": "ls", "qg": "lh",
          "qh": "lhl", "qndrop": "lnc", "qnr": "lnr", "qni": "lni", "qns": "lns",
          "qng": "lnh", "qnh": "lnhl", "qnn": "lccn", "qvolg": "lvh", "qvolh": "lvhl"}


def read_dumps(path: Path, ncol: int):
    raw = path.read_bytes()
    off, stages = 0, {}
    while off < len(raw):
        head = np.frombuffer(raw, dtype="<i4", count=21, offset=off)
        stage, nx, nz, na = (int(v) for v in head[:4])
        idx = dict(zip(IDX, (int(v) for v in head[4:21])))
        off += 21 * 4
        an = np.frombuffer(raw, dtype="<f4", count=nx * nz * na, offset=off)
        an = an.reshape(na, nz, 1, nx)  # Fortran (nx,1,nz,na) -> C reversed
        off += nx * nz * na * 4
        t0 = np.frombuffer(raw, dtype="<f4", count=nx * nz, offset=off).reshape(nz, 1, nx)
        off += nx * nz * 4
        t7 = np.frombuffer(raw, dtype="<f4", count=nx * nz, offset=off).reshape(nz, 1, nx)
        off += nx * nz * 4
        stages.setdefault(stage, []).append((idx, an[:, :, 0, :], t0[:, 0, :], t7[:, 0, :]))
    out = {}
    for stage, items in stages.items():
        if len(items) != ncol:
            raise ValueError(f"stage {stage}: {len(items)} dumps for {ncol} columns")
        idx = items[0][0]
        an = np.concatenate([a for _, a, _, _ in items], axis=2)  # (na, nz, ncol)
        t0 = np.concatenate([t for _, _, t, _ in items], axis=1)  # (nz, ncol)
        t7 = np.concatenate([t for _, _, _, t in items], axis=1)
        fields = {"theta": an[idx["lt"] - 1].T, "t0": t0.T, "t7": t7.T}
        for name in WOOF_FIELDS:
            fields[name] = an[idx[WRF_OF[name]] - 1].T
        out[stage] = fields
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wrf_dump", type=Path)
    p.add_argument("gpu_stages", type=Path)
    p.add_argument("columns", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    regime = np.load(a.columns)["regime"]
    ncol = regime.size
    wrf = read_dumps(a.wrf_dump, ncol)
    gpu = np.load(a.gpu_stages)
    report = {}
    for stage in (1, 2, 3):
        st = gpu[f"stage{stage}_state"]  # (16, nz, ncol)
        g = {name: st[i].T for i, name in enumerate(WOOF_FIELDS)}
        g["theta"] = gpu[f"stage{stage}_theta"]
        g["t0"] = gpu[f"stage{stage}_t0"]
        rows = {}
        for name in ("theta", "t0") + WOOF_FIELDS:
            w, v = wrf[stage][name], g[name].astype(np.float32)
            diff = w.view(np.uint32) != v.view(np.uint32)
            if not diff.any():
                continue
            u = ulp(w, v)
            cols = np.nonzero(diff.any(axis=1))[0]
            c0 = int(cols[0]); k0 = int(np.nonzero(diff[c0])[0][0])
            rows[name] = {"mismatches": int(diff.sum()), "columns": int(cols.size),
                          "max_ulp": int(u.max()),
                          "first": {"column": c0, "level": k0 + 1, "regime": str(regime[c0]),
                                    "wrf": float(w[c0, k0]), "woof": float(v[c0, k0])}}
        report[f"stage{stage}"] = rows
        print(f"stage {stage}: " + (", ".join(f"{k}={r['mismatches']}(max {r['max_ulp']} ulp)"
                                             for k, r in rows.items()) or "EXACT"))
    a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
