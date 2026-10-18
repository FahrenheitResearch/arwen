"""Bitwise comparison of the WOOF and WRF outputs of one oracle run.

    python compare.py CASE WRF_OUT WOOF_OUT [--json FILE] [--label L]

Every output word is compared as its IEEE-754 bit pattern.  The distance
reported is in units in the last place, on the ordered integer line, so
+0 and -0 are 0 apart only if they are the same word (a signed-zero
difference counts as 1 ULP and is reported as a differing word).

A replicated WOOF run (nx*R columns) is compared replica by replica
against the same WRF words.

Columns are also tagged by regime from their inputs and from WRF's own
answer, so the summary says which regimes the agreement covers:

  ocean, land, high_terrain (surface below 750 hPa), cold_surface
  (t < 273.15 K at the lowest level), ice_aloft (qi > 1e-6 somewhere),
  stable_surface (HFX < 0), active (any non-zero WRF tendency other than
  rqvcuten, or rain),
  raining (RAINCV > 0), deep (WRF heating reaches above 500 hPa),
  shallow_or_mid (active, heating confined below 500 hPa).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_subset import classes, load  # noqa: E402

LEV_OUT = ("rthcuten", "rqvcuten", "rqccuten", "rqicuten", "rucuten",
           "rvcuten")
SFC_OUT = ("raincv", "pratec")


def ordered(a):
    i = a.view(np.int32).astype(np.int64)
    return np.where(i < 0, -(i & 0x7FFFFFFF), i)


def ulp(a, b):
    return np.abs(ordered(a) - ordered(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("wrf")
    ap.add_argument("woof")
    ap.add_argument("--json", default=None)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    meta, lev, iface, sfc = load(a.case)
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    wrf_dir, woof_dir = pathlib.Path(a.wrf), pathlib.Path(a.woof)
    woof_info = json.loads((woof_dir / "woof-run.json").read_text())
    R = int(woof_info["replicate"])

    def wrf_lev(n):
        return np.fromfile(wrf_dir / f"{n}.f32", "<f4").reshape(ny, nz, nx)

    def woof_lev(n):
        return np.fromfile(woof_dir / f"{n}.f32", "<f4").reshape(
            ny, nz, nx * R)

    fields = {}
    differing_columns = np.zeros(ny * nx, dtype=bool)
    total_words = total_diff = 0
    for n in LEV_OUT + SFC_OUT:
        if n in SFC_OUT:
            p = woof_dir / f"{n}.f32"
            if not p.exists():
                fields[n] = {"graded": False}
                continue
            w = np.fromfile(wrf_dir / f"{n}.f32", "<f4").reshape(ny, 1, nx)
            g = np.fromfile(p, "<f4").reshape(ny, 1, nx * R)
        else:
            w, g = wrf_lev(n), woof_lev(n)
        worst = 0
        ndiff = 0
        maxabs = 0.0
        nan = 0
        for r in range(R):
            gr = g[:, :, r * nx:(r + 1) * nx]
            same = gr.view(np.int32) == w.view(np.int32)
            ndiff += int((~same).sum())
            d = ulp(gr, w)
            worst = max(worst, int(d.max()))
            maxabs = max(maxabs, float(np.nanmax(np.abs(
                gr.astype(np.float64) - w.astype(np.float64)))))
            nan += int(np.isnan(gr).sum())
            differing_columns |= (~same).any(axis=1).ravel()
        fields[n] = {"graded": True, "words": int(w.size * R),
                     "differing_words": ndiff, "max_ulp": worst,
                     "max_abs_diff": maxabs, "woof_nan": nan,
                     "wrf_nonzero_words": int((w != 0).sum())}
        total_words += w.size * R
        total_diff += ndiff

    # Regimes, per column.
    tags = classes(lev, iface, sfc)
    rth = wrf_lev("rthcuten")
    rain = np.fromfile(wrf_dir / "raincv.f32", "<f4").reshape(ny, nx)
    # rqvcuten is left out of "active": cu_ntiedtke_run converts qv to
    # specific humidity and back (q/(1+q), then q/(1-q)), and that round
    # trip is not exact in float32, so WRF writes a tiny rqvcuten in every
    # column whether or not anything convected.
    nonzero = np.zeros((ny, nx), dtype=bool)
    for n in LEV_OUT:
        if n != "rqvcuten":
            nonzero |= (wrf_lev(n) != 0).any(axis=1)
    active = (nonzero | (rain > 0)).ravel()
    p = lev["pcps"]
    heat_high = ((rth != 0) & (p < 50000.0)).any(axis=1).ravel()
    tags.update({
        "land": ~tags["ocean"], "active": active,
        "raining": (rain > 0).ravel(), "deep": active & heat_high,
        "shallow_or_mid": active & ~heat_high,
    })
    regimes = {k: {"columns": int(v.sum()),
                   "columns_with_any_difference":
                       int((v & differing_columns).sum())}
               for k, v in tags.items()}
    flag = np.fromfile(wrf_dir / "cu_act_flag.i32", "<i4")
    summary = {
        "label": a.label, "case": str(a.case), "columns": nx * ny,
        "replicate": R, "nz": nz, "woof": woof_info,
        "total_words": int(total_words), "total_differing_words":
            int(total_diff),
        "bitwise_identical": total_diff == 0,
        "max_ulp": max(f.get("max_ulp", 0) for f in fields.values()),
        "fields": fields, "regimes": regimes,
        "wrf_cu_act_flag_true": int(flag.sum()),
        "columns_with_any_difference": int(differing_columns.sum()),
    }
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{a.label}: {nx * ny} cols x{R}, words {total_words}, "
          f"differing {total_diff}, max_ulp {summary['max_ulp']}, "
          f"active {int(active.sum())}, deep {int(tags['deep'].sum())}, "
          f"raining {int(tags['raining'].sum())}, strict "
          f"{woof_info['strict']}")
    return 0 if total_diff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
