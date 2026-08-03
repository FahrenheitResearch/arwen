"""Reduce a WRF ``wrfout`` from the matched case to the comparison layout.

Writes one ``frame_t<seconds>.npz`` per history time plus a ``series.json``,
in exactly the layout ``run_arwen.py`` writes, so a single comparison script
reads both models without a format branch.

Usage:  python extract_wrfout.py --wrfout PATH --out DIR --mp {8,28}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import netCDF4

MASS_3D = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
           "QNRAIN", "QNICE")
AERO_3D = ("QNCLOUD", "QNWFA", "QNIFA")

#: WRF field -> the ArWen state attribute whose series entry it feeds.
SERIES_MAP = {
    "QCLOUD": "qc", "QRAIN": "qr", "QICE": "qi", "QSNOW": "qs",
    "QGRAUP": "qg", "QNCLOUD": "nc", "QNRAIN": "nr", "QNICE": "ni",
    "QNWFA": "nwfa", "QNIFA": "nifa",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrfout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mp", type=int, required=True, choices=(8, 28))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    wanted = list(MASS_3D) + (list(AERO_3D) if args.mp == 28 else [])
    series = []
    with netCDF4.Dataset(args.wrfout) as ds:
        times = ds.variables["Times"][:]
        n = times.shape[0]
        # WRF history times are equispaced; recover seconds from the header.
        dt_hist = None
        stamps = ["".join(c.decode() for c in row) for row in times]
        for i in range(n):
            fields = {}
            for name in wanted:
                if name in ds.variables:
                    fields[name] = np.asarray(ds.variables[name][i],
                                              dtype=np.float32)
            fields["W"] = np.asarray(ds.variables["W"][i], dtype=np.float32)
            fields["T"] = np.asarray(ds.variables["T"][i], dtype=np.float32)
            fields["RAINNC"] = np.asarray(ds.variables["RAINNC"][i],
                                          dtype=np.float32)
            t_s = _stamp_seconds(stamps[i], stamps[0])
            np.savez_compressed(args.out / f"frame_t{t_s}.npz", **fields)

            row = {"step": None, "time_s": float(t_s),
                   "w_max": float(fields["W"].max()),
                   "w_min": float(fields["W"].min()),
                   "rainnc_sum": float(fields["RAINNC"]
                                       .astype(np.float64).sum()),
                   "rainnc_max": float(fields["RAINNC"].max())}
            for wrf, key in SERIES_MAP.items():
                if wrf in fields:
                    row[f"{key}_max"] = float(fields[wrf].max())
                    row[f"{key}_mean"] = float(
                        fields[wrf].astype(np.float64).mean())
            series.append(row)
    (args.out / "series.json").write_text(json.dumps(series, indent=1))
    print(f"wrote {len(series)} frames to {args.out}")
    return 0


def _stamp_seconds(stamp: str, first: str) -> int:
    """Seconds between two WRF ``Times`` stamps (same day, idealized)."""
    def parse(s):
        d, t = s.split("_")
        yy, mm, dd = (int(v) for v in d.split("-"))
        hh, mi, ss = (int(v) for v in t.split(":"))
        return ((dd * 24 + hh) * 60 + mi) * 60 + ss
    return parse(stamp) - parse(first)


if __name__ == "__main__":
    raise SystemExit(main())
