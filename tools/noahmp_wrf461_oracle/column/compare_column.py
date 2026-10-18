"""Bitwise comparison of the two column-oracle outputs.

Reads ``run_column`` (WRF v4.6.1) and ``run_woof_column.py`` (WOOF) outputs
and reports, for every field at every snapshot, how many column values
differ in their bits, the largest ULP distance and the largest absolute
difference.  Integer fields compare exactly.  NaN equals NaN only when
the bit patterns agree.

Documented exclusions are listed by name, never silently dropped:

* ``glacier_undefined``: on a glacier column (VEGTYP = ISICE_TABLE)
  ``noahmplsm`` writes EFLXBXY, SOILENERGY and SNOWENERGY from EFLXB and
  HCPCT, two locals that only NOAHMP_SFLX assigns and the glacier arm never
  does (module_sf_noahmpdrv.F:1045-1150, :1378-1394).  WRF publishes
  whatever the previous non-glacier column left in them -- a read of an
  undefined variable, i.e. a WRF bug.  WOOF writes a defined replacement
  (the glacier column's own EFLXB and HCPCT).  These three fields on
  glacier columns are reported separately and never counted as a match or
  a miss.

Usage:  python compare_column.py WRF.bin WOOF.bin INPUT.bin [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.noahmp_wrf461_oracle.column.gen_columns import read_records  # noqa: E402

GLACIER_UNDEFINED = ("eflxbxy", "soilenergy", "snowenergy")
ISICE = 15


def snapshots(path):
    """Split an output file into a list of {field: array} per snapshot."""
    import struct
    data = Path(path).read_bytes()
    pos = 0
    snaps = []
    while pos < len(data):
        name = data[pos:pos + 16].decode("ascii").strip()
        kind, count = struct.unpack("<ii", data[pos + 16:pos + 24])
        pos += 24
        values = np.frombuffer(data, dtype="<f4" if kind == 1 else "<i4",
                               count=count, offset=pos).copy()
        pos += 4 * count
        if name == "step":
            snaps.append({"step": int(values[0])})
        else:
            snaps[-1][name] = values
    return snaps


def ulp_distance(a, b):
    """ULP distance between float32 arrays on the ordered-integer line."""
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ib = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(ia - ib)


def compare(wrf_path, woof_path, input_path):
    wrf = snapshots(wrf_path)
    woof = snapshots(woof_path)
    inp = read_records(input_path)
    n = int(inp["nx"][0][0])
    veg = inp["ivgtyp"][0]
    xland = inp["xland"][0]
    xice = inp["xice"][0]
    # The columns noahmplsm routes to NOAHMP_GLACIER: land (XLAND < 1.5),
    # not sea ice (XICE < XICE_THRESHOLD = 0.5), VEGTYP = ISICE_TABLE.
    glacier_cols = np.flatnonzero((veg == ISICE) & (xice < np.float32(0.5))
                                  & (xland - np.float32(1.5) < 0))
    if len(wrf) != len(woof):
        raise SystemExit(f"snapshot count differs: {len(wrf)} vs {len(woof)}")
    expected_steps = list(range(int(inp['nstep'][0][0]) + 1))
    for label, data in (('WRF', wrf), ('WOOF', woof)):
        if [snapshot['step'] for snapshot in data] != expected_steps:
            raise SystemExit(f'{label} snapshot sequence must be {expected_steps}')
    fields = [k for k in wrf[0] if k != "step"]
    if [k for k in woof[0] if k != "step"] != fields:
        raise SystemExit("field lists differ between the two outputs")

    per_field = {name: {"values": 0, "differing": 0, "max_ulp": 0,
                        "max_abs": 0.0, "first": None} for name in fields}
    per_column = np.zeros(n, dtype=np.int64)
    excluded = {"glacier_undefined": {"values": 0, "differing": 0,
                                      "max_abs": 0.0}}
    for s_wrf, s_woof in zip(wrf, woof):
        step = s_wrf["step"]
        if s_woof["step"] != step:
            raise SystemExit("snapshot order differs")
        for name in fields:
            a = s_wrf[name]
            b = s_woof[name]
            if a.shape != b.shape:
                raise SystemExit(f"{name}: shape {a.shape} vs {b.shape}")
            layers = a.size // n
            a2 = a.reshape(layers, n)
            b2 = b.reshape(layers, n)
            keep = np.ones((layers, n), dtype=bool)
            if name in GLACIER_UNDEFINED and glacier_cols.size:
                keep[:, glacier_cols] = False
                ga = a2[:, glacier_cols]
                gb = b2[:, glacier_cols]
                ex = excluded["glacier_undefined"]
                ex["values"] += ga.size
                ex["differing"] += int((ga.view(np.int32)
                                        != gb.view(np.int32)).sum())
                with np.errstate(invalid="ignore"):
                    ex["max_abs"] = max(ex["max_abs"], float(np.nanmax(
                        np.abs(ga.astype(np.float64) - gb.astype(np.float64)),
                        initial=0.0)))
            if a.dtype == np.int32:
                diff = (a2 != b2) & keep
                ulp = np.where(diff, np.abs(a2.astype(np.int64)
                                            - b2.astype(np.int64)), 0)
                absd = ulp.astype(np.float64)
            else:
                diff = (a2.view(np.int32) != b2.view(np.int32)) & keep
                ulp = np.where(diff, ulp_distance(a2, b2), 0)
                with np.errstate(invalid="ignore", over="ignore"):
                    absd = np.where(diff, np.abs(a2.astype(np.float64)
                                                 - b2.astype(np.float64)), 0.0)
            rec = per_field[name]
            rec["values"] += int(keep.sum())
            nd = int(diff.sum())
            rec["differing"] += nd
            if nd:
                rec["max_ulp"] = max(rec["max_ulp"], int(ulp.max()))
                rec["max_abs"] = max(rec["max_abs"],
                                     float(np.nanmax(absd, initial=0.0)))
                per_column += diff.any(axis=0)
                if rec["first"] is None:
                    layer, col = (int(x[0]) for x in np.nonzero(diff))
                    rec["first"] = {"step": step, "column": col + 1,
                                    "layer": layer,
                                    "wrf": float(a2[layer, col]),
                                    "woof": float(b2[layer, col])}
    total_values = sum(r["values"] for r in per_field.values())
    total_diff = sum(r["differing"] for r in per_field.values())
    return {
        "columns": n, "snapshots": len(wrf), "fields": len(fields),
        "compared_values": total_values, "differing_values": total_diff,
        "max_ulp": max(r["max_ulp"] for r in per_field.values()),
        "max_abs": max(r["max_abs"] for r in per_field.values()),
        "columns_with_any_difference": int((per_column > 0).sum()),
        "per_field": per_field, "excluded": excluded,
        "glacier_columns": [int(c) + 1 for c in glacier_cols],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wrf")
    ap.add_argument("woof")
    ap.add_argument("input")
    ap.add_argument("--json")
    args = ap.parse_args()
    r = compare(args.wrf, args.woof, args.input)
    print(f"columns {r['columns']}  snapshots {r['snapshots']}  "
          f"fields {r['fields']}  values {r['compared_values']}")
    print(f"differing values {r['differing_values']}  max ULP {r['max_ulp']}"
          f"  max |diff| {r['max_abs']:.6g}  columns touched "
          f"{r['columns_with_any_difference']}")
    for name, rec in r["per_field"].items():
        if rec["differing"]:
            print(f"  {name:12s} differing {rec['differing']:6d}  max ULP "
                  f"{rec['max_ulp']:>10d}  max |diff| {rec['max_abs']:.6g}  "
                  f"first {rec['first']}")
    ex = r["excluded"]["glacier_undefined"]
    print(f"excluded (WRF undefined read on glacier columns "
          f"{r['glacier_columns']}): {ex['values']} values, "
          f"{ex['differing']} differ")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=1))
    return 0 if r["differing_values"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
