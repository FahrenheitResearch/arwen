#!/usr/bin/env python3
"""Render one or more ``summary.json`` files from real_column_parity.py as
plain-text tables, pooled over the files given.

usage: report.py SUMMARY.json [SUMMARY.json ...]
"""

from __future__ import annotations

import json
import sys


def _pool(entries):
    """Pool per-file metric dicts: counts add, maxima take the max."""
    out = {}
    for e in entries:
        for k, v in e.items():
            if k == "worst":
                if "worst" not in out or v["rel"] > out["worst"]["rel"]:
                    out["worst"] = v
            elif k.startswith("n_"):
                out[k] = out.get(k, 0) + v
            elif k in ("rel_max", "ulp32_max", "rel_p999", "ulp32_p999",
                       "rel_p99", "rel_median", "abs_diff_max", "scale_p99",
                       "far_abs_max"):
                out[k] = max(out.get(k, 0.0), v)
    return out


#: The classification ``real_column_parity.classify`` writes: cells beyond
#: the rounding gate, and how many of them the port's own one-unit
#: sensitivity or the cell's float32 scale explains.
CLASS_COLS = ["n_active", "n_beyond_rounding", "n_beyond_within_sensitivity",
              "n_beyond_within_cell_scale", "n_beyond_unexplained",
              "n_beyond_1e-2", "n_beyond_1e-2_unexplained"]


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2e}"
    return str(v)


def table(title, rows, cols):
    print(f"\n{title}")
    head = f"{'':10s} " + " ".join(f"{c:>11s}" for c in cols)
    print(head)
    for name, e in rows.items():
        print(f"{name:10s} " + " ".join(
            f"{_fmt(e.get(c, '-')):>11s}" for c in cols))


def main(paths):
    sums = [json.loads(open(p, encoding="utf-8").read()) for p in paths]
    print("files:", ", ".join(f"{s['columns'].split('/')[-1]} "
                              f"({s['ncol']} cols, "
                              f"{s['columns_with_microphysics']} with micro)"
                              for s in sums))
    print("instrumentation neutral: WRF",
          all(s["wrf_instrumentation_neutral"] for s in sums),
          "port", all(s["port_instrumentation_neutral"] for s in sums))
    cols = ["n_active", "n_exact", "rel_p99", "rel_p999", "rel_max",
            "ulp32_max", "n_rel_gt_2e-6", "n_rel_gt_1e-4", "n_rel_gt_1e-2"]
    rates = {name: _pool([s["rates"][name] for s in sums])
             for name in sums[0]["rates"]}
    for name, s in rates.items():
        s["n_wrf_nonzero"] = sum(x["rates"][name]["n_wrf_nonzero"]
                                 for x in sums)
    table("PROCESS RATES (cells where either side is nonzero)", rates,
          ["n_wrf_nonzero"] + cols)
    table("PROCESS RATES, cells beyond the 2e-6 rounding gate, classified",
          {name: _pool([s["rates"][name]["class"] for s in sums])
           for name in sums[0]["rates"]}, CLASS_COLS)
    if "rates_not_carried" in sums[0]:
        table("RATES THE SCHEME'S PORT DOES NOT CARRY (WRF activity only)",
              {name: _pool([s["rates_not_carried"][name] for s in sums])
               for name in sums[0]["rates_not_carried"]}, ["n_wrf_nonzero"])
    for s in sums:
        name = s["columns"].split("/")[-1]
        if "rain_evaporation_at_saturation" in s:
            print(f"rain evaporation at saturation, {name}:",
                  s["rain_evaporation_at_saturation"])
        if "conditioning" in s:
            print(f"rain self-collection conditioning, {name}:",
                  s["conditioning"])
    for stage in sums[0]["stages"]:
        rows = {v: _pool([s["stages"][stage][v] for s in sums])
                for v in sums[0]["stages"][stage]}
        table(f"STAGE STATE after {stage}", rows, cols)
    tcols = ["n_moved", "gap_ulp_p99", "gap_ulp_max", "n_gap_gt_4ulp",
             "n_gap_gt_64ulp"]
    for key in sums[0]["stage_tendencies"]:
        rows = {v: _pool([s["stage_tendencies"][key][v] for s in sums])
                for v in sums[0]["stage_tendencies"][key]}
        for v in rows:
            for s_ in sums:
                e = s_["stage_tendencies"][key][v]
                for k2 in ("gap_ulp_p99", "gap_ulp_max"):
                    if k2 in e:
                        rows[v][k2] = max(rows[v].get(k2, 0.0), e[k2])
        table(f"STAGE TENDENCY {key} (gap in float32 ulps of the state)",
              rows, tcols)
    rows = {v: _pool([s["final"][v] for s in sums])
            for v in sums[0]["final"] if v != "refl_dbz_abs"}
    table("FINAL", rows, cols)
    table("FINAL, cells beyond the 2e-6 rounding gate, classified",
          {v: _pool([s["final"][v]["class"] for s in sums])
           for v in sums[0]["final"] if "class" in sums[0]["final"][v]},
          CLASS_COLS)
    for s in sums:
        print("named differences,", s["columns"].split("/")[-1] + ":",
              s.get("attribution"))
    rows = {v: _pool([s["final_clean"][v] for s in sums])
            for v in sums[0]["final_clean"]}
    print("\ncolumns clean before sedimentation:",
          sum(s["columns_clean_before_sedimentation"] for s in sums))
    table("FINAL over columns clean before sedimentation", rows, cols)
    for s in sums:
        print("refl", s["columns"].split("/")[-1], s["final"]["refl_dbz_abs"])
    print("\nworst cells (rates with n_rel_gt_2e-6 > 0):")
    for name, e in rates.items():
        if e.get("n_rel_gt_2e-6", 0):
            print(f"  {name}: {e['worst']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
