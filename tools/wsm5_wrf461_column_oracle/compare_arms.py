"""Attribute every difference between the WRF WSM5 arms (CPU only).

Reads build.sh's outputs and states, per run:

* control: the scalar tree's own object against the stock arm (the same
  source recompiled from the tree's compile line, plus integer probe
  counters).  Must be zero differing words, or the arms prove nothing.
* stock psevp is dead: the stock arm's psevp counter is zero on every
  column of every run (WRF's WSM5 never evaporates melting snow).
* each declared fix differs from stock on EXACTLY the columns its probe
  names: remap arm vs the remap probe (target cells where the stock search
  falls through), psevp arm vs the psevp probe (levels with a non-zero
  corrected psevp), defined arm vs the union of both.
* strict tree vs scalar tree: reported (WRF's vectoriser moves words).
* the transport-only remap probe (lane 073's control) at 2, 50, 80 levels:
  final column + fallout against the initial inventory, per arm.

usage: python compare_arms.py ORACLE_DIR [--out JSON]
Exit status 0 only when every assertion above holds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import oracle_io as io  # type: ignore
else:
    from . import oracle_io as io


def _probe_sets(run, names):
    remap_counts, psevp_counts = run["probes"][-1]
    return ({names[i] for i in np.nonzero(remap_counts)[0]},
            {names[i] for i in np.nonzero(psevp_counts)[0]})


def budget(rows):
    """Lane 073's closure: sum(rq*dz) + precip against 0.001 * 100 * nz."""
    out = []
    for nz, dt, rq, precip in rows:
        dz = np.float64(100.0)
        initial = float(np.sum(np.full(nz, np.float32(0.001), np.float64) * dz))
        final = float(np.sum(rq.astype(np.float64) * dz))
        total = final + precip
        out.append({"nz": nz, "dt": dt, "initial_kg_m2": initial,
                    "final_column_kg_m2": final, "fallout_kg_m2": precip,
                    "lost_kg_m2": initial - total,
                    "relative_error": abs(total - initial) / initial,
                    "top_two_density": [float(v) for v in rq[-2:]]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oracle_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    d = args.oracle_dir
    report = {"runs": {}, "failures": []}
    fail = report["failures"]
    for set_name, dt, nsteps in io.RUNS:
        tag = io.tag_of(set_name, dt, nsteps)
        names = io.column_names(d, set_name)
        arm = {a: io.read_wrf(d / f"wrf-{a}-{tag}.bin") for a in io.ARMS}
        row = {}
        ctrl = io.compare_runs(arm["scalar"], arm["stock"], names)
        row["control_scalar_vs_stock_recompiled"] = sum(
            r["bit_mismatch"] for r in ctrl.values())
        if row["control_scalar_vs_stock_recompiled"]:
            fail.append(f"{tag}: recompiled stock differs from the scalar tree")
        stock_remap, stock_psevp = _probe_sets(arm["stock"], names)
        row["stock_fallthrough_columns"] = sorted(stock_remap)
        row["stock_nonzero_psevp_columns"] = sorted(stock_psevp)
        if stock_psevp:
            fail.append(f"{tag}: stock psevp is non-zero somewhere")
        strict = io.compare_runs(arm["scalar"], arm["strict"], names)
        row["strict_vs_scalar"] = {
            "bit_mismatch": sum(r["bit_mismatch"] for r in strict.values()),
            "columns": sorted(io.mismatch_columns(strict)),
            "fields": {n: [r["bit_mismatch"], r["max_ulp"]] for n, r in strict.items()
                       if r["bit_mismatch"]}}
        for fix in ("remap", "psevp", "defined"):
            diff = io.compare_runs(arm["scalar"], arm[fix], names)
            cols = io.mismatch_columns(diff)
            p_remap, p_psevp = _probe_sets(arm[fix], names)
            want = {"remap": p_remap, "psevp": p_psevp, "defined": p_remap | p_psevp}[fix]
            row[f"{fix}_vs_stock"] = {
                "bit_mismatch": sum(r["bit_mismatch"] for r in diff.values()),
                "columns": sorted(cols),
                "probe_columns": sorted(want),
                "remap_probe_columns": sorted(p_remap),
                "psevp_probe_columns": sorted(p_psevp),
                "equal": cols == want,
                "fields": {n: [r["bit_mismatch"], r["max_ulp"]] for n, r in diff.items()
                           if r["bit_mismatch"]}}
            if cols != want:
                fail.append(f"{tag}: {fix} mismatch columns {sorted(cols ^ want)} "
                            "are not its probe's")
        report["runs"][tag] = row
        print(f"{tag}: control {row['control_scalar_vs_stock_recompiled']}; "
              f"strict {row['strict_vs_scalar']['bit_mismatch']} words; "
              + "; ".join(f"{fix} {row[f'{fix}_vs_stock']['bit_mismatch']} words in "
                          f"{len(row[f'{fix}_vs_stock']['columns'])} cols "
                          f"(probe {len(row[f'{fix}_vs_stock']['probe_columns'])}, "
                          f"equal {row[f'{fix}_vs_stock']['equal']})"
                          for fix in ("remap", "psevp", "defined")), flush=True)
    report["remap_probe"] = {a: budget(io.read_probe(d / f"remap-probe-{a}.bin"))
                             for a in io.ARMS}
    for row in report["remap_probe"]["defined"]:
        if row["relative_error"] >= 1.0e-6:
            fail.append(f"defined remap probe nz {row['nz']} dt {row['dt']}: "
                        f"relative error {row['relative_error']:.3e}")
    for a in ("scalar", "defined"):
        print(a, "remap probe:", "; ".join(
            f"nz{r['nz']} dt{r['dt']:g} lost {r['lost_kg_m2']:.6g} rel {r['relative_error']:.2e}"
            for r in report["remap_probe"][a]))
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"failures": fail}))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
