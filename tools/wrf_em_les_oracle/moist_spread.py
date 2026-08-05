#!/usr/bin/env python3
"""Realisation spread of the moist metrics across draws (P1).

WRF's em_les initial perturbation is unseeded and decomposition-ordered, so
two runs of the same namelist are two draws from the same distribution and
nothing else.  This aggregates the per-run receipts that `score_moist_les.py`
already wrote and reports, per metric: n, mean, sample standard deviation,
min, max, and the coefficient of variation.

**It computes no band, no threshold, and no verdict.** It is arithmetic over
committed numbers -- deliberately so, because it was written after the data
existed and anything it produced that could gate a result would be a band cut
after seeing output. Bands are ratified by the owner from committed receipts;
this only makes the receipts summable.

`sd` is the sample standard deviation (ddof=1) and is reported as `null` for
n < 2 rather than as 0.0, because one draw has no spread and saying it does is
the same error class as an absent denominator reading 1.0.

Usage: moist_spread.py <label> <out.json> <run1.json> <run2.json> ...
"""
import sys
import json
import math


METRICS = [
    "zi_thetav_load_m", "zi_thetav_novload_m",
    "wthv_res_max", "wthv_total_min",
    "wthv_res_max_over_qs", "wthv_total_min_over_qs",
    "wqv_res_max", "wqv_sgs_max",
    "qv_surface", "qc_max_profile", "qc_max_pointwise_run",
    "qr_max_pointwise_run", "first_cloud_minutes",
    "cloud_fraction_max", "cloud_base_m", "cloud_top_m",
    "sat_fraction_max", "lwp_kg_m2", "rwp_kg_m2",
    "rainnc_mm_end", "lcl_m_window", "lcl_over_zi",
    "mass_drift_rel",
]
NESTED = [("saturated_branch", "max_level_fraction_window"),
          ("saturated_branch", "n_levels_engaged_window")]


def stats(vals):
    vals = [v for v in vals if v is not None
            and not (isinstance(v, float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return dict(n=0, mean=None, sd=None, min=None, max=None, cv=None)
    mean = sum(vals) / n
    if n < 2:
        return dict(n=n, mean=mean, sd=None, min=min(vals), max=max(vals),
                    cv=None)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    return dict(n=n, mean=mean, sd=sd, min=min(vals), max=max(vals),
                cv=(sd / abs(mean) if mean else None))


def main():
    label, out_path = sys.argv[1], sys.argv[2]
    paths = sys.argv[3:]
    runs = [json.load(open(p)) for p in paths]

    rows = {}
    for m in METRICS:
        rows[m] = stats([r.get(m) for r in runs])
    for outer, inner in NESTED:
        rows["%s.%s" % (outer, inner)] = stats(
            [r.get(outer, {}).get(inner) for r in runs])

    # Configuration identity: a spread over draws is only a spread if the
    # draws are the same configuration.  Anything that differs is reported,
    # not assumed away.
    ident_keys = ["km_opt", "mp_physics", "nx", "ny", "nz", "dx_m",
                  "surface_heat_flux", "window_minutes", "t_end_minutes",
                  "n_frames_in_window"]
    identity = {}
    for k in ident_keys:
        vals = sorted({json.dumps(r.get(k)) for r in runs})
        identity[k] = json.loads(vals[0]) if len(vals) == 1 else \
            {"DIFFERS": [json.loads(v) for v in vals]}
    differing = [k for k, v in identity.items()
                 if isinstance(v, dict) and "DIFFERS" in v]

    engaged = [r.get("saturated_branch", {}) for r in runs]
    out = dict(
        label=label,
        n_draws=len(runs),
        source_receipts=[p.split("/")[-1] for p in paths],
        wrfout_sha256={p.split("/")[-1]: r["wrfout_sha256"]
                       for p, r in zip(paths, runs)},
        configuration=identity,
        configuration_differs_on=differing,
        saturated_branch_all_engaged_somewhere=all(
            e.get("engaged_somewhere") for e in engaged),
        saturated_branch_any_engaged_everywhere=any(
            e.get("engaged_everywhere") for e in engaged),
        metrics=rows,
        note="no band, threshold or verdict is computed here",
    )
    with open(out_path, "w", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    print("=== %s : %d draws ===" % (label, len(runs)))
    if differing:
        print("!! configuration DIFFERS across draws on: %s" % differing)
    print("%-34s %3s %14s %12s %12s %10s"
          % ("metric", "n", "mean", "sd", "cv", "max-min"))
    for m, s in rows.items():
        if s["n"] == 0:
            print("%-34s %3d %14s" % (m, 0, "absent in all draws"))
            continue
        rng = s["max"] - s["min"]
        print("%-34s %3d %14.6g %12s %12s %10.4g"
              % (m, s["n"], s["mean"],
                 "-" if s["sd"] is None else "%.4g" % s["sd"],
                 "-" if s["cv"] is None else "%.4f" % s["cv"], rng))


if __name__ == "__main__":
    main()
