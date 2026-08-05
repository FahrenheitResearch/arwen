#!/usr/bin/env python3
"""Is the cloud deck stationary, clear of the lid, and owned by the flow?

The three quantities the capped-family decider is accepted on.  They are
computed from the arrays `score_moist_les.py` already writes -- this reads
`<prefix>_moist_profiles.npz` and adds no new model diagnostic -- so the
instrument is unchanged and this is a reduction of committed output.

Definitions, fixed before the capped runs existed:

  instantaneous cloud top  the highest mass level whose cloud fraction in
                           THAT frame is >= 0.01.  The 0.01 threshold is the
                           committed instrument's own `cloud_top_m`
                           threshold; only the averaging differs (per frame
                           here, window-mean there).
  trend                    least-squares slope of instantaneous cloud top
                           against time over the final `--window-min` minutes,
                           reported in m/h.
  within-run sd            standard deviation (ddof=1) of the same series over
                           the same window, in metres and in model levels.
  lid margin               window-mean cloud top divided by ztop.

Why the within-run sd matters: `cloud_top_m` is quantised to the model level
spacing (about 40 m on this grid), so two draws landing on the same level is
not by itself evidence of a pinned deck.  A deck pinned to a rigid boundary
has no within-run fluctuation either -- the uncapped case's top level sat at
cloud fraction 1.0 for the last third of the run.  A deck held by a
deformable inversion fluctuates.  The two tests together can tell those apart;
either alone cannot.

Nothing here is a band.  The thresholds live in the receipt that registered
them, not in this file, so the file cannot be edited into agreement.

Usage: deck_metrics.py <prefix>_moist_profiles.npz [--window-min 60]
                       [--json out.json] [--ztop 2400]
"""
import sys
import json

import numpy as np

CLOUD_FRAC_THRESHOLD = 0.01


def main():
    path = sys.argv[1]
    win = 60.0
    if "--window-min" in sys.argv:
        win = float(sys.argv[sys.argv.index("--window-min") + 1])
    ztop = None
    if "--ztop" in sys.argv:
        ztop = float(sys.argv[sys.argv.index("--ztop") + 1])

    d = np.load(path)
    z = np.asarray(d["z_mass"], dtype=np.float64)
    t_min = np.asarray(d["t_seconds"], dtype=np.float64) / 60.0
    cf = np.asarray(d["cloud_frac"], dtype=np.float64)
    lwp = np.asarray(d["lwp"], dtype=np.float64)

    dz = float(np.median(np.diff(z)))

    tops = []
    for i in range(cf.shape[0]):
        hit = np.where(cf[i] >= CLOUD_FRAC_THRESHOLD)[0]
        tops.append(float(z[hit[-1]]) if len(hit) else float("nan"))
    tops = np.array(tops)

    sel = t_min >= (t_min.max() - win - 1e-9)
    ts = t_min[sel]
    ys = tops[sel]
    ok = ~np.isnan(ys)
    n = int(ok.sum())

    if n >= 2:
        slope_per_min = float(np.polyfit(ts[ok], ys[ok], 1)[0])
        trend_m_per_h = slope_per_min * 60.0
        sd_m = float(np.std(ys[ok], ddof=1))
    else:
        trend_m_per_h = float("nan")
        sd_m = float("nan")

    lw = lwp[sel]
    if len(lw) >= 2 and lw.mean() > 0:
        lwp_slope = float(np.polyfit(ts, lw, 1)[0]) * 60.0
        lwp_trend_pct_per_h = 100.0 * lwp_slope / lw.mean()
    else:
        lwp_trend_pct_per_h = float("nan")

    mean_top = float(np.nanmean(ys)) if n else float("nan")
    out = dict(
        npz=path,
        window_min=win, n_frames_in_window=n,
        dz_m=dz,
        cloud_frac_threshold=CLOUD_FRAC_THRESHOLD,
        cloud_top_mean_m=mean_top,
        cloud_top_trend_m_per_h=trend_m_per_h,
        cloud_top_sd_m=sd_m,
        cloud_top_sd_levels=(sd_m / dz if dz else float("nan")),
        cloud_top_min_m=float(np.nanmin(ys)) if n else None,
        cloud_top_max_m=float(np.nanmax(ys)) if n else None,
        cloud_top_range_levels=(float((np.nanmax(ys) - np.nanmin(ys)) / dz)
                                if n and dz else None),
        lwp_mean_kg_m2=float(lw.mean()),
        lwp_trend_pct_per_h=lwp_trend_pct_per_h,
    )
    if ztop:
        out["ztop_m"] = ztop
        out["cloud_top_over_ztop"] = mean_top / ztop

    for k, v in out.items():
        print("%-26s %s" % (k, v))
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w",
                  newline="\n") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
