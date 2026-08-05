#!/usr/bin/env python3
"""Moist instrument for WRF em_les output (P1 oracle side).

WRITTEN AND VALIDATED BEFORE ANY MATCHED MOIST RUN EXISTED.  Every definition
below -- the virtual-temperature convention, the saturation formula, the
engagement predicate, the cloud threshold, the averaging window, the fit bands
-- is transcribed from an authority (WRF v4.6.1 source, quoted inline) or
carried over unchanged from the already-committed dry scorer.  See
INSTRUMENT-HISTORY.md; this file's entry says what it says because the order
is auditable.

Why a NEW file rather than an edit of ``score_wrf_les.py``: the dry scorer has
seen dry output and its provenance is committed.  Editing it to add moist
metrics would put the dry lane's numbers and the moist lane's numbers behind
one mutable artifact.  This module adds only what the dry one cannot see
(§3.2 item 2 of the LES completion spec); run both on a moist run.

No gpuwm import, by construction -- the oracle side must be able to disagree
with the engine it scores.

Usage:  score_moist_les.py <run_dir> <out_prefix> [--window-min N]
"""
import sys
import os
import json
import glob
import hashlib

import numpy as np
from netCDF4 import Dataset

# ---------------------------------------------------------------------------
# Constants, verbatim from WRF v4.6.1 share/module_model_constants.F.  They
# are repeated here rather than approximated because the engagement predicate
# below is a conformance statement, not an estimate.
G = 9.81
T0 = 300.0              # WRF base potential temperature (t0)
R_D = 287.0
R_V = 461.6
CP = 1004.5             # 7*r_d/2
XLV = 2.5e6
P1000MB = 1.0e5
SVP1 = 0.6112           # kPa
SVP2 = 17.67
SVP3 = 29.65            # K
SVPT0 = 273.15          # K
EP_1 = R_V / R_D - 1.0  # 0.60829...  virtual-temperature coefficient
EP_2 = R_D / R_V        # 0.62198...
RCP = R_D / CP

# dyn_em/module_diffusion_em.F:1544 -- `qc_cr = 0.00001  ! in Kg/Kg`.  The
# same constant is the cloud-presence threshold here, deliberately: the
# saturated-BN2 arm and "there is cloud at this point" then have one
# predicate instead of two that can disagree.
QC_CR = 1.0e-5


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def slab_mean(a):
    return a.astype(np.float64).mean(axis=(1, 2))


def qvs_wrf(t_k, p_pa):
    """Saturation mixing ratio, WRF's own formula.

    dyn_em/module_diffusion_em.F:1626-1631, verbatim:
        tc  = t - SVPT0
        es  = 1000.0 * SVP1 * EXP( SVP2*tc / (t - SVP3) )
        qvs = EP_2 * es / (p - es)
    """
    tc = t_k - SVPT0
    es = 1000.0 * SVP1 * np.exp(SVP2 * tc / (t_k - SVP3))
    return EP_2 * es / (p_pa - es)


def sgs_flux_fnm(field, khv, fnm, fnp, zm, nz):
    """SGS flux of a mass-point field on interior w-levels.

    Identical construction to the dry scorer's `wth_sgs` (score_wrf_les.py:
    108-117) -- WRF's fnm/fnp vertical interpolation of XKHV onto the w-level,
    times the centred gradient.  Applied here to theta_v and to qv.  Reusing
    the construction rather than re-deriving it is the point: a moist-vs-dry
    difference must not be a difference in how the SGS term was assembled.
    """
    out = np.zeros(nz + 1)
    for kw in range(1, nz):
        rdz = 1.0 / (zm[kw] - zm[kw - 1])
        kw_c = fnm[kw] * khv[kw] + fnp[kw] * khv[kw - 1]
        out[kw] = np.mean(-kw_c * (field[kw] - field[kw - 1]) * rdz)
    return out


def frame(nc, it, nz, ny, nx, has_tke):
    """Every moist profile for one history frame."""
    th = nc.variables["T"][it].astype(np.float64) + T0
    W = nc.variables["W"][it].astype(np.float64)
    PH = nc.variables["PH"][it].astype(np.float64)
    PHB = nc.variables["PHB"][it].astype(np.float64)
    P = nc.variables["P"][it].astype(np.float64)
    PB = nc.variables["PB"][it].astype(np.float64)
    p_full = P + PB
    t_k = th * (p_full / P1000MB) ** RCP

    def moist_var(name):
        if name in nc.variables:
            return nc.variables[name][it].astype(np.float64)
        return np.zeros_like(th)

    qv = moist_var("QVAPOR")
    qc = moist_var("QCLOUD")
    qr = moist_var("QRAIN")
    qi = moist_var("QICE")          # zero under Kessler; read anyway

    zw = slab_mean((PH + PHB) / G)
    zm = 0.5 * (zw[:-1] + zw[1:])

    # theta_v, both conventions.  The dry scorer computes both unstagger
    # conventions for the same reason: a definitional choice must never be
    # mistakable for a physics difference.
    #   *_novload : theta*(1 + EP_1*qv)              -- vapour only
    #   *_load    : theta*(1 + EP_1*qv - qc - qr - qi) -- with condensate loading
    thv_novload = th * (1.0 + EP_1 * qv)
    thv_load = th * (1.0 + EP_1 * qv - qc - qr - qi)

    wm = 0.5 * (W[:-1] + W[1:])
    wp = wm - slab_mean(wm)[:, None, None]

    out = {}
    out["z_mass"] = zm
    out["z_w"] = zw
    out["qv"] = slab_mean(qv)
    out["qc"] = slab_mean(qc)
    out["qr"] = slab_mean(qr)
    out["theta"] = slab_mean(th)
    out["thetav_load"] = slab_mean(thv_load)
    out["thetav_novload"] = slab_mean(thv_novload)

    for tag, fld in (("load", thv_load), ("novload", thv_novload)):
        fp = fld - slab_mean(fld)[:, None, None]
        out["wthv_res_" + tag] = slab_mean(wp * fp)
    qvp = qv - slab_mean(qv)[:, None, None]
    out["wqv_res"] = slab_mean(wp * qvp)

    zeros = np.zeros(nz + 1)
    if "XKHV" in nc.variables:
        khv = nc.variables["XKHV"][it].astype(np.float64)
        fnm = nc.variables["FNM"][it].astype(np.float64)
        fnp = nc.variables["FNP"][it].astype(np.float64)
        out["wthv_sgs_load"] = sgs_flux_fnm(thv_load, khv, fnm, fnp, zm, nz)
        out["wthv_sgs_novload"] = sgs_flux_fnm(thv_novload, khv, fnm, fnp,
                                               zm, nz)
        out["wqv_sgs"] = sgs_flux_fnm(qv, khv, fnm, fnp, zm, nz)
    else:
        out["wthv_sgs_load"] = zeros.copy()
        out["wthv_sgs_novload"] = zeros.copy()
        out["wqv_sgs"] = zeros.copy()

    # --- saturated-branch engagement, WRF's exact predicate.
    # dyn_em/module_diffusion_em.F:1636 --
    #   IF ( moist(i,k,j,P_QV) .GE. qvs(i,k,j) .OR. qctmp(i,k,j) .GE. qc_cr )
    qvs = qvs_wrf(t_k, p_full)
    engaged = (qv >= qvs) | (qc >= QC_CR)
    out["n2_moist_frac"] = engaged.mean(axis=(1, 2)).astype(np.float64)
    out["sat_frac"] = (qv >= qvs).mean(axis=(1, 2)).astype(np.float64)
    out["cloud_frac"] = (qc >= QC_CR).mean(axis=(1, 2)).astype(np.float64)
    out["rh"] = slab_mean(qv / np.maximum(qvs, 1e-30))

    # --- condensate paths from the model's own dry-mass coordinate:
    # layer dry mass per unit area = -(MU+MUB)*DNW(k)/g.
    MU = nc.variables["MU"][it].astype(np.float64)
    MUB = nc.variables["MUB"][it].astype(np.float64)
    DNW = nc.variables["DNW"][it].astype(np.float64)
    mu = MU + MUB
    dm = (-DNW)[:, None, None] * mu[None, :, :] / G     # (nz,ny,nx) kg/m2
    out["lwp"] = float((qc * dm).sum(axis=0).mean())
    out["rwp"] = float((qr * dm).sum(axis=0).mean())
    out["vip"] = float((qv * dm).sum(axis=0).mean())    # vapour path
    out["mass"] = float(mu.sum())

    # --- LCL of the near-surface slab-mean parcel, lifted along the model's
    # own pressure profile.  This is what decides whether a sounding can make
    # a cloud-topped CBL at all, so it is reported every frame.
    p_bar = slab_mean(p_full)
    th_sfc = float(out["theta"][0])
    qv_sfc = float(out["qv"][0])
    t_parcel = th_sfc * (p_bar / P1000MB) ** RCP
    qvs_parcel = qvs_wrf(t_parcel, p_bar)
    lcl = float("nan")
    hit = np.where(qv_sfc >= qvs_parcel)[0]
    if len(hit):
        lcl = float(zm[hit[0]])
    out["lcl_m"] = lcl
    out["qvs_parcel_min_over_qv"] = float(qvs_parcel.min() / max(qv_sfc, 1e-30))

    if has_tke and "TKE" in nc.variables:
        out["e_sgs"] = slab_mean(nc.variables["TKE"][it].astype(np.float64))
    else:
        out["e_sgs"] = np.zeros(nz)

    out["qc_max"] = float(qc.max())
    out["qr_max"] = float(qr.max())
    if "RAINNC" in nc.variables:
        rain = nc.variables["RAINNC"][it].astype(np.float64)
        out["rainnc_mean"] = float(rain.mean())
        out["rainnc_max"] = float(rain.max())
    else:
        out["rainnc_mean"] = 0.0
        out["rainnc_max"] = 0.0
    return out


def main():
    run_dir = sys.argv[1]
    out_prefix = sys.argv[2]
    window_min = 30.0
    if "--window-min" in sys.argv:
        window_min = float(sys.argv[sys.argv.index("--window-min") + 1])

    files = sorted(glob.glob(os.path.join(run_dir, "wrfout_d01_*")))
    if not files:
        print("no wrfout in %s" % run_dir)
        return 2

    nml = open(os.path.join(run_dir, "namelist.input")).read()

    def nml_get(key, cast=float, default=None):
        """First column of a namelist entry.

        The dry lane's namelists are single-domain, so its reader takes the
        whole right-hand side.  WRF's shipped em_quarter_ss namelist writes
        per-domain columns (`km_opt = 2, 2, 2,`), so the first column is
        taken here.  This is a parser difference and touches no metric.
        """
        for line in nml.splitlines():
            s = line.strip()
            if s.startswith(key):
                after = s.split("=", 1)
                if len(after) == 2:
                    return cast(after[1].split(",")[0].strip())
        return default

    # No default for the prescribed surface heat flux.  The dry lane's scorer
    # defaults it to 0.24 because every namelist in that lane sets it; the P1
    # secondary arm (em_quarter_ss as shipped) does NOT set it, and dividing
    # its buoyancy flux by a value it never had would manufacture a number.
    # Where it is absent the /Qs entries are reported absent and the raw
    # fluxes are reported instead.
    Qs = nml_get("tke_heat_flux", float, None)
    km_opt = nml_get("km_opt", int, 3)
    mp_physics = nml_get("mp_physics", int, 0)
    dx = nml_get("dx", float, 100.0)

    frames = []
    tmin = []
    nz = ny = nx = None
    for f in files:
        d = Dataset(f)
        xt = d.variables["XTIME"][:].astype(np.float64)
        if nz is None:
            nz = d.dimensions["bottom_top"].size
            ny = d.dimensions["south_north"].size
            nx = d.dimensions["west_east"].size
        for it in range(len(xt)):
            frames.append(frame(d, it, nz, ny, nx, km_opt == 2))
            tmin.append(float(xt[it]))
        d.close()

    tmin = np.array(tmin)
    t_end = tmin.max()
    idx = np.where(tmin >= (t_end - window_min - 1e-9))[0]
    nwin = len(idx)

    def avg(key):
        return np.mean([frames[i][key] for i in idx], axis=0)

    zm = avg("z_mass")
    zw = avg("z_w")

    def total(res, sgs):
        return res + 0.5 * (sgs[:-1] + sgs[1:])

    tot_load = total(avg("wthv_res_load"), avg("wthv_sgs_load"))
    tot_nov = total(avg("wthv_res_novload"), avg("wthv_sgs_novload"))
    zi_load = float(zm[int(np.argmin(tot_load))])
    zi_nov = float(zm[int(np.argmin(tot_nov))])

    cloud = avg("cloud_frac")
    engaged = avg("n2_moist_frac")
    satf = avg("sat_frac")

    def band_edges(profile, thresh):
        hit = np.where(profile >= thresh)[0]
        if not len(hit):
            return None, None
        return float(zm[hit[0]]), float(zm[hit[-1]])

    cb, ct = band_edges(cloud, 0.01)

    # --- whole-run scan.  A window average can hide a case that condensed
    # once and stopped, and the question "does this sounding ever condense"
    # is exactly the question a probe run is asked.
    qc_max_run = max(f["qc_max"] for f in frames)
    qr_max_run = max(f["qr_max"] for f in frames)
    first_cloud = None
    for i, f in enumerate(frames):
        if f["qc_max"] >= QC_CR:
            first_cloud = float(tmin[i])
            break
    engaged_run_max = max(float(np.asarray(f["n2_moist_frac"]).max())
                          for f in frames)

    # Clamp-coverage discipline (tests/test_smag2d.py), promoted to a run
    # receipt: an instrument that reports "engaged everywhere" or "engaged
    # nowhere" has not demonstrated it can see the switch.
    eng_max = float(engaged.max())
    eng_min = float(engaged.min())
    engagement = dict(
        predicate="qv >= qvs .OR. qc >= 1e-5  "
                  "(dyn_em/module_diffusion_em.F:1636, v4.6.1)",
        qc_cr=QC_CR,
        engaged_somewhere=bool(eng_max > 0.0),
        engaged_everywhere=bool(eng_min >= 1.0),
        max_level_fraction_window=eng_max,
        max_level_fraction_run=engaged_run_max,
        n_levels_engaged_window=int((engaged > 0.0).sum()),
    )

    lcl_win = float(np.mean([frames[i]["lcl_m"] for i in idx]))
    lcl_series = [f["lcl_m"] for f in frames]

    res = dict(
        run_dir=os.path.abspath(run_dir),
        wrfout_files=[os.path.basename(f) for f in files],
        wrfout_sha256={os.path.basename(f): sha256_file(f) for f in files},
        wrfout_bytes={os.path.basename(f): os.path.getsize(f) for f in files},
        km_opt=km_opt, mp_physics=mp_physics, dx_m=dx, nx=nx, ny=ny, nz=nz,
        surface_heat_flux=Qs,
        n_frames=len(frames), n_frames_in_window=nwin,
        window_minutes=window_min, t_end_minutes=float(t_end),
        # --- buoyancy-flux z_i, both theta_v conventions
        zi_thetav_load_m=zi_load,
        zi_thetav_novload_m=zi_nov,
        # --- the moist headline set
        wthv_res_max=float(avg("wthv_res_load").max()),
        wthv_total_min=float(tot_load.min()),
        wthv_res_max_over_qs=(float(avg("wthv_res_load").max() / Qs)
                              if Qs else None),
        wthv_total_min_over_qs=(float(tot_load.min() / Qs) if Qs else None),
        wqv_res_max=float(avg("wqv_res").max()),
        wqv_sgs_max=float(np.abs(avg("wqv_sgs")).max()),
        qv_surface=float(avg("qv")[0]),
        qc_max_profile=float(avg("qc").max()),
        qc_max_pointwise_run=qc_max_run,
        qr_max_pointwise_run=qr_max_run,
        first_cloud_minutes=first_cloud,
        cloud_fraction_max=float(cloud.max()),
        cloud_base_m=cb, cloud_top_m=ct,
        sat_fraction_max=float(satf.max()),
        lwp_kg_m2=float(np.mean([frames[i]["lwp"] for i in idx])),
        rwp_kg_m2=float(np.mean([frames[i]["rwp"] for i in idx])),
        rainnc_mm_end=float(frames[-1]["rainnc_mean"]),
        rainnc_mm_max_point=float(frames[-1]["rainnc_max"]),
        lcl_m_window=lcl_win,
        lcl_m_first=float(lcl_series[0]),
        lcl_over_zi=(lcl_win / zi_load) if zi_load > 0 else None,
        saturated_branch=engagement,
        mass_drift_rel=float(max(abs(f["mass"] - frames[0]["mass"])
                                 for f in frames) / abs(frames[0]["mass"])),
        conventions=dict(
            theta_v_load="theta*(1 + EP_1*qv - qc - qr - qi)",
            theta_v_novload="theta*(1 + EP_1*qv)",
            EP_1=EP_1, EP_2=EP_2,
            sgs_flux="fnm/fnp interpolation of XKHV onto w-levels times the "
                     "centred gradient (same construction as the dry "
                     "scorer's wth_sgs)",
            cloud_threshold_kg_kg=QC_CR,
            window="final %g minutes" % window_min,
        ),
    )

    with open(out_prefix + ".json", "w", newline="\n") as fh:
        json.dump(res, fh, indent=2, sort_keys=True)

    def stack(key):
        return np.array([f[key] for f in frames])

    np.savez_compressed(
        out_prefix + "_moist_profiles.npz",
        z_mass=zm, z_w=zw, t_seconds=tmin * 60.0,
        # per-frame stacks: the same-instrument reducer needs frames, not a
        # pre-averaged profile, so both sides can be reduced by one routine.
        wthv_res=stack("wthv_res_load"), wthv_sgs=stack("wthv_sgs_load"),
        wthv_res_novload=stack("wthv_res_novload"),
        wthv_sgs_novload=stack("wthv_sgs_novload"),
        wqv_res=stack("wqv_res"), wqv_sgs=stack("wqv_sgs"),
        qv=stack("qv"), qc=stack("qc"), qr=stack("qr"),
        theta=stack("theta"), thetav=stack("thetav_load"),
        cloud_frac=stack("cloud_frac"), sat_frac=stack("sat_frac"),
        n2_moist_frac=stack("n2_moist_frac"), rh=stack("rh"),
        e_sgs=stack("e_sgs"),
        lwp=np.array([f["lwp"] for f in frames]),
        rwp=np.array([f["rwp"] for f in frames]),
        lcl_m=np.array(lcl_series),
        rainnc=np.array([f["rainnc_mean"] for f in frames]),
    )

    for k in ("mp_physics", "zi_thetav_load_m", "lcl_m_window", "lcl_over_zi",
              "qc_max_pointwise_run", "qr_max_pointwise_run",
              "first_cloud_minutes", "cloud_fraction_max", "cloud_base_m",
              "cloud_top_m", "lwp_kg_m2", "rainnc_mm_end", "wthv_res_max",
              "wthv_res_max_over_qs", "wqv_res_max", "mass_drift_rel"):
        print("%-28s %s" % (k, res[k]))
    print("%-28s %s" % ("saturated_branch", json.dumps(engagement)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
