"""Build the RRTMG LW oracle case file from CPU-reference wrfouts + extremes.

Pulls diverse real columns (clear, thick liquid/ice cloud, snow surface,
water/land, warm/cold) from locally available CPU-reference wrfout files,
read in place, and appends synthetic extreme columns.  Values are written
with 9 significant digits so every FP32 value round-trips exactly.

Each physical column is emitted under multiple wrapper flag regimes so the
fixtures cover every cloud-optics path the WRF v4.6.1 wrapper can select
for the campaign schemes:
  (has_reqc,has_reqi,has_reqs) = (1,1,1) -> inflglw=5, iceflglw=5 (Thompson)
                                 (0,0,0) -> inflglw=2, iceflglw=3 (Morrison)
                                 (1,1,0) -> P3 special case, snow<-ice swap
                                 (1,0,0) -> inflglw=3, iceflglw=3
                                 (0,1,0) -> inflglw=4, iceflglw=4

Usage: python lw_make_inputs.py WRFOUT_DIR OUT_TXT
"""

import glob
import os
import sys

import numpy as np

R_D = 287.0
CP = 1004.5
G = 9.81
AMDO = 0.603461

# O3DATA annual-mean profile (mmr) and pressures (hPa), transcribed from the
# WRF module for realistic o33d input synthesis (input realism only; the
# oracle's own climatology comes from the unmodified INIRAD).
O3SUM = np.array([5.297e-8, 5.852e-8, 6.579e-8, 7.505e-8, 8.577e-8, 9.895e-8,
                  1.175e-7, 1.399e-7, 1.677e-7, 2.003e-7, 2.571e-7, 3.325e-7,
                  4.438e-7, 6.255e-7, 8.168e-7, 1.036e-6, 1.366e-6, 1.855e-6,
                  2.514e-6, 3.240e-6, 4.033e-6, 4.854e-6, 5.517e-6, 6.089e-6,
                  6.689e-6, 1.106e-5, 1.462e-5, 1.321e-5, 9.856e-6, 5.960e-6,
                  5.960e-6])
PPSUM = np.array([955.890, 850.532, 754.599, 667.742, 589.841, 519.421,
                  455.480, 398.085, 347.171, 301.735, 261.310, 225.360,
                  193.419, 165.490, 141.032, 120.125, 102.689, 87.829,
                  75.123, 64.306, 55.086, 47.209, 40.535, 34.795, 29.865,
                  19.122, 9.277, 4.660, 2.421, 1.294, 0.647])


def o3_vmr_profile(p_hpa, scale):
    mmr = np.interp(np.log(np.maximum(p_hpa, PPSUM[-1])),
                    np.log(PPSUM[::-1]), O3SUM[::-1])
    return mmr * AMDO * scale


def f9(x):
    return "%.9e" % np.float32(x)


class Case:
    def __init__(self, caseid, hdr_a, hdr_b, cols, ifaces):
        self.caseid = caseid
        self.hdr_a = hdr_a      # list of ints (after caseid)
        self.hdr_b = hdr_b      # list of floats
        self.cols = cols        # (nz, 16) float array
        self.ifaces = ifaces    # (nz+1, 2) float array


def column_from_wrfout(ds, i, j, rng, re_seedcase):
    nz = ds.dimensions["bottom_top"].size
    p = ds.variables["P"][0, :, j, i] + ds.variables["PB"][0, :, j, i]
    theta = ds.variables["T"][0, :, j, i] + 300.0
    pi3d = (p / 1.0e5) ** (R_D / CP)
    t = theta * pi3d
    ph = ds.variables["PH"][0, :, j, i] + ds.variables["PHB"][0, :, j, i]
    z_w = ph / G
    dz = np.diff(z_w)
    z_m = 0.5 * (z_w[:-1] + z_w[1:])
    qv = np.maximum(ds.variables["QVAPOR"][0, :, j, i], 0.0)
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (R_D * tv)

    def gv(name):
        if name in ds.variables:
            return np.maximum(ds.variables[name][0, :, j, i], 0.0)
        return np.zeros(nz)

    qc, qr, qi = gv("QCLOUD"), gv("QRAIN"), gv("QICE")
    qs, qg = gv("QSNOW"), gv("QGRAUP")
    cldfra = np.clip(ds.variables["CLDFRA"][0, :, j, i], 0.0, 1.0) \
        if "CLDFRA" in ds.variables else np.zeros(nz)

    psfc = float(ds.variables["PSFC"][0, j, i])
    p_top = float(np.asarray(ds.variables["P_TOP"][:]).ravel()[0])
    if not np.all(np.diff(p) < 0):
        return None  # non-monotone column; caller skips it
    # Interfaces from strictly-decreasing midpoints: interleaving guarantees
    # strict monotonicity (a zero-thickness layer would give coldry=0 and a
    # NaN chain inside RRTMG that real WRF dynamics can never produce).
    p8w = np.empty(nz + 1)
    p8w[1:-1] = 0.5 * (p[:-1] + p[1:])
    p8w[0] = max(psfc, float(p[0]) + 1.0)
    p8w[-1] = p_top
    assert np.all(np.diff(p8w) < 0)
    t8w = np.empty(nz + 1)
    t8w[1:-1] = np.interp(z_w[1:-1], z_m, t)
    t8w[0] = t[0] + (t[0] - t8w[1])
    t8w[-1] = t[-1] + (t[-1] - t8w[-2])

    # Effective radii (m) in physical ranges; wrfouts here don't carry RE_*.
    # Cover the wrapper's clamp branches deliberately per seed case.  Upper
    # bounds respect cldprmc's hard aborts (ice dge <= 140 um: WRF fatals
    # beyond it and the coupled schemes stay below; snow is clamped to
    # 130 um by the wrapper itself so large values are legal and exercise
    # the snow_mass_factor branch).
    re_c = np.where(qc > 1e-8,
                    rng.uniform(3.0, 28.0, nz), rng.uniform(0.0, 2.4, nz))
    re_i = np.where(qi > 1e-9,
                    rng.uniform(6.0, 135.0, nz), rng.uniform(0.0, 4.9, nz))
    re_s = np.where(qs > 1e-9,
                    rng.uniform(12.0, 220.0, nz), rng.uniform(0.0, 9.0, nz))
    if re_seedcase % 3 == 1:      # push snow>130 clamp and tiny-ice retab
        re_s = np.where(qs > 1e-9, rng.uniform(120.0, 500.0, nz), re_s)
        re_i = np.where(qi > 1e-9, rng.uniform(0.5, 6.0, nz), re_i)
    if re_seedcase % 3 == 2:      # tiny cloud droplets -> land/ocean branch
        re_c = np.where(qc > 1e-8, rng.uniform(0.5, 2.6, nz), re_c)
    re_c, re_i, re_s = re_c * 1e-6, re_i * 1e-6, re_s * 1e-6

    o3 = o3_vmr_profile(p / 100.0, rng.uniform(0.8, 1.25))

    cols = np.column_stack([p, t, dz, pi3d, rho, qv, qc, qr, qi, qs, qg,
                            cldfra, re_c, re_i, re_s, o3])
    ifaces = np.column_stack([p8w, t8w])

    meta = dict(
        emiss=float(ds.variables["EMISS"][0, j, i]),
        tsk=float(ds.variables["TSK"][0, j, i]),
        psfc=psfc,
        xland=float(ds.variables["XLAND"][0, j, i]),
        xice=float(ds.variables["XICE"][0, j, i]) if "XICE" in ds.variables else 0.0,
        snow=float(ds.variables["SNOW"][0, j, i]),
        snowh=float(ds.variables["SNOWH"][0, j, i]),
        xlat=float(ds.variables["XLAT"][0, j, i]),
        xlong=float(ds.variables["XLONG"][0, j, i]),
        p_top=p_top,
    )
    return cols, ifaces, meta


def pick_columns(ds):
    """Indices (i, j, label) of diverse columns in one wrfout time slab."""
    qc = ds.variables["QCLOUD"][0]
    qi = ds.variables["QICE"][0]
    qs = ds.variables["QSNOW"][0]
    qv = ds.variables["QVAPOR"][0]
    cf = ds.variables["CLDFRA"][0]
    tsk = ds.variables["TSK"][0]
    snow = ds.variables["SNOW"][0]
    xland = ds.variables["XLAND"][0]
    colqc = qc.sum(axis=0)
    colqi = qi.sum(axis=0)
    colqs = qs.sum(axis=0)
    colcf = cf.max(axis=0)
    colall = colqc + colqi + colqs

    picks = []

    def add(mask2d, field, mode, label):
        f = np.where(mask2d, field, -np.inf if mode == "max" else np.inf)
        idx = np.argmax(f) if mode == "max" else np.argmin(f)
        j, i = np.unravel_index(idx, field.shape)
        picks.append((int(i), int(j), label))

    ny, nx = tsk.shape
    interior = np.zeros((ny, nx), bool)
    interior[8:-8, 8:-8] = True
    add(interior, colqc, "max", "thick-liquid")
    add(interior, colqi, "max", "thick-ice")
    add(interior, colqs, "max", "snowy-cloud")
    add(interior & (colcf > 0), colall, "min", "thin-cloud")
    add(interior, colall + colcf, "min", "clear")
    add(interior, tsk, "max", "hot-sfc")
    add(interior, tsk, "min", "cold-sfc")
    add(interior & (snow > 1.0), colall, "max", "snow-sfc") \
        if (snow[interior] > 1.0).any() else None
    add(interior & (xland > 1.5), colqc, "max", "water-cloudy") \
        if (xland[interior] > 1.5).any() else None
    add(interior, qv.sum(axis=0), "max", "very-moist")
    return picks


def synth_column(nz, kind, rng, psfc=97000.0):
    """Synthetic extreme columns on an eta-like 49-layer grid."""
    ptop = 10000.0
    eta_w = np.linspace(1.0, 0.0, nz + 1) ** 1.4
    p8w = ptop + eta_w * (psfc - ptop)
    p = 0.5 * (p8w[:-1] + p8w[1:])

    def t_std(pp):
        z = 44330.0 * (1.0 - (pp / 101325.0) ** 0.1903)
        return np.where(z < 11000.0, 288.15 - 0.0065 * z, 216.65)

    t = t_std(p)
    t8w = t_std(p8w)
    qv = np.maximum(1.0e-9, 0.014 * (p / psfc) ** 3)
    qc = np.zeros(nz); qr = np.zeros(nz); qi = np.zeros(nz)
    qs = np.zeros(nz); qg = np.zeros(nz); cf = np.zeros(nz)
    tsk, emiss, xland, snow, snowh, xice = 292.0, 0.97, 1.0, 0.0, 0.0, 0.0

    if kind == "isothermal-cold":
        t[:] = 200.0; t8w[:] = 200.0; tsk = 200.0; qv[:] = 1.0e-9
    elif kind == "hot-dry":
        t = t + 25.0; t8w = t8w + 25.0; tsk = 330.0; qv *= 0.05
    elif kind == "deck-full":
        cf[:] = 1.0
        qc[:] = 4.0e-4
        qi[:] = np.where(t < 263.0, 3.0e-4, 0.0)
        qs[:] = np.where(t < 268.0, 8.0e-4, 0.0)
        tsk = 285.0
    elif kind == "thin-wisp":
        k = int(0.55 * nz)
        cf[k] = 0.004; qc[k] = 1.0e-7
    elif kind == "top-layer-cloud":
        cf[-1] = 0.9; qi[-1] = 2.0e-4; qs[-1] = 1.0e-4
    elif kind == "sfc-cloud":
        cf[0] = 1.0; qc[0] = 6.0e-4
    elif kind == "huge-snow":
        kk = slice(int(0.2 * nz), int(0.6 * nz))
        cf[kk] = 0.95; qs[kk] = 1.0e-2; qi[kk] = 1.0e-3
        tsk = 270.0; snow = 20.0; snowh = 0.02
    elif kind == "cold-sfc-inversion":
        tsk = t[0] - 30.0; emiss = 0.985; snow = 120.0; snowh = 0.4
    elif kind == "warm-rain-only":
        cf[2:10] = 0.7; qc[2:10] = 3.0e-4; qr[2:10] = 1.0e-3
    elif kind == "mixed-phase-deep":
        kk = slice(2, int(0.8 * nz))
        cf[kk] = rng.uniform(0.2, 1.0, kk.stop - kk.start)
        qc[kk] = rng.uniform(0, 5e-4, kk.stop - kk.start)
        qi[kk] = rng.uniform(0, 4e-4, kk.stop - kk.start)
        qs[kk] = rng.uniform(0, 6e-4, kk.stop - kk.start)
        qg[kk] = rng.uniform(0, 3e-4, kk.stop - kk.start)
        qr[kk] = rng.uniform(0, 2e-4, kk.stop - kk.start)
    elif kind == "ocean-warm":
        xland = 2.0; tsk = 302.0; emiss = 0.985
        cf[3:8] = 0.6; qc[3:8] = 2.5e-4

    pi3d = (p / 1.0e5) ** (R_D / CP)
    tv = t * (1.0 + 0.608 * qv)
    rho = p / (R_D * tv)
    dz = -np.diff(p8w) / (rho * G)
    re_c = np.where(qc > 0, rng.uniform(2.0, 30.0, nz), 0.0) * 1e-6
    re_i = np.where(qi > 0, rng.uniform(4.0, 135.0, nz), 0.0) * 1e-6
    re_s = np.where(qs > 0, rng.uniform(8.0, 400.0, nz), 0.0) * 1e-6
    o3 = o3_vmr_profile(p / 100.0, rng.uniform(0.5, 2.0))
    cols = np.column_stack([p, t, dz, pi3d, rho, qv, qc, qr, qi, qs, qg,
                            cf, re_c, re_i, re_s, o3])
    ifaces = np.column_stack([p8w, t8w])
    meta = dict(emiss=emiss, tsk=tsk, psfc=psfc, xland=xland, xice=xice,
                snow=snow, snowh=snowh, xlat=36.5, xlong=-97.0,
                p_top=ptop)
    return cols, ifaces, meta


FLAG_REGIMES = [(1, 1, 1), (0, 0, 0), (1, 1, 0), (1, 0, 0), (0, 1, 0)]


def main():
    import netCDF4

    wrfdir, out_txt = sys.argv[1], sys.argv[2]
    rng = np.random.default_rng(19740403)

    files = []
    for dom, hours in (("d01", ["13", "18", "23"]), ("d02", ["18"]),
                       ("d03", ["16", "21"])):
        for h in hours:
            pat = os.path.join(wrfdir, f"wrfout_{dom}_1974-04-0*_{h}_00_00")
            m = sorted(glob.glob(pat))
            if m:
                files.append((dom, h, m[0]))

    cases = []
    caseid = 0
    ncol_seen = 0
    nz_ref = None

    for dom, h, path in files:
        ds = netCDF4.Dataset(path)
        nz = ds.dimensions["bottom_top"].size
        if nz_ref is None:
            nz_ref = nz
        assert nz == nz_ref, "mixed vertical dims"
        yr, julday = 1974, 93 + (0 if "04-03" in path else 1)
        julian = float(julday) + (int(h) / 24.0)
        for (i, j, label) in pick_columns(ds):
            got = column_from_wrfout(ds, i, j, rng, caseid)
            if got is None:
                continue
            cols, ifaces, meta = got
            # cycle flag regimes by a dedicated column counter; every column
            # gets the two campaign-critical regimes, others rotate over all
            # five so inflag 3/4 and iceflag 4 paths are exercised too.
            ncol_seen += 1
            regs = [(1, 1, 1), (0, 0, 0), FLAG_REGIMES[ncol_seen % 5]]
            for reqc, reqi, reqs in dict.fromkeys(regs):
                caseid += 1
                hdr_a = [yr, julday, 8 if reqc else 10, 1, 2, 0, 2,
                         reqc, reqi, reqs, 1, 1, 1, 1, 1, 1, 0]
                hdr_b = [julian, meta["xlat"], meta["xlong"], meta["emiss"],
                         meta["tsk"], meta["psfc"], meta["xland"],
                         meta["xice"], meta["snow"], meta["snowh"],
                         meta["p_top"]]
                cases.append(Case(caseid, hdr_a, hdr_b, cols, ifaces))
        ds.close()

    # Synthetic extremes (same nz and p_top guard: p_top must equal the
    # wrfout p_top for the single-init constraint).
    ptop_ref = cases[0].hdr_b[-1] if cases else 10000.0
    synth_kinds = ["isothermal-cold", "hot-dry", "deck-full", "thin-wisp",
                   "top-layer-cloud", "sfc-cloud", "huge-snow",
                   "cold-sfc-inversion", "warm-rain-only", "mixed-phase-deep",
                   "ocean-warm"]
    # Vary surface pressure across synthetic columns so laytrop and the
    # jp/jt interpolation indices move (all wrfout columns share one psfc).
    synth_psfc = [70000.0, 78000.0, 85000.0, 92000.0, 100000.0, 103000.0]
    for si, kind in enumerate(synth_kinds):
        cols, ifaces, meta = synth_column(nz_ref or 49, kind, rng,
                                          psfc=synth_psfc[si % len(synth_psfc)])
        # force synthetic p_top/psfc-interface consistency with fixtures
        scale = None
        if abs(meta["p_top"] - ptop_ref) > 1e-3:
            scale = ptop_ref / meta["p_top"]
        for reqc, reqi, reqs in [(1, 1, 1), (0, 0, 0)]:
            caseid += 1
            hdr_a = [1974, 94, 8 if reqc else 10, 1, 2, 0, 2,
                     reqc, reqi, reqs, 1, 1, 1, 1, 1, 1, 0]
            hdr_b = [94.5, meta["xlat"], meta["xlong"], meta["emiss"],
                     meta["tsk"], meta["psfc"], meta["xland"], meta["xice"],
                     meta["snow"], meta["snowh"], ptop_ref]
            cases.append(Case(caseid, hdr_a, hdr_b, cols, ifaces))

    # Two cases exercising the no-F_QI winter split (MP option 3 style)
    for variant in range(2):
        cols, ifaces, meta = synth_column(nz_ref or 49, "warm-rain-only", rng)
        caseid += 1
        hdr_a = [1974, 94, 3, 1, 2, 0, 2, 0, 0, 0,
                 1, 1, 1, 0, 0, 0, variant]   # f_qi=f_qs=f_qg=0
        hdr_b = [94.5, meta["xlat"], meta["xlong"], meta["emiss"],
                 260.0 if variant == 0 else meta["tsk"], meta["psfc"],
                 meta["xland"], meta["xice"], meta["snow"], meta["snowh"],
                 ptop_ref]
        if variant == 0:
            cols = cols.copy()
            cols[:, 1] -= 25.0   # push t below 273.15 for the swap branch
            ifaces = ifaces.copy()
            ifaces[:, 1] -= 25.0
        cases.append(Case(caseid, hdr_a, hdr_b, cols, ifaces))

    # One icld=1 (random overlap) regime case
    cols, ifaces, meta = synth_column(nz_ref or 49, "mixed-phase-deep", rng)
    caseid += 1
    hdr_a = [1974, 94, 8, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    hdr_b = [94.5, meta["xlat"], meta["xlong"], meta["emiss"], meta["tsk"],
             meta["psfc"], meta["xland"], meta["xice"], meta["snow"],
             meta["snowh"], ptop_ref]
    cases.append(Case(caseid, hdr_a, hdr_b, cols, ifaces))

    with open(out_txt, "w", newline="\n") as fh:
        fh.write(f"{len(cases)} {nz_ref}\n")
        for c in cases:
            fh.write(" ".join(str(v) for v in [c.caseid] + c.hdr_a) + "\n")
            fh.write(" ".join(f9(v) for v in c.hdr_b) + "\n")
            for k in range(c.cols.shape[0]):
                fh.write(" ".join(f9(v) for v in c.cols[k]) + "\n")
            for k in range(c.ifaces.shape[0]):
                fh.write(" ".join(f9(v) for v in c.ifaces[k]) + "\n")
    print(f"{len(cases)} cases written to {out_txt} (nz={nz_ref})")


if __name__ == "__main__":
    main()
