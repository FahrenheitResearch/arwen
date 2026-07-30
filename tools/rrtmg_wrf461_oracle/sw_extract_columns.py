"""Extract diverse real atmospheric columns from CPU-reference wrfout files
into the sw_fixture_driver input format.

Field reconstruction follows WRF's own formulas (read-only from wrfout):
  p3d  = P + PB                                  [Pa]
  th   = T + 300 (dry potential temperature)     [K]
  pi3d = (p3d/1e5)**(Rd/cp), t3d = th*pi3d
  z_w  = (PH + PHB)/g, dz8w(k) = z_w(k+1) - z_w(k)
  p8w  = FNM/FNP staggering exactly as dyn_em phy_prep, with phy_prep's
         surface (linear-in-z) and top (log-p) boundary forms
  t8w  = same staggering, linear boundary forms
  rho3d ~ p3d/(Rd*t3d*(1+0.608qv))  (unused by the EM_CORE SW path; realism only)
  o31d = WRF o3input=2 pipeline: oznini latitude interpolation of
         ozone.formatted + ozn_time_int month interpolation + ozn_p_int
         pressure interpolation (transcribed from module_ra_cam_support /
         module_radiation_driver of the same source bundle)
  coszen = wrfout COSZEN (the actual WRF value at output time)
  declin/solcon = radconst(julian) transcribed from module_radiation_driver

RE_CLOUD/RE_ICE/RE_SNOW are not archived in the CPU reference wrfouts, so
for has_req* = 1 cases they are synthesized as deterministic, physically
plausible functions of the hydrometeor fields (documented below), with some
columns pinned at the WRF driver's boundary-handling values (re_c <= 2.5um
land/ocean fills, re_i < 5um retab interpolation, re_s > 130um snow-path
clamp).  Fixture inputs need to be realistic and branch-covering, not
bit-identical to what the 500m CPU run happened to feed its own radiation.

usage: sw_extract_columns.py WRFOUT_DIR WRF_SOURCE_ROOT OUT_TXT
"""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np

F = np.float32
RD = 287.0
CP = 1004.5
RCP = RD / CP
G = 9.81
P1000 = 1.0e5

DATE_OZ = np.array([16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350],
                   dtype=np.float64)


# ----------------------------------------------------------------------
# WRF o3input=2 ozone pipeline (transcribed)
# ----------------------------------------------------------------------

def load_ozone(source_root):
    run = Path(source_root) / "run"
    plev = np.loadtxt(run / "ozone_plev.formatted", dtype=np.float64) * 100.0
    lat = np.loadtxt(run / "ozone_lat.formatted", dtype=np.float64)
    raw = np.loadtxt(run / "ozone.formatted", dtype=np.float64)
    levsiz, latsiz = len(plev), len(lat)
    # read order: m, j(lat), k(lev)
    ozmixin = raw.reshape(12, latsiz, levsiz)  # [month, lat, lev]
    return plev, lat, ozmixin


def lin_interpol2(x, f, y):
    n = len(x)
    if y <= x[0]:
        k = 0
    elif y >= x[n - 1]:
        k = n - 2
    else:
        k = 0
        while y > x[k + 1] and k < n - 1:
            k += 1
    a = (f[k + 1] - f[k]) / (x[k + 1] - x[k])
    return f[k] + a * (y - x[k])


def ozn_time_int(julian, ozmix_lat):
    """ozmix_lat: [12 months, levsiz] at the column latitude."""
    intjulian = julian + 1.0
    ijul = int(intjulian)
    intjulian = intjulian - float(ijul)
    ijul = ijul % 365
    if ijul == 0:
        ijul = 365
    intjulian = intjulian + ijul
    np1 = 1
    finddate = False
    for m in range(1, 13):
        if DATE_OZ[m - 1] > intjulian and not finddate:
            np1 = m
            finddate = True
    cdayozp = DATE_OZ[np1 - 1]
    if np1 > 1:
        cdayozm = DATE_OZ[np1 - 2]
        npp, nm = np1, np1 - 1
    else:
        cdayozm = DATE_OZ[11]
        npp, nm = np1, 12
    if np1 == 1:
        deltat = cdayozp + 365.0 - cdayozm
        if intjulian > cdayozp:
            fact1 = (cdayozp + 365.0 - intjulian) / deltat
            fact2 = (intjulian - cdayozm) / deltat
        else:
            fact1 = (cdayozp - intjulian) / deltat
            fact2 = (intjulian + 365.0 - cdayozm) / deltat
    else:
        deltat = cdayozp - cdayozm
        fact1 = (cdayozp - intjulian) / deltat
        fact2 = (intjulian - cdayozm) / deltat
    return ozmix_lat[nm - 1] * fact1 + ozmix_lat[npp - 1] * fact2


def ozn_p_int(pmid_bottom_up, pin, ozmixt):
    """Transcribed ozn_p_int for a single column.

    pmid_bottom_up: model layer pressures [Pa], bottom-up.
    pin: ozone data pressures [Pa], top-down.  ozmixt: [levsiz].
    Returns o3vmr bottom-up.
    """
    pver = len(pmid_bottom_up)
    levsiz = len(pin)
    pmid = pmid_bottom_up[::-1]  # top-down, mirrors the kk = kte-k+kts copy
    o3 = np.zeros(pver)
    kupper = 1  # Fortran index
    for k in range(1, pver + 1):
        kout = pver - k + 1
        found = False
        for kk in range(kupper, levsiz):
            if pin[kk - 1] < pmid[k - 1] <= pin[kk]:
                kupper = kk
                dpu = pmid[k - 1] - pin[kupper - 1]
                dpl = pin[kupper] - pmid[k - 1]
                o3[kout - 1] = (ozmixt[kupper - 1] * dpl +
                                ozmixt[kupper] * dpu) / (dpl + dpu)
                found = True
                break
        if not found:
            if pmid[k - 1] < pin[0]:
                o3[kout - 1] = ozmixt[0] * pmid[k - 1] / pin[0]
            elif pmid[k - 1] > pin[levsiz - 1]:
                o3[kout - 1] = ozmixt[levsiz - 1]
            else:
                dpu = pmid[k - 1] - pin[kupper - 1]
                dpl = pin[kupper] - pmid[k - 1]
                o3[kout - 1] = (ozmixt[kupper - 1] * dpl +
                                ozmixt[kupper] * dpu) / (dpl + dpu)
    return o3  # already bottom-up: kout=pver receives the top layer


def radconst(julian):
    degrad = 3.1415926 / 180.0
    dpd = 360.0 / 365.0
    obecl = 23.5 * degrad
    sinob = np.sin(obecl)
    if julian >= 80.0:
        sxlong = dpd * (julian - 80.0)
    else:
        sxlong = dpd * (julian + 285.0)
    sxlong = sxlong * degrad
    arg = sinob * np.sin(sxlong)
    declin = np.arcsin(arg)
    djul = julian * 360.0 / 365.0
    rjul = djul * degrad
    eccfac = (1.000110 + 0.034221 * np.cos(rjul) + 0.001280 * np.sin(rjul)
              + 0.000719 * np.cos(2 * rjul) + 0.000077 * np.sin(2 * rjul))
    solcon = 1370.0 * eccfac
    return declin, solcon


# ----------------------------------------------------------------------
# column reconstruction
# ----------------------------------------------------------------------

def column(ds, i, j, plev_oz, lat_oz, ozmixin, julian):
    """Reconstruct WRF driver inputs for column (i, j) (0-based numpy)."""
    def v3(name):
        return np.asarray(ds.variables[name][0, :, j, i], dtype=np.float64)

    def v2(name):
        return float(ds.variables[name][0, j, i])

    p = v3("P") + v3("PB")
    th = v3("T") + 300.0
    pi = (p / P1000) ** RCP
    t = th * pi
    zw = (v3("PH") + v3("PHB")) / G
    dz = zw[1:] - zw[:-1]
    zm = 0.5 * (zw[1:] + zw[:-1])
    nz = len(p)

    fnm = np.asarray(ds.variables["FNM"][0], dtype=np.float64)
    fnp = np.asarray(ds.variables["FNP"][0], dtype=np.float64)
    p8w = np.zeros(nz + 1)
    t8w = np.zeros(nz + 1)
    for k in range(1, nz):  # interior w-levels (Fortran k=2..kde-1)
        p8w[k] = fnm[k] * p[k] + fnp[k] * p[k - 1]
        t8w[k] = fnm[k] * t[k] + fnp[k] * t[k - 1]
    # surface (phy_prep linear-in-z)
    z0, z1, z2 = zw[0], zm[0], zm[1]
    w1 = (z0 - z2) / (z1 - z2)
    w2 = 1.0 - w1
    p8w[0] = w1 * p[0] + w2 * p[1]
    t8w[0] = w1 * t[0] + w2 * t[1]
    # top (log-p / linear-t)
    z0, z1, z2 = zw[nz], zm[nz - 1], zm[nz - 2]
    w1 = (z0 - z2) / (z1 - z2)
    w2 = 1.0 - w1
    p8w[nz] = np.exp(w1 * np.log(p[nz - 1]) + w2 * np.log(p[nz - 2]))
    t8w[nz] = w1 * t[nz - 1] + w2 * t[nz - 2]

    qv = np.maximum(v3("QVAPOR"), 0.0)
    rho = p / (RD * t * (1.0 + 0.608 * qv))

    xlat = v2("XLAT")
    ozmix_lat = np.array([
        [lin_interpol2(lat_oz, ozmixin[m, :, k], xlat)
         for k in range(len(plev_oz))]
        for m in range(12)
    ])
    ozmixt = ozn_time_int(julian, ozmix_lat)
    o3 = ozn_p_int(p, plev_oz, ozmixt)

    col = {
        "p3d": p, "t3d": t, "dz8w": dz, "pi3d": pi, "rho3d": rho,
        "qv": qv,
        "qc": np.asarray(ds.variables["QCLOUD"][0, :, j, i], dtype=np.float64),
        "qr": np.asarray(ds.variables["QRAIN"][0, :, j, i], dtype=np.float64),
        "qi": np.asarray(ds.variables["QICE"][0, :, j, i], dtype=np.float64),
        "qs": np.asarray(ds.variables["QSNOW"][0, :, j, i], dtype=np.float64),
        "qg": np.asarray(ds.variables["QGRAUP"][0, :, j, i], dtype=np.float64),
        "cldfra": np.asarray(ds.variables["CLDFRA"][0, :, j, i], dtype=np.float64),
        "o31d": o3,
        "p8w": p8w, "t8w": t8w,
        "coszen": v2("COSZEN"), "albedo": v2("ALBEDO"), "tsk": v2("TSK"),
        "xland": v2("XLAND"), "snow": v2("SNOW"),
        "xlat": xlat, "xlong": v2("XLONG"),
    }
    return col


def synth_re(col, style):
    """Deterministic, plausible effective radii [m] from hydrometeors.

    style "mid":  interior-of-table values;
    style "edge": values pinned at driver boundary handling (re_c below
                  2.5um in cloudy layers, re_i below 5um, re_s above 130um).
    """
    qc, qi, qs = col["qc"], col["qi"], col["qs"]
    if style == "edge":
        re_c = np.where(qc > 1e-8, 1.0e-6, 0.0)
        re_i = np.where(qi > 1e-9, 4.0e-6, 0.0)
        re_s = np.where(qs > 1e-9, 250.0e-6, 0.0)
    else:
        re_c = np.where(qc > 1e-8,
                        np.clip(4.0 + 6.0 * np.log10(1.0 + qc * 1e6), 2.6, 45.0),
                        0.0) * 1e-6
        re_i = np.where(qi > 1e-9,
                        np.clip(20.0 + 25.0 * np.log10(1.0 + qi * 1e7), 5.1, 125.0),
                        0.0) * 1e-6
        re_s = np.where(qs > 1e-9,
                        np.clip(40.0 + 40.0 * np.log10(1.0 + qs * 1e6), 10.5, 128.0),
                        0.0) * 1e-6
    col["re_cloud"], col["re_ice"], col["re_snow"] = re_c, re_i, re_s


def fmt(v):
    return np.format_float_scientific(F(v), precision=9, unique=False,
                                      exp_digits=2)


def emit_case(out, caseid, col, meta):
    yr, julday, mp, has_c, has_i, has_s, itap, julian, gmt, xtime = meta
    declin, solcon = radconst(julian)
    out.append(f"{caseid} {yr} {julday} {mp} 1 2 0 2 {has_c} {has_i} {has_s} 2 {itap}")
    out.append(" ".join(fmt(x) for x in [
        julian, gmt, xtime, 1.0, declin, solcon, col["coszen"],
        col["xlat"], col["xlong"], col["albedo"], col["tsk"], col["xland"],
        col.get("xice", 0.0), col["snow"], 0.0]))
    nz = len(col["p3d"])
    for k in range(nz):
        out.append(" ".join(fmt(a[k]) for a in (
            col["p3d"], col["t3d"], col["dz8w"], col["pi3d"], col["rho3d"],
            col["qv"], col["qc"], col["qr"], col["qi"], col["qs"], col["qg"],
            col["cldfra"], col["re_cloud"], col["re_ice"], col["re_snow"],
            col["o31d"])))
    for k in range(nz + 1):
        out.append(f"{fmt(col['p8w'][k])} {fmt(col['t8w'][k])}")


def pick_columns(ds):
    """Return dict of (i, j) picks by category for one file."""
    qtot = np.zeros(ds.variables["QCLOUD"].shape[2:])
    for q in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP"):
        qtot += np.asarray(ds.variables[q][0]).sum(axis=0)
    cf = np.asarray(ds.variables["CLDFRA"][0])
    cfcol = cf.sum(axis=0)
    picks = {}
    picks["convective"] = np.unravel_index(np.argmax(qtot), qtot.shape)
    clear = np.where(cfcol == 0.0)
    if len(clear[0]):
        picks["clear"] = (clear[0][len(clear[0]) // 2], clear[1][len(clear[1]) // 2])
    partial = np.where((cfcol > 0.5) & (cfcol < 3.0) & (qtot < qtot.max() * 0.01))
    if len(partial[0]):
        picks["partial"] = (partial[0][0], partial[1][0])
    return {k: (int(i), int(j)) for k, (j, i) in picks.items()}


def main(wrfout_dir, source_root, out_txt):
    wrfout_dir = Path(wrfout_dir)
    plev_oz, lat_oz, ozmixin = load_ozone(source_root)
    yr, gmt = 1974, 12.0

    def jd(day, hour):
        # julian = 0-based day-of-year + fraction; Apr 3 1974 = day index 92
        return (92 + (day - 3)) + hour / 24.0, 93 + (day - 3)

    cases = []
    caseid = 0

    def add(fname, ij, mp, flags, itap, style="mid"):
        nonlocal caseid
        day = 4 if "04-04" in fname else 3
        hour = int(fname.split("_")[-3])
        julian, julday = jd(day, hour)
        xtime = ((day - 3) * 24 + hour - 12) * 60.0
        ds = netCDF4.Dataset(wrfout_dir / fname)
        try:
            col = column(ds, ij[0], ij[1], plev_oz, lat_oz, ozmixin, julian)
            synth_re(col, style)
            caseid += 1
            emit_case(cases, caseid, col,
                      (yr, julday, mp, *flags, itap, julian, gmt, xtime))
            print(f"case {caseid}: {fname} ij={ij} mp={mp} flags={flags} "
                  f"itap={itap} style={style} coszen={col['coszen']:.4f} "
                  f"cldsum={col['cldfra'].sum():.2f}")
        finally:
            ds.close()

    # --- category scan on two key files ---
    with netCDF4.Dataset(wrfout_dir / "wrfout_d03_1974-04-03_18_00_00") as ds:
        p18 = pick_columns(ds)
    with netCDF4.Dataset(wrfout_dir / "wrfout_d01_1974-04-04_00_00_00") as ds:
        cz = np.asarray(ds.variables["COSZEN"][0])
        cf = np.asarray(ds.variables["CLDFRA"][0]).sum(axis=0)
        night = np.where(cz <= 0.0)
        lowsun = np.where((cz > 0.0) & (cz < 0.02))
        dusk_cloudy = np.where((cz > 0.02) & (cz < 0.2) & (cf > 5.0))
        n_ij = (int(night[1][0]), int(night[0][0]))
        l_ij = (int(lowsun[1][len(lowsun[0]) // 2]), int(lowsun[0][len(lowsun[0]) // 2]))
        d_ij = (int(dusk_cloudy[1][0]), int(dusk_cloudy[0][0])) if len(dusk_cloudy[0]) else l_ij
        snowy = np.where(np.asarray(ds.variables["SNOW"][0]) > 10.0)
        s_ij = (int(snowy[1][0]), int(snowy[0][0])) if len(snowy[0]) else l_ij
        water = np.where(np.asarray(ds.variables["XLAND"][0]) > 1.5)
        w_ij = (int(water[1][len(water[0]) // 2]), int(water[0][len(water[0]) // 2]))

    T = (1, 1, 1)   # Thompson-style flags
    W = (0, 0, 0)   # WSM6-style (relcalc/reicalc + Fu path)
    C = (1, 0, 0)   # cloud-re only (inflg=3, iceflg=3)
    P3 = (1, 1, 0)  # P3 special case

    # d03 18Z convection, high sun
    add("wrfout_d03_1974-04-03_18_00_00", p18["convective"], 8, T, 1)
    add("wrfout_d03_1974-04-03_18_00_00", p18["convective"], 6, W, 1)
    add("wrfout_d03_1974-04-03_18_00_00", p18["convective"], 8, T, 0, style="edge")
    add("wrfout_d03_1974-04-03_18_00_00", p18["convective"], 28, P3, 0)
    if "clear" in p18:
        add("wrfout_d03_1974-04-03_18_00_00", p18["clear"], 8, T, 1)
        add("wrfout_d03_1974-04-03_18_00_00", p18["clear"], 6, W, 0)
    if "partial" in p18:
        add("wrfout_d03_1974-04-03_18_00_00", p18["partial"], 8, T, 1)
        add("wrfout_d03_1974-04-03_18_00_00", p18["partial"], 8, C, 0)

    # d02 20Z mixed
    with netCDF4.Dataset(wrfout_dir / "wrfout_d02_1974-04-03_20_00_00") as ds:
        p20 = pick_columns(ds)
    add("wrfout_d02_1974-04-03_20_00_00", p20["convective"], 8, T, 1)
    if "partial" in p20:
        add("wrfout_d02_1974-04-03_20_00_00", p20["partial"], 6, W, 1, style="mid")

    # d01 00Z: night / dusk / low sun
    add("wrfout_d01_1974-04-04_00_00_00", n_ij, 8, T, 0)          # night
    add("wrfout_d01_1974-04-04_00_00_00", l_ij, 8, T, 1)          # coszen < 0.02
    add("wrfout_d01_1974-04-04_00_00_00", d_ij, 8, T, 1)          # dusk cloudy
    add("wrfout_d01_1974-04-04_00_00_00", s_ij, 8, T, 0)          # snow cover
    add("wrfout_d01_1974-04-04_00_00_00", s_ij, 6, W, 0)
    add("wrfout_d01_1974-04-04_00_00_00", w_ij, 8, T, 0)          # water

    # d01 13Z: very low morning sun
    with netCDF4.Dataset(wrfout_dir / "wrfout_d01_1974-04-03_13_00_00") as ds:
        cz13 = np.asarray(ds.variables["COSZEN"][0])
        cf13 = np.asarray(ds.variables["CLDFRA"][0]).sum(axis=0)
        vlow = np.where((cz13 > 0.0) & (cz13 < 0.06))
        m_ij = (int(vlow[1][0]), int(vlow[0][0]))
        cl13 = np.where((cz13 > 0.3) & (cf13 > 5.0))
        c_ij = (int(cl13[1][0]), int(cl13[0][0])) if len(cl13[0]) else m_ij
    add("wrfout_d01_1974-04-03_13_00_00", m_ij, 8, T, 1)
    add("wrfout_d01_1974-04-03_13_00_00", c_ij, 8, T, 0)
    add("wrfout_d01_1974-04-03_13_00_00", c_ij, 6, W, 1)

    # d01 18Z high sun continental
    with netCDF4.Dataset(wrfout_dir / "wrfout_d01_1974-04-03_18_00_00") as ds:
        p118 = pick_columns(ds)
    add("wrfout_d01_1974-04-03_18_00_00", p118["convective"], 8, T, 0)
    if "clear" in p118:
        add("wrfout_d01_1974-04-03_18_00_00", p118["clear"], 8, T, 1)

    nz = 49
    ncase = caseid
    header = f"{ncase} {nz}"
    Path(out_txt).write_text(header + "\n" + "\n".join(cases) + "\n")
    print(f"wrote {out_txt}: {ncase} cases, nz={nz}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
