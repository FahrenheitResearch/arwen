"""Generate the MYJ PBL oracle's input columns (run_myjpbl.F90's case files).

    python tools/myjpbl_wrf461_oracle/make_columns.py OUT_DIR

Writes one ``case_<name>.bin`` per case in the stream layout run_myjpbl.F90
reads (see its header), plus ``cases.txt`` listing the case names.  The
columns are deterministic (seeded) and physically consistent: hydrostatic
pressure from the virtual temperature, EXNER = (PMID/1e5)**(R_d/c_p),
T = TH*EXNER, specific-humidity surface fields, TKE >= epsq2/2.

Regimes (main cases, 80 columns each, one call with QCI and one without):
daytime convective land, stable night land (with and without a low-level
jet), near-neutral windy, warm and cold ocean, snow cover, sea ice, high
terrain (HT up to 4500 m, PSFC down to ~57 kPa), below-sea-level terrain,
stratocumulus (liquid), mixed-phase and ice cloud, deep moist tropical, and
edge columns: CT /= 0 with a DTH sign change (MIXLEN's CT fix), CHKLOWQ = 0
(QFC1 = 0, QSFC kept), dew (ELFLX < 0), USTAR = 0 and USTAR = 2.5 m/s,
calm isothermal (GM at its EPSGM floor), exactly neutral and dry (GH set to
EPSGH), TKE at its floor everywhere, TKE large everywhere (LPBL = 1),
completely dry column, subnormal and signed-zero condensate/winds, and a
2 m lowest layer.  Extra cases: nz = 4 (the kernel's floor), nz = 128 (its
ceiling), and a coarse nz = 35 grid at DT = 90 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F = np.float32
RD, RV, G = 287.0, 461.6, 9.81
CP = 7.0 * RD / 2.0
RCP = RD / CP
EP1 = RV / RD - 1.0


def grid(nz, dz0, ratio, dzmax, ztop=None):
    dz = np.minimum(dz0 * ratio ** np.arange(nz), dzmax)
    if ztop is not None:
        dz *= ztop / dz.sum()
    return dz


def qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / np.maximum(p - 0.378 * es, 1.0)


def column(rng, nz, dz, *, kind, ht=0.0, xland=1.0, th0=295.0, h=1000.0,
           q0=0.010, ws=6.0, wd=0.8, cloud=None, tke_bl=0.8, snow=0.0,
           sice=0.0, ct=0.0, chklowq=None, elflx=None, ustar=None,
           dskin=None, akhs=None, akms=None, extra=None):
    """One column as a dict of float64 arrays (bottom-up, nz levels)."""
    z_int = np.concatenate(([0.0], np.cumsum(dz)))
    zc = 0.5 * (z_int[:-1] + z_int[1:])
    if kind == "convective":
        th = th0 + np.where(zc < h, -1.5 * np.exp(-zc / 60.0),
                            3.0 + 0.004 * (zc - h))
        th += np.where((zc > h) & (zc < h + 150), 3.0 * (zc - h) / 150, 0)
    elif kind == "stable":
        th = th0 + 6.0 * (1.0 - np.exp(-zc / 150.0)) + 0.003 * zc
    elif kind == "neutral":
        th = th0 + np.where(zc < h, 0.0, 0.004 * (zc - h))
    elif kind == "isothermal":
        th = np.full(nz, th0)            # replaced by T = const below
    elif kind == "flat":
        th = np.full(nz, th0)
    else:
        raise ValueError(kind)
    th = th + 0.15 * np.sin(0.7 * np.arange(nz) + rng.uniform(0, 6))
    th = th + np.where(zc > 11000.0, 0.012 * (zc - 11000.0), 0.0)
    if kind == "flat":
        th = np.full(nz, th0)
    qv = q0 * np.exp(-np.maximum(zc - (h if kind != "stable" else 0), 0)
                     / 2200.0)
    qv *= 1.0 + 0.04 * np.cos(0.5 * np.arange(nz))
    # Winds: log-ish to the BL top, geostrophic above, veering with height.
    shape = np.log1p(zc / 1.0) / np.log1p(max(h, 200.0) / 1.0)
    speed = ws * np.minimum(shape, 1.0) * (1.0 + 0.00005 * zc)
    if extra and extra.get("jet"):
        speed += 9.0 * np.exp(-((zc - 250.0) / 120.0) ** 2)
    ang = wd + 0.0004 * zc
    u = speed * np.cos(ang)
    v = speed * np.sin(ang)
    qc = np.zeros(nz)
    qi = np.zeros(nz)
    if cloud:
        for (zb, zt, liq, ice) in cloud:
            m = (zc >= zb) & (zc <= zt)
            qc[m] = liq
            qi[m] = ice
    # Hydrostatic pressure from the surface up.
    psfc = 101325.0 * np.exp(-ht / 8200.0) * (1.0 + rng.uniform(-0.01, 0.01))
    pint = np.empty(nz + 1)
    pint[0] = psfc
    t = np.empty(nz)
    exner = np.empty(nz)
    pmid = np.empty(nz)
    for k in range(nz):
        # iterate for consistency of T with pmid
        p_guess = pint[k] * np.exp(-G * dz[k] / (2 * RD * 280.0))
        for _ in range(3):
            ex = (p_guess / 1e5) ** RCP
            tk = (th[k] * ex if kind != "isothermal"
                  else th0 * (psfc / 1e5) ** RCP)
            tv = tk * (1.0 + EP1 * qv[k])
            pint[k + 1] = pint[k] * np.exp(-G * dz[k] / (RD * tv))
            p_guess = 0.5 * (pint[k] + pint[k + 1])
        pmid[k] = p_guess
        exner[k] = (pmid[k] / 1e5) ** RCP
        t[k] = tk
        if kind == "isothermal":
            th[k] = t[k] / exner[k]
    qv = np.minimum(qv, 0.97 * qsat(t, pmid))
    rho = pmid / (RD * t * (1.0 + EP1 * qv))
    # TKE: BL profile above the floor.
    tke = np.full(nz, 0.1)
    if tke_bl > 0:
        prof = np.clip(1.0 - zc / max(h, 50.0), 0.0, 1.0) ** 1.5
        tke = np.maximum(0.1, tke_bl * (0.3 + prof) * (zc < 1.3 * h))
    # Surface.
    t1 = t[0]
    if dskin is None:
        dskin = {"convective": 6.0, "stable": -4.0, "neutral": 0.5,
                 "isothermal": 0.0, "flat": 0.0}[kind]
    tsk = t1 + dskin
    thz0 = tsk * (1e5 / psfc) ** RCP
    qs = qsat(tsk, psfc)
    if xland == 2.0:
        qz0 = qs * 0.98
        uz0 = 0.03 * u[0]
        vz0 = 0.03 * v[0]
        chk = 1.0 if chklowq is None else chklowq
    else:
        qz0 = min(qs, qv[0] / (1.0 + qv[0]) * 1.05)
        uz0 = 0.0
        vz0 = 0.0
        chk = rng.uniform(0.25, 1.0) if chklowq is None else chklowq
    if elflx is None:
        elflx = {"convective": rng.uniform(150, 420), "stable":
                 rng.uniform(-8, 15), "neutral": rng.uniform(20, 80),
                 "isothermal": 5.0, "flat": 0.0}[kind]
    if ustar is None:
        ustar = max(0.03, 0.06 * ws * rng.uniform(0.7, 1.3))
    if akhs is None:
        akhs = rng.uniform(0.004, 0.04) if kind != "stable" else rng.uniform(0.0005, 0.006)
    if akms is None:
        akms = akhs * rng.uniform(1.0, 1.6)
    return dict(dz=dz, pmid=pmid, pint=pint, th=th, t=t, exner=exner, qv=qv,
                qcw=qc, qci=qi, u=u, v=v, rho=rho, tke=tke,
                exch_h=np.zeros(nz), ht=ht, tsk=tsk, qsfc=qz0, chklowq=chk,
                thz0=thz0, qz0=qz0, uz0=uz0, vz0=vz0, xland=xland,
                sice=sice, snow=snow, ustar=ustar,
                znt=(2e-4 if xland == 2.0 else rng.uniform(0.01, 0.9)),
                ct=ct, akhs=akhs, akms=akms, elflx=elflx)


def main_columns(rng, nz, dz):
    cols = []
    terr = [0.0, 120.0, 250.0, 1500.0, 2600.0, 3500.0, 4500.0, -25.0]
    # Daytime convective land, 10 columns over the terrain set.
    for i in range(10):
        cols.append(column(rng, nz, dz, kind="convective",
                           ht=terr[i % len(terr)], th0=290 + 2 * i,
                           h=600 + 250 * i, q0=0.004 + 0.0012 * i,
                           ws=2 + i, wd=0.3 * i, tke_bl=0.6 + 0.25 * i,
                           dskin=2 + 1.2 * i))
    # Stable night land, 10 columns (half with a low-level jet).
    for i in range(10):
        cols.append(column(rng, nz, dz, kind="stable",
                           ht=terr[(i + 3) % len(terr)], th0=275 + 2 * i,
                           h=150 + 40 * i, q0=0.003 + 0.0008 * i,
                           ws=1 + 0.8 * i, wd=2.0 + 0.2 * i,
                           tke_bl=0.12 + 0.05 * i, dskin=-1.5 - 0.7 * i,
                           extra={"jet": i % 2 == 0}))
    # Near-neutral windy, 6.
    for i in range(6):
        cols.append(column(rng, nz, dz, kind="neutral",
                           ht=terr[(i + 1) % len(terr)], th0=285 + i,
                           h=900 + 200 * i, q0=0.006, ws=10 + 2.5 * i,
                           wd=4.0 + 0.3 * i, tke_bl=1.5 + 0.3 * i))
    # Ocean: warm (unstable) and cold (stable), 5 each.
    for i in range(5):
        cols.append(column(rng, nz, dz, kind="convective", xland=2.0,
                           th0=296 + i, h=500 + 150 * i, q0=0.014 + 0.001 * i,
                           ws=5 + 2 * i, wd=1.0 + 0.5 * i, tke_bl=0.5 + 0.2 * i,
                           dskin=1.0 + 0.6 * i))
        cols.append(column(rng, nz, dz, kind="stable", xland=2.0,
                           th0=280 + i, h=200, q0=0.006, ws=7 + i,
                           wd=5.0, tke_bl=0.2, dskin=-1.0 - 0.8 * i))
    # Snow cover (RLIVWV arm), 4; sea ice, 3.
    for i in range(4):
        cols.append(column(rng, nz, dz, kind="stable" if i % 2 else "neutral",
                           ht=terr[(i + 2) % len(terr)], th0=262 + 2 * i,
                           h=300 + 100 * i, q0=0.0015, ws=3 + 2 * i,
                           snow=0.05 + 0.1 * i, tke_bl=0.3, dskin=-2.0 + i))
    for i in range(3):
        cols.append(column(rng, nz, dz, kind="stable", th0=250 + 3 * i,
                           h=200, q0=0.0008, ws=6 + i, sice=1.0,
                           snow=0.02 * i, tke_bl=0.25, dskin=-3.0))
    # Clouds: stratocumulus (liquid) 4, mixed-phase 3, ice-only cirrus 2,
    # deep moist tropical 3.
    for i in range(4):
        cols.append(column(rng, nz, dz, kind="convective",
                           xland=2.0 if i < 2 else 1.0, th0=288 + i,
                           h=800 + 100 * i, q0=0.009, ws=6 + i,
                           cloud=[(600 + 100 * i, 900 + 100 * i, 3e-4 + 1e-4 * i, 0.0)],
                           tke_bl=0.7, dskin=0.5 + i))
    for i in range(3):
        cols.append(column(rng, nz, dz, kind="neutral",
                           ht=terr[(i + 4) % len(terr)], th0=272 + 2 * i,
                           h=1200, q0=0.004, ws=8 + 2 * i,
                           cloud=[(400, 1500, 1.5e-4, 8e-5 * (i + 1)),
                                  (3000, 5000, 0.0, 4e-5)], tke_bl=1.0))
    for i in range(2):
        cols.append(column(rng, nz, dz, kind="stable", th0=280 + 4 * i,
                           h=300, q0=0.004, ws=15 + 5 * i,
                           cloud=[(7000, 10000, 0.0, 6e-5 + 4e-5 * i)],
                           tke_bl=0.2))
    for i in range(3):
        cols.append(column(rng, nz, dz, kind="convective", xland=2.0 if i else 1.0,
                           th0=300 + i, h=1500, q0=0.019, ws=4 + i,
                           cloud=[(1500, 9000, 6e-4, 0.0), (5500, 12000, 0.0, 2e-4)],
                           tke_bl=2.0, dskin=1.5))
    # ---- edge columns ----------------------------------------------------
    # CT /= 0 where DTH changes sign (MIXLEN :845-850; VDIFH RKCT).
    for ctv in (0.4, -0.25, 1.5):
        cols.append(column(rng, nz, dz, kind="convective", th0=293, h=900,
                           q0=0.008, ws=5, ct=ctv, tke_bl=1.2, dskin=4))
    # CHKLOWQ = 0 -> QFC1 = 0, QSFC kept (:554).
    cols.append(column(rng, nz, dz, kind="convective", th0=300, h=1200,
                       q0=0.006, ws=4, chklowq=0.0, dskin=8))
    cols.append(column(rng, nz, dz, kind="stable", th0=280, h=200, q0=0.005,
                       ws=2, chklowq=0.0, ht=1800.0))
    # Dew: ELFLX < 0, and ELFLX = 0 exactly.
    cols.append(column(rng, nz, dz, kind="stable", th0=283, h=150, q0=0.008,
                       ws=1.5, elflx=-35.0))
    cols.append(column(rng, nz, dz, kind="stable", th0=283, h=150, q0=0.008,
                       ws=1.5, elflx=0.0))
    # USTAR = 0 (Q2 boundary at EPSQ2) and USTAR = 2.5 m/s.
    cols.append(column(rng, nz, dz, kind="stable", th0=278, h=100, q0=0.004,
                       ws=0.5, ustar=0.0, tke_bl=0.0))
    cols.append(column(rng, nz, dz, kind="neutral", th0=284, h=1500, q0=0.005,
                       ws=35.0, ustar=2.5, tke_bl=4.0))
    # Calm isothermal column: zero shear everywhere (GM -> EPSGM).
    c = column(rng, nz, dz, kind="isothermal", th0=285, h=500, q0=0.0,
               ws=0.0, tke_bl=0.0, elflx=0.0)
    c["u"][:] = 0.0
    c["v"][:] = 0.0
    c["qv"][:] = 0.0
    cols.append(c)
    # Exactly neutral and dry: DTH = 0 -> GHL = 0 -> EPSGH (:869).
    c = column(rng, nz, dz, kind="flat", th0=290, h=3000, q0=0.0, ws=8.0,
               tke_bl=0.5)
    c["qv"][:] = 0.0
    cols.append(c)
    # TKE at its floor everywhere, and TKE large everywhere (LPBL = 1).
    c = column(rng, nz, dz, kind="stable", th0=276, h=100, q0=0.003, ws=3,
               tke_bl=0.0)
    cols.append(c)
    c = column(rng, nz, dz, kind="convective", th0=300, h=3000, q0=0.012,
               ws=12, tke_bl=6.0)
    c["tke"][:] = 6.0
    cols.append(c)
    # Completely dry column with clouds of nothing.
    c = column(rng, nz, dz, kind="convective", th0=310, h=2500, q0=0.0, ws=7,
               tke_bl=2.0, elflx=0.0)
    c["qv"][:] = 0.0
    cols.append(c)
    # Subnormal and signed-zero condensate and winds.
    c = column(rng, nz, dz, kind="neutral", th0=287, h=1000, q0=0.005, ws=5)
    c["qcw"][3:9] = 1.0e-40
    c["qci"][5:12] = 3.0e-42
    c["qcw"][12] = -0.0
    c["u"][20:24] = -0.0
    c["v"][20:22] = 0.0
    cols.append(c)
    # Two sign-flip winds (direction reversal through the column).
    c = column(rng, nz, dz, kind="stable", th0=281, h=300, q0=0.004, ws=6,
               tke_bl=0.3)
    c["u"] = c["u"] * np.cos(np.linspace(0, 3 * np.pi, nz))
    cols.append(c)
    return cols


def pack(cols, nz):
    keys3 = ("dz", "pmid", "pint", "th", "t", "exner", "qv", "qcw", "qci",
             "u", "v", "rho", "tke", "exch_h")
    keys2 = ("ht", "tsk", "qsfc", "chklowq", "thz0", "qz0", "uz0", "vz0",
             "xland", "sice", "snow", "ustar", "znt", "ct", "akhs", "akms",
             "elflx")
    ncol = len(cols)
    arr3 = {}
    for k in keys3:
        a = np.zeros((nz + 1, ncol), dtype=F)       # C order (k, i) == Fortran (i, k)
        for i, c in enumerate(cols):
            x = np.asarray(c[k], dtype=np.float64)
            a[: x.size, i] = x.astype(F)
        arr3[k] = a
    arr2 = {k: np.array([c[k] for c in cols], dtype=np.float64).astype(F)
            for k in keys2}
    # Signed zeros set above survive the float64 -> float32 cast.
    for k in ("qcw", "qci", "u", "v"):
        for i, c in enumerate(cols):
            src = np.asarray(c[k])
            neg0 = (src == 0) & np.signbit(src)
            arr3[k][: src.size, i][neg0] = F(-0.0)
    return keys3, keys2, arr3, arr2


def write_case(path, cols, nz, nstep, flqi, dt):
    keys3, keys2, arr3, arr2 = pack(cols, nz)
    with open(path, "wb") as fh:
        np.array([len(cols), nz, nstep, flqi, 461], dtype="<i4").tofile(fh)
        np.array([dt], dtype="<f4").tofile(fh)
        for k in keys3:
            arr3[k].astype("<f4").tofile(fh)
        for k in keys2:
            arr2[k].astype("<f4").tofile(fh)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    rng = np.random.default_rng(461)
    nz = 50
    dz = grid(nz, 24.0, 1.085, 750.0, ztop=20000.0)
    cols = main_columns(rng, nz, dz)
    # one column with a 2 m lowest layer on the same nz
    rng2 = np.random.default_rng(4612)
    dz2 = dz.copy()
    dz2[0] = 2.0
    cols.append(column(rng2, nz, dz2, kind="convective", th0=297, h=1100,
                       q0=0.01, ws=5, dskin=5))
    write_case(out / "case_main_ice.bin", cols, nz, 3, 1, 30.0)
    write_case(out / "case_main_noice.bin", cols, nz, 3, 0, 30.0)
    names += ["main_ice", "main_noice"]
    # nz = 4, the kernel's floor (VDIFQ's interior rows LMH-2 = 2).
    rng = np.random.default_rng(4)
    dz4 = np.array([40.0, 160.0, 600.0, 2000.0])
    c4 = [column(rng, 4, dz4, kind=k, th0=th, h=h, q0=q, ws=w, ht=hh,
                 xland=xl, tke_bl=tk)
          for k, th, h, q, w, hh, xl, tk in (
              ("convective", 295, 800, 0.01, 5, 0.0, 1.0, 1.0),
              ("stable", 280, 150, 0.004, 2, 1200.0, 1.0, 0.2),
              ("neutral", 288, 1500, 0.006, 15, 0.0, 2.0, 2.0),
              ("convective", 300, 1500, 0.015, 3, 0.0, 2.0, 1.5),
              ("stable", 265, 200, 0.002, 4, 3000.0, 1.0, 0.0),
              ("neutral", 290, 900, 0.008, 9, 400.0, 1.0, 0.6),
              ("convective", 305, 2500, 0.012, 1, 2200.0, 1.0, 3.0),
              ("stable", 285, 120, 0.009, 0.5, 0.0, 1.0, 0.1))]
    c4[1]["snow"] = 0.1
    c4[3]["qcw"][1] = 2e-4
    c4[5]["qci"][2] = 5e-5
    write_case(out / "case_nz4.bin", c4, 4, 3, 1, 60.0)
    names.append("nz4")
    # nz = 128, the kernel's ceiling.
    rng = np.random.default_rng(128)
    dz128 = grid(128, 10.0, 1.03, 300.0, ztop=22000.0)
    c128 = [column(rng, 128, dz128, kind=k, th0=th, h=h, q0=q, ws=w, ht=hh,
                   xland=xl, tke_bl=tk, cloud=cl)
            for k, th, h, q, w, hh, xl, tk, cl in (
                ("convective", 296, 1400, 0.012, 6, 0.0, 1.0, 1.5, None),
                ("stable", 278, 120, 0.004, 3, 2100.0, 1.0, 0.15, None),
                ("neutral", 286, 1100, 0.007, 14, 0.0, 2.0, 1.8,
                 [(800, 1100, 2e-4, 0.0)]),
                ("convective", 301, 900, 0.017, 4, 0.0, 2.0, 0.9,
                 [(900, 4000, 4e-4, 1e-4)]),
                ("stable", 258, 200, 0.001, 5, 3700.0, 1.0, 0.0, None),
                ("convective", 290, 2200, 0.006, 9, 1300.0, 1.0, 2.5,
                 [(2500, 3000, 0.0, 6e-5)]),
                ("neutral", 279, 600, 0.004, 20, 0.0, 1.0, 3.0, None),
                ("stable", 287, 250, 0.011, 1, -10.0, 1.0, 0.2, None))]
    c128[6]["snow"] = 0.2
    write_case(out / "case_nz128.bin", c128, 128, 3, 1, 10.0)
    names.append("nz128")
    # Coarse 35-level grid at DT = 90 s (a 9-12 km parent's DTBL).
    rng = np.random.default_rng(35)
    dz35 = grid(35, 60.0, 1.12, 1100.0, ztop=20000.0)
    c35 = [column(rng, 35, dz35, kind=("convective", "stable", "neutral")[i % 3],
                  th0=280 + 3 * i, h=500 + 150 * i, q0=0.003 + 0.001 * i,
                  ws=2 + 1.5 * i, ht=(0.0, 900.0, 2400.0, 0.0)[i % 4],
                  xland=2.0 if i % 5 == 4 else 1.0, tke_bl=0.2 + 0.3 * i)
           for i in range(16)]
    write_case(out / "case_coarse35.bin", c35, 35, 3, 1, 90.0)
    names.append("coarse35")
    (out / "cases.txt").write_text("\n".join(names) + "\n")
    print("wrote", ", ".join(f"{n}" for n in names), "->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cases")
