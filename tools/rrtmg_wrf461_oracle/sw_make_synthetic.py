"""Synthetic extreme columns for the SW oracle - branches the real columns
cannot reach.  Same input format as sw_extract_columns.py; the fixture
driver still proves every case against the untouched RRTMG_SWRAD before a
fixture is written, so these rows are oracle-recorded, not fabricated.

Cases:
  S1  overcast warm liquid cloud, tau extreme, high sun          (1,1,1)
  S2  same cloud, coszen = 1e-11: the rrtmg_sw zepzen clamp      (1,1,1)
  S3  ice cloud with has_req = (0,1,0): inflg=4/iceflg=4 leg     (0,1,0)
  S4  table-edge radii: radliq 59.5, radice ~139 (dge), resnow
      200 um -> the >130 snow_mass_factor clamp                  (1,1,1)
  S5  hot dry column: tsk 330 K, qv floor 1e-12, indfor clamp    (1,1,1)
  S6  deep cold high-pressure column: psfc 1080 hPa (jp/laylow
      low-end clamp), tavel down to 165 K (jt clamps)            (0,0,0)
  S7  cldfra = 1 with zero water paths: cldprmc skip leg         (1,1,1)
  S8  snow-only cloud (qc = qi = 0, qs large): snow branch with
      zero liquid/ice                                            (1,1,1)

usage: sw_make_synthetic.py OUT_TXT
"""

import sys

import numpy as np

F = np.float32
NZ = 49
RD, G0, CP = 287.0, 9.81, 1004.5


def fmt(v):
    return np.format_float_scientific(F(v), precision=9, unique=False,
                                      exp_digits=2)


def base_column(psfc=97000.0, t0=288.0, lapse=6.5e-3, t_min=195.0):
    """Hydrostatic-ish column on 49 layers up to ~10 hPa."""
    p8w = np.geomspace(psfc, 1000.0, NZ + 1)
    p3d = np.sqrt(p8w[:-1] * p8w[1:])
    z = -RD * 260.0 / G0 * np.log(p3d / psfc)
    t3d = np.maximum(t0 - lapse * z, t_min)
    zw = -RD * 260.0 / G0 * np.log(p8w / psfc)
    t8w = np.maximum(t0 - lapse * zw, t_min)
    dz = zw[1:] - zw[:-1]
    pi3d = (p3d / 1e5) ** (RD / CP)
    rho = p3d / (RD * t3d)
    rh = np.clip(0.7 - 0.5 * (z / 12000.0), 0.05, 0.95)
    es = 611.2 * np.exp(17.67 * (t3d - 273.15) / (t3d - 29.65))
    qv = np.clip(0.622 * rh * es / (p3d - rh * es), 2e-7, 0.02)
    o3 = np.interp(np.log(p3d), np.log([100000, 20000, 5000, 1000]),
                   [6e-8, 1.5e-7, 2e-6, 6e-6])
    col = dict(p3d=p3d, t3d=t3d, dz8w=dz, pi3d=pi3d, rho3d=rho, qv=qv,
               qc=np.zeros(NZ), qr=np.zeros(NZ), qi=np.zeros(NZ),
               qs=np.zeros(NZ), qg=np.zeros(NZ), cldfra=np.zeros(NZ),
               re_cloud=np.zeros(NZ), re_ice=np.zeros(NZ),
               re_snow=np.zeros(NZ), o31d=o3, p8w=p8w, t8w=t8w,
               coszen=0.9, albedo=0.18, tsk=t0 + 1.5, xland=1.0, xice=0.0,
               snow=0.0, obscur=0.0, xlat=39.0, xlong=-98.0)
    return col


def emit(out, caseid, col, mp, flags, itap):
    julian, julday, gmt, xtime = 92.75, 93, 12.0, 360.0
    declin, solcon = 0.0975, 1368.0
    out.append(f"{caseid} 1974 {julday} {mp} 1 2 0 2 "
               f"{flags[0]} {flags[1]} {flags[2]} 2 {itap}")
    out.append(" ".join(fmt(x) for x in [
        julian, gmt, xtime, 1.0, declin, solcon, col["coszen"], col["xlat"],
        col["xlong"], col["albedo"], col["tsk"], col["xland"], col["xice"],
        col["snow"], col["obscur"]]))
    for k in range(NZ):
        out.append(" ".join(fmt(a[k]) for a in (
            col["p3d"], col["t3d"], col["dz8w"], col["pi3d"], col["rho3d"],
            col["qv"], col["qc"], col["qr"], col["qi"], col["qs"], col["qg"],
            col["cldfra"], col["re_cloud"], col["re_ice"], col["re_snow"],
            col["o31d"])))
    for k in range(NZ + 1):
        out.append(f"{fmt(col['p8w'][k])} {fmt(col['t8w'][k])}")


def main(out_txt):
    out = []
    cid = 100

    # S1: overcast warm liquid, extreme tau
    c = base_column()
    sl = slice(5, 25)
    c["qc"][sl] = 4e-3
    c["cldfra"][sl] = 1.0
    c["re_cloud"][sl] = 9e-6
    cid += 1; emit(out, cid, c, 8, (1, 1, 1), 1)

    # S2: same, coszen just above zero -> rrtmg_sw clamps cossza to 1e-10
    c2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
          for k, v in c.items()}
    c2["coszen"] = 1e-11
    cid += 1; emit(out, cid, c2, 8, (1, 1, 1), 1)

    # S3: ice cloud, has_req=(0,1,0): inflgsw=4 / iceflgsw=4
    c = base_column()
    sl = slice(28, 40)
    c["qi"][sl] = 4e-4
    c["cldfra"][sl] = 1.0
    c["re_ice"][sl] = 6e-5
    cid += 1; emit(out, cid, c, 50, (0, 1, 0), 1)

    # S4: table-edge radii; resnow 200um hits the >130 snow-path clamp
    c = base_column()
    sl = slice(8, 20)
    c["qc"][sl] = 1e-3
    c["qi"][sl] = 2e-4
    c["qs"][sl] = 8e-4
    c["cldfra"][sl] = 1.0
    c["re_cloud"][sl] = 59.5e-6
    c["re_ice"][sl] = 134.5e-6
    c["re_snow"][sl] = 200e-6
    cid += 1; emit(out, cid, c, 8, (1, 1, 1), 1)

    # S5: hot dry column (indfor low clamp; qv floor 1e-12 in driver)
    c = base_column(t0=330.0, lapse=8.0e-3, t_min=210.0)
    c["qv"][:] = 1e-13     # driver floors to 1e-12
    c["tsk"] = 333.0
    cid += 1; emit(out, cid, c, 8, (1, 1, 1), 1)

    # S6: cold deep column, psfc 1080 hPa (jp -> 1 clamp, laylow) WSM6 flags
    c = base_column(psfc=108000.0, t0=250.0, lapse=4.0e-3, t_min=165.0)
    sl = slice(10, 30)
    c["qi"][sl] = 1e-4
    c["qs"][sl] = 2e-4
    c["cldfra"][sl] = 0.8
    c["tsk"] = 248.0
    c["snow"] = 40.0
    c["albedo"] = 0.6
    c["coszen"] = 0.25
    cid += 1; emit(out, cid, c, 6, (0, 0, 0), 1)

    # S7: cloud fraction 1 with zero condensate: cldprmc skip leg
    c = base_column()
    c["cldfra"][12:20] = 1.0
    cid += 1; emit(out, cid, c, 8, (1, 1, 1), 0)

    # S8: snow-only cloud
    c = base_column(t0=270.0)
    sl = slice(6, 18)
    c["qs"][sl] = 1.2e-3
    c["cldfra"][sl] = 1.0
    c["re_snow"][sl] = 80e-6
    c["re_cloud"][sl] = 8e-6     # unused: qc = 0
    c["re_ice"][sl] = 40e-6      # unused: qi = 0
    cid += 1; emit(out, cid, c, 8, (1, 1, 1), 1)

    n = cid - 100
    with open(out_txt, "w", encoding="ascii", newline="\n") as f:
        f.write(f"{n} {NZ}\n" + "\n".join(out) + "\n")
    print(f"wrote {out_txt}: {n} cases")


if __name__ == "__main__":
    main(sys.argv[1])
