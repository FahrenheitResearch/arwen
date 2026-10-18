"""Columns and step sequences for the WRF v4.6.1 swint_opt = 1 oracle.

Usage: python make_columns.py BUILD_DIRECTORY [--clamp-calendar]

Writes BUILD_DIRECTORY/<scenario>.in.bin (run_swint.F90's input layout) and
BUILD_DIRECTORY/columns.npz (every input, keyed ``<scenario>/<name>``, plus
the per-column regime labels), for these scenarios:

conus_day     a 24 h July day over a CONUS-like set of points, HRRR's
              radt = 15 min and dt = 300 s: clear land, high terrain, ocean,
              afternoon convection (cloud switching call to call), overcast
              stratus whose direct beam stays under 1 W m-2 (the exponent-
              zero fit), dusk and dawn, the stable night (every column).
polar         21 June, 24 h: Arctic polar day up to the pole itself (where
              the zenith cosine barely moves, so the fit ratio is exactly
              one), snow and sea ice with the live albedo changing between
              radiation calls (fresh snowfall, melt), the Antarctic polar
              night, and points on the terminator.
subsolar      19 to 22 March, 72 h across radconst's JULIAN = 80 branch and
              xtime past 1440 min: points placed at the subsolar latitude
              and at the longitude where a step lands on local noon, so the
              unclamped cosine rounds above one (calc_coszen's v4.3 clamp),
              plus the equator, the dateline and the antisolar point.
edge          a 6 h morning with forced fit branches: both exponent clamps
              (-0.5, 2.5), fluxes at, under and just over 1 W m-2, zero
              direct beam, direct larger than global, steady fluxes, and
              a column that goes from cloud to clear to cloud, and a direct
              beam in float32's subnormal range.
stress        4096 random points over the globe, 3 h, random fluxes and
              albedos each call/step (randomised coverage).
clamp         15 April, 24 h: points where calc_coszen's float32 sum rounds
              past one at some step, found by clamp_search.c over this
              scenario's own calendar (``--clamp-calendar`` writes it,
              build.sh runs the search, and this script reads
              clamp_hits.txt), so v4.3's [-1, 1] clamp changes the word.

The fluxes are inputs (the shortwave scheme's SWDOWN/SWDDIR on each
radiation call); they are built from a float64 estimate of the call's sun
and need only be plausible, because both sides read the same words.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

F = np.float32
RADT_MIN = 15.0
DT_S = 300.0


def calendar(start, elapsed):
    """The engine's own calendar scalars (gpuwm.core.swint), copied here so
    the column builder runs without the engine importable."""
    from datetime import timedelta
    valid = start + timedelta(seconds=float(elapsed))
    julday = int(valid.timetuple().tm_yday)
    hour = (valid.hour + valid.minute / 60.0 + valid.second / 3600.0
            + valid.microsecond / 3.6e9)
    julian = F((julday - 1) + hour / 24.0)
    gmt = F(start.hour + start.minute / 60.0 + start.second / 3600.0
            + start.microsecond / 3.6e9)
    xtime = F(float(elapsed) / 60.0)
    return julian, xtime, gmt


def approx_coszen(lat, lon, julian, xtime, gmt):
    """float64 estimate of calc_coszen (for building plausible fluxes)."""
    j = float(julian)
    dpd = 360.0 / 365.0
    sx = dpd * (j - 80.0) if j >= 80.0 else dpd * (j + 285.0)
    declin = np.arcsin(np.sin(np.radians(23.5)) * np.sin(np.radians(sx)))
    da = 2 * np.pi * (j - 1.0) / 365.0
    eot = (0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
           - 0.014615 * np.cos(2 * da) - 0.04089 * np.sin(2 * da)) * 229.18
    xt24 = np.fmod(float(xtime), 1440.0) + eot
    tloc = float(gmt) + xt24 / 60.0 + np.asarray(lon, np.float64) / 15.0
    h = np.radians(15.0 * (tloc - 12.0))
    la = np.radians(np.asarray(lat, np.float64))
    return np.sin(la) * np.sin(declin) + np.cos(la) * np.cos(declin) * np.cos(h)


def sequence(start, hours):
    nstep = int(round(hours * 3600.0 / DT_S))
    per_call = int(round(RADT_MIN * 60.0 / DT_S))
    elapsed = np.arange(nstep) * DT_S
    cal = [calendar(start, e) for e in elapsed]
    julian = np.array([c[0] for c in cal], F)
    xtime = np.array([c[1] for c in cal], F)
    gmt = np.array([c[2] for c in cal], F)
    radflag = (np.arange(nstep) % per_call == 0).astype(np.int32)
    return elapsed, julian, xtime, gmt, radflag


def call_suns(lat, lon, julian, xtime, gmt, radflag):
    idx = np.flatnonzero(radflag)
    return idx, np.stack([approx_coszen(
        lat, lon, julian[s], F(xtime[s] + F(F(RADT_MIN) * F(0.5))), gmt[s])
        for s in idx])


def clear_sky(cz, tau):
    cz = np.maximum(cz, 0.0)
    return np.where(cz > 1.0e-4, 1361.0 * tau * cz ** 1.15, 0.0)


def build_conus_day(rng):
    start = datetime(2026, 7, 15, 10, 0)
    elapsed, julian, xtime, gmt, radflag = sequence(start, 24.0)
    regimes = {
        "clear_land": (32, (30.0, 47.0), (-100.0, -80.0), 0.18),
        "high_terrain": (24, (35.0, 45.0), (-112.0, -105.0), 0.20),
        "ocean": (32, (20.0, 40.0), (-130.0, -70.0), 0.08),
        "convective": (40, (30.0, 42.0), (-102.0, -85.0), 0.17),
        "overcast": (32, (35.0, 48.0), (-125.0, -75.0), 0.20),
        "dusk_dawn_west_east": (32, (25.0, 50.0), (-125.0, -67.0), 0.16),
    }
    lat, lon, alb, label = [], [], [], []
    for name, (n, la, lo, a) in regimes.items():
        lat.append(rng.uniform(*la, n)); lon.append(rng.uniform(*lo, n))
        alb.append(np.full(n, a)); label += [name] * n
    lat = np.concatenate(lat).astype(F); lon = np.concatenate(lon).astype(F)
    alb0 = np.concatenate(alb)
    label = np.array(label)
    idx, cz = call_suns(lat, lon, julian, xtime, gmt, radflag)
    ncall, ncol = cz.shape
    tau = np.ones((ncall, ncol)) * 0.78
    dirf = np.ones((ncall, ncol)) * 0.85
    hi = label == "high_terrain"
    tau[:, hi] = 0.9; dirf[:, hi] = 0.9
    oc = label == "ocean"
    tau[:, oc] = 0.74; dirf[:, oc] = 0.8
    cv = label == "convective"
    # afternoon convection: cloud cover switches call to call after 17Z
    hour = (start.hour + idx * DT_S / 3600.0) % 24.0
    storm = (hour >= 17.0) | (hour < 2.0)
    for i in np.flatnonzero(cv):
        cloud = rng.uniform(0.0, 1.0, ncall) < 0.55
        tau[storm & cloud, i] = rng.uniform(0.08, 0.45, (storm & cloud).sum())
        dirf[storm & cloud, i] = rng.uniform(0.0, 0.25, (storm & cloud).sum())
        dirf[storm & ~cloud, i] = rng.uniform(0.55, 0.88, (storm & ~cloud).sum())
    ov = label == "overcast"
    tau[:, ov] = rng.uniform(0.08, 0.16, (ncall, ov.sum()))
    # direct beam under 1 W m-2 through thick stratus: max(1, flux) makes
    # both logs zero, so the fit exponent is exactly 0
    dirf[:, ov] = 0.0
    swdown = clear_sky(cz, tau)
    swddir = swdown * dirf
    swddir[:, ov] = np.where(swdown[:, ov] > 0,
                             rng.uniform(0.05, 0.95, (ncall, ov.sum())), 0.0)
    nstep = julian.size
    albedo = np.repeat(alb0[None, :], nstep, 0)
    return dict(start=start, elapsed=elapsed, julian=julian, xtime=xtime,
                gmt=gmt, radflag=radflag, xlat=lat, xlon=lon,
                swdown=swdown.astype(F), swddir=swddir.astype(F),
                albedo=albedo.astype(F), label=label)


def build_polar(rng):
    start = datetime(2026, 6, 21, 0, 0)
    elapsed, julian, xtime, gmt, radflag = sequence(start, 24.0)
    lat = np.concatenate([
        [90.0, 89.99, 89.9, 88.0, 85.0],
        rng.uniform(70.0, 84.0, 27),             # Arctic sea ice / snow
        rng.uniform(65.5, 67.5, 16),             # near the Arctic circle
        rng.uniform(-90.0, -70.0, 16),           # Antarctic polar night
        [-66.6, -66.4, -66.0, -65.0],            # terminator, south
    ]).astype(F)
    lat[48] = F(-90.0)
    lon = rng.uniform(-180.0, 180.0, lat.size).astype(F)
    lon[0] = F(0.0)
    label = np.array(["pole"] * 5 + ["arctic_snow_ice"] * 27
                     + ["arctic_circle"] * 16 + ["antarctic_night"] * 16
                     + ["terminator_south"] * 4)
    idx, cz = call_suns(lat, lon, julian, xtime, gmt, radflag)
    ncall, ncol = cz.shape
    tau = rng.uniform(0.55, 0.8, (ncall, ncol))
    dirf = rng.uniform(0.3, 0.8, (ncall, ncol))
    swdown = clear_sky(cz, tau)
    swddir = swdown * dirf
    # some pole columns steady to the bit (the fit sees no change at all)
    swdown[:, 1] = np.where(cz[:, 1] > 1e-4, 412.5, 0.0)
    swddir[:, 1] = np.where(cz[:, 1] > 1e-4, 250.0, 0.0)
    nstep = julian.size
    base = np.where(np.abs(lat) > 69.0, 0.8, 0.55)
    base[label == "antarctic_night"] = 0.85
    albedo = np.repeat(base[None, :], nstep, 0)
    # live albedo between radiation calls: snowfall jumps, steady melt
    for i in range(5, 32):
        jump = rng.integers(1, nstep - 1)
        albedo[jump:, i] = rng.uniform(0.82, 0.9)
    melt = np.linspace(0.0, 0.25, nstep)
    for i in range(32, 48):
        albedo[:, i] = np.maximum(0.3, base[i] - melt)
    return dict(start=start, elapsed=elapsed, julian=julian, xtime=xtime,
                gmt=gmt, radflag=radflag, xlat=lat, xlon=lon,
                swdown=swdown.astype(F), swddir=swddir.astype(F),
                albedo=albedo.astype(F), label=label)


def subsolar_points(julian, xtime, gmt, steps, rng, per_step=6):
    """Points at the subsolar latitude whose longitude puts a step on local
    noon (hrang ~ 0): the float32 cosine can round past one there."""
    lat, lon = [], []
    for s in steps:
        j = float(julian[s])
        dpd = 360.0 / 365.0
        sx = dpd * (j - 80.0) if j >= 80.0 else dpd * (j + 285.0)
        dec = np.degrees(np.arcsin(np.sin(np.radians(23.5))
                                   * np.sin(np.radians(sx))))
        da = 2 * np.pi * (j - 1.0) / 365.0
        eot = (0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
               - 0.014615 * np.cos(2 * da) - 0.04089 * np.sin(2 * da)) * 229.18
        xt24 = np.fmod(float(xtime[s]), 1440.0) + eot
        lo = 15.0 * (12.0 - float(gmt[s]) - xt24 / 60.0)
        lo = (lo + 180.0) % 360.0 - 180.0
        for k in range(per_step):
            lat.append(dec + rng.uniform(-2e-3, 2e-3))
            lon.append(lo + rng.uniform(-2e-3, 2e-3))
    return np.array(lat), np.array(lon)


def build_subsolar(rng):
    start = datetime(2026, 3, 19, 6, 0)
    elapsed, julian, xtime, gmt, radflag = sequence(start, 72.0)
    steps = np.linspace(10, julian.size - 10, 40).astype(int)
    slat, slon = subsolar_points(julian, xtime, gmt, steps, rng)
    other_lat = [0.0, 0.0, 0.0, 0.0, 0.5, -0.5, 23.44, -23.44, 45.0, -45.0]
    other_lon = [180.0, -180.0, 179.99, -179.99, 0.0, 90.0, -90.0, 135.0,
                 -135.0, 10.0]
    lat = np.concatenate([slat, other_lat]).astype(F)
    lon = np.concatenate([slon, other_lon]).astype(F)
    label = np.array(["subsolar"] * slat.size + ["equator_dateline"] * 10)
    idx, cz = call_suns(lat, lon, julian, xtime, gmt, radflag)
    ncall, ncol = cz.shape
    swdown = clear_sky(cz, rng.uniform(0.7, 0.82, (ncall, ncol)))
    swddir = swdown * rng.uniform(0.6, 0.9, (ncall, ncol))
    nstep = julian.size
    albedo = np.repeat(rng.uniform(0.06, 0.25, ncol)[None, :], nstep, 0)
    return dict(start=start, elapsed=elapsed, julian=julian, xtime=xtime,
                gmt=gmt, radflag=radflag, xlat=lat, xlon=lon,
                swdown=swdown.astype(F), swddir=swddir.astype(F),
                albedo=albedo.astype(F), label=label)


def build_edge(rng):
    start = datetime(2026, 5, 1, 12, 0)
    elapsed, julian, xtime, gmt, radflag = sequence(start, 6.0)
    ncol = 96
    lat = rng.uniform(30.0, 45.0, ncol).astype(F)
    lon = rng.uniform(-95.0, -75.0, ncol).astype(F)
    idx, cz = call_suns(lat, lon, julian, xtime, gmt, radflag)
    ncall = cz.shape[0]
    day = cz > 1e-4
    swdown = clear_sky(cz, 0.78)
    swddir = swdown * 0.8
    label = np.array(["edge"] * ncol, dtype=object)
    names = ["clamp_high", "clamp_low", "flux_one", "flux_under_one",
             "flux_just_over_one", "zero_direct", "direct_over_global",
             "steady", "cloud_clear_cloud", "huge_jump", "tiny_global",
             "alternating", "subnormal_direct"]
    groups = np.array_split(np.arange(ncol), len(names))
    for g, name in zip(groups, names):
        label[g] = name
        for i in g:
            if name == "clamp_high":     # flux collapses as the sun climbs
                swdown[:, i] *= np.where(np.arange(ncall) % 2, 0.02, 1.0)
                swddir[:, i] *= np.where(np.arange(ncall) % 2, 0.01, 1.0)
            elif name == "clamp_low":    # flux leaps as the sun barely moves
                swdown[:, i] *= np.where(np.arange(ncall) % 2, 1.0, 0.05)
                swddir[:, i] *= np.where(np.arange(ncall) % 2, 1.0, 0.03)
            elif name == "flux_one":
                swdown[:, i] = np.where(day[:, i], 1.0, 0.0)
                swddir[:, i] = np.where(day[:, i], 1.0, 0.0)
            elif name == "flux_under_one":
                swdown[:, i] = np.where(day[:, i], rng.uniform(0.1, 0.99, ncall), 0)
                swddir[:, i] = np.where(day[:, i], rng.uniform(0.0, 0.09, ncall), 0)
            elif name == "flux_just_over_one":
                swdown[:, i] = np.where(day[:, i], np.float32(1.0000001), 0)
                swddir[:, i] = np.where(day[:, i], np.float32(1.0000002), 0)
            elif name == "zero_direct":
                swddir[:, i] = 0.0
            elif name == "direct_over_global":
                swddir[:, i] = swdown[:, i] * 1.05
            elif name == "steady":
                swdown[:, i] = np.where(day[:, i], 640.0, 0.0)
                swddir[:, i] = np.where(day[:, i], 500.0, 0.0)
            elif name == "cloud_clear_cloud":
                f = np.where((np.arange(ncall) // 4) % 2 == 0, 0.15, 1.0)
                swdown[:, i] *= f
                swddir[:, i] *= np.where(f < 1, 0.0, 1.0)
            elif name == "huge_jump":
                swdown[:, i] = np.where(day[:, i], np.where(
                    np.arange(ncall) % 3 == 0, 1500.0, 3.0), 0.0)
                swddir[:, i] = swdown[:, i] * 0.5
            elif name == "tiny_global":
                swdown[:, i] = np.where(day[:, i], 1.0e-3, 0.0)
                swddir[:, i] = np.where(day[:, i], 1.0e-30, 0.0)
            elif name == "subnormal_direct":
                # a direct beam in float32's subnormal range: the scalar
                # CPU keeps it, a flush-to-zero unit would not
                swddir[:, i] = np.where(day[:, i], 1.0e-40, 0.0)
                swdown[:, i] = np.where(day[:, i], 2.0e-39, 0.0)
            elif name == "alternating":
                swdown[:, i] = np.where(day[:, i], np.where(
                    np.arange(ncall) % 2, 700.0, 300.0), 0.0)
                swddir[:, i] = swdown[:, i] * 0.7
    nstep = julian.size
    albedo = np.repeat(rng.uniform(0.1, 0.3, ncol)[None, :], nstep, 0)
    return dict(start=start, elapsed=elapsed, julian=julian, xtime=xtime,
                gmt=gmt, radflag=radflag, xlat=lat, xlon=lon,
                swdown=swdown.astype(F), swddir=swddir.astype(F),
                albedo=albedo.astype(F), label=label.astype(str))


def build_stress(rng):
    start = datetime(2026, 10, 7, 15, 0)
    elapsed, julian, xtime, gmt, radflag = sequence(start, 3.0)
    ncol = 4096
    lat = rng.uniform(-90.0, 90.0, ncol).astype(F)
    lon = rng.uniform(-180.0, 180.0, ncol).astype(F)
    idx, cz = call_suns(lat, lon, julian, xtime, gmt, radflag)
    ncall = cz.shape[0]
    swdown = clear_sky(cz, rng.uniform(0.02, 0.95, (ncall, ncol)))
    swddir = swdown * rng.uniform(0.0, 1.0, (ncall, ncol))
    nstep = julian.size
    albedo = rng.uniform(0.03, 0.95, (nstep, ncol))
    return dict(start=start, elapsed=elapsed, julian=julian, xtime=xtime,
                gmt=gmt, radflag=radflag, xlat=lat, xlon=lon,
                swdown=swdown.astype(F), swddir=swddir.astype(F),
                albedo=albedo.astype(F), label=np.array(["stress"] * ncol))


CLAMP_START = datetime(2026, 4, 15, 0, 0)
CLAMP_HOURS = 24.0
BUILD_DIR = Path(".")


def write_clamp_calendar(build):
    """``step julian xtime gmt`` (hex floats) for clamp_search.c."""
    _e, julian, xtime, gmt, _r = sequence(CLAMP_START, CLAMP_HOURS)
    lines = [f"{i} {float(julian[i]).hex()} {float(xtime[i]).hex()} "
             f"{float(gmt[i]).hex()}\n" for i in range(julian.size)]
    (build / "clamp_cal.txt").write_text("".join(lines), encoding="ascii",
                                         newline="\n")


def build_clamp(rng):
    elapsed, julian, xtime, gmt, radflag = sequence(CLAMP_START, CLAMP_HOURS)
    hits = {}
    for line in (BUILD_DIR / "clamp_hits.txt").read_text().split("\n"):
        if not line.strip():
            continue
        step, la, lo, _cz = line.split()
        hits.setdefault(int(step), []).append(
            (float.fromhex(la), float.fromhex(lo)))
    lat, lon = [], []
    for step in sorted(hits)[:48]:
        seen = set()
        for la, lo in hits[step]:
            if la in seen:
                continue
            seen.add(la)
            lat.append(la); lon.append(lo)
            if len(seen) == 2:
                break
    if len(lat) < 32:
        raise SystemExit(f"clamp_hits.txt gave {len(lat)} points; want 32+")
    lat = np.array(lat, F); lon = np.array(lon, F)
    idx, cz = call_suns(lat, lon, julian, xtime, gmt, radflag)
    ncall, ncol = cz.shape
    swdown = clear_sky(cz, rng.uniform(0.72, 0.85, (ncall, ncol)))
    swddir = swdown * rng.uniform(0.7, 0.9, (ncall, ncol))
    nstep = julian.size
    albedo = np.repeat(rng.uniform(0.06, 0.2, ncol)[None, :], nstep, 0)
    return dict(start=CLAMP_START, elapsed=elapsed, julian=julian,
                xtime=xtime, gmt=gmt, radflag=radflag, xlat=lat, xlon=lon,
                swdown=swdown.astype(F), swddir=swddir.astype(F),
                albedo=albedo.astype(F),
                label=np.array(["subsolar_clamp"] * ncol))


BUILDERS = {"conus_day": build_conus_day, "polar": build_polar,
            "subsolar": build_subsolar, "edge": build_edge,
            "stress": build_stress, "clamp": build_clamp}


def write_input(path, sc):
    ncol = sc["xlat"].size
    nstep = sc["julian"].size
    ncall = int(sc["radflag"].sum())
    assert sc["swdown"].shape == (ncall, ncol)
    with open(path, "wb") as fh:
        fh.write(np.array([ncol, nstep, ncall], np.int32).tobytes())
        fh.write(np.array([RADT_MIN], F).tobytes())
        fh.write(sc["xlat"].astype(F).tobytes())
        fh.write(sc["xlon"].astype(F).tobytes())
        for k in ("julian", "xtime", "gmt"):
            fh.write(sc[k].astype(F).tobytes())
        fh.write(sc["radflag"].astype(np.int32).tobytes())
        # Fortran (ncol, ncall): column fastest == C (ncall, ncol)
        fh.write(np.ascontiguousarray(sc["swdown"], F).tobytes())
        fh.write(np.ascontiguousarray(sc["swddir"], F).tobytes())
        fh.write(np.ascontiguousarray(sc["albedo"], F).tobytes())


def main() -> None:
    global BUILD_DIR
    build = Path(sys.argv[1])
    build.mkdir(parents=True, exist_ok=True)
    BUILD_DIR = build
    if "--clamp-calendar" in sys.argv[2:]:
        write_clamp_calendar(build)
        return
    rng = np.random.default_rng(20261007)
    pack = {"scenarios": np.array(list(BUILDERS)),
            "radt_minutes": np.array(RADT_MIN, F),
            "dt_seconds": np.array(DT_S)}
    for name, fn in BUILDERS.items():
        sc = fn(rng)
        write_input(build / f"{name}.in.bin", sc)
        for k, v in sc.items():
            if k == "start":
                pack[f"{name}/start"] = np.array(v.isoformat())
            else:
                pack[f"{name}/{k}"] = np.asarray(v)
        print(f"{name}: {sc['xlat'].size} columns, {sc['julian'].size} steps, "
              f"{int(sc['radflag'].sum())} radiation calls")
    np.savez(build / "columns.npz", **pack)


if __name__ == "__main__":
    main()
