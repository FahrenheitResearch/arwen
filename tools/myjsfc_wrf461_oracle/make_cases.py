"""Column inputs for the Eta similarity surface-layer oracle.

    python make_cases.py OUT_DIR

Writes ``case_<name>.bin`` (the stream run_myjsfc.F90 reads) and
``cases.txt``.  Deterministic: a fixed seed per set, NumPy only, every word
cast to float32 once, so the same script writes the same bytes on any host.

Three sets:

* ``cold``  ITIMESTEP 1, 2, 3 -- the first-step branches (USTAR=0.1, TZ0=TSK,
  QS=QLOW, AKMS=AKHS=CXCHS over a calm sea) and two warm calls on WRF's own
  carried state.
* ``warm``  ITIMESTEP 8, 9, 10 from seeded INOUT state, so the sea branch
  starts in each of its three viscous regimes (USTAR < 0.225,
  0.225 <= USTAR < 0.7, USTAR >= 0.7) on its first iteration.
* ``nz2``   the two-level minimum column (the PBLH scan has one level).

Each set mixes the regimes the scheme separates: convective and stable
land, snow and sea ice (land branch, cold skin, small roughness), warm and
cold ocean, high terrain (HT up to 4.4 km, PSFC down to about 58 kPa), and
edge columns: calm air (DU2 floor), cloud water at the lowest level, a TKE
column that never drops below EPSQ2 (LPBL = 1), a dead TKE column, a deep
boundary layer above 1000 m (BTGH branch), extreme instability and extreme
stability (zeta clamps), a near-saturated cold column (shelter clamp) and
storm-force wind over water and a lowest layer under 4 m deep (the 2 m
diagnostic above the lowest level).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

F = np.float32
G, RD, CP = 9.81, 287.0, 1004.5
CAPA = RD / CP

REGIMES = ("convective_land", "stable_land", "snow_ice", "ocean_warm",
           "ocean_cold", "high_terrain", "calm", "cloudy", "deep_tke",
           "dead_tke", "deep_pbl", "very_unstable", "very_stable",
           "cold_saturated", "storm_ocean", "rough_forest",
           "thin_lowest_layer")


def _qsat(t, p):
    es = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
    return 0.622 * es / (p - es)


def _column(rng, regime, nz):
    """One column as a dict of float64 values; cast once at write time."""
    ht = 0.0
    psfc = rng.uniform(98000.0, 102500.0)
    dz0 = rng.uniform(25.0, 70.0)
    wind = rng.uniform(1.5, 9.0)
    dt_skin = 0.0                    # TSK - T1
    t1 = rng.uniform(278.0, 300.0)
    rh = rng.uniform(0.3, 0.9)
    xland = 1.0
    z0base = 10.0 ** rng.uniform(-2.0, -0.3)
    mavail = rng.uniform(0.2, 1.0)
    qc1 = 0.0
    tke0 = rng.uniform(0.3, 2.0)
    pbl_depth = rng.uniform(150.0, 900.0)
    lapse = 0.0065
    if regime == "convective_land":
        dt_skin = rng.uniform(4.0, 16.0)
        t1 = rng.uniform(290.0, 312.0)
        pbl_depth = rng.uniform(600.0, 2500.0)
    elif regime == "stable_land":
        dt_skin = -rng.uniform(1.0, 9.0)
        wind = rng.uniform(0.6, 5.0)
        tke0 = rng.uniform(0.05, 0.4)
        pbl_depth = rng.uniform(40.0, 250.0)
    elif regime == "snow_ice":
        t1 = rng.uniform(238.0, 271.0)
        dt_skin = -rng.uniform(0.2, 7.0)
        z0base = 10.0 ** rng.uniform(-4.0, -2.3)
        mavail = 1.0
        rh = rng.uniform(0.6, 0.98)
    elif regime == "ocean_warm":
        xland = 2.0
        dt_skin = rng.uniform(0.5, 4.0)
        wind = rng.uniform(2.0, 18.0)
        mavail = 1.0
        rh = rng.uniform(0.65, 0.95)
    elif regime == "ocean_cold":
        xland = 2.0
        dt_skin = -rng.uniform(0.5, 5.0)
        wind = rng.uniform(1.0, 12.0)
        mavail = 1.0
        rh = rng.uniform(0.7, 0.98)
    elif regime == "high_terrain":
        ht = rng.uniform(1300.0, 4400.0)
        psfc = 101325.0 * (1.0 - 2.25577e-5 * ht) ** 5.25588
        t1 = rng.uniform(255.0, 292.0)
        dz0 = rng.uniform(12.0, 45.0)
        dt_skin = rng.uniform(-8.0, 14.0)
        rh = rng.uniform(0.1, 0.8)
    elif regime == "calm":
        wind = 0.0
        dt_skin = rng.uniform(-3.0, 3.0)
        xland = 1.0 + float(rng.integers(0, 2))
    elif regime == "cloudy":
        rh = 1.0
        qc1 = rng.uniform(1.0e-5, 1.2e-3)
        dt_skin = rng.uniform(-2.0, 2.0)
        xland = 1.0 + float(rng.integers(0, 2))
    elif regime == "deep_tke":
        tke0 = rng.uniform(3.0, 6.0)
        pbl_depth = 1.0e7            # never drops below EPSQ2
        dt_skin = rng.uniform(1.0, 8.0)
    elif regime == "dead_tke":
        tke0 = 0.0
        dt_skin = rng.uniform(-4.0, 4.0)
    elif regime == "deep_pbl":
        tke0 = rng.uniform(1.5, 4.0)
        pbl_depth = rng.uniform(1300.0, 3500.0)
        dt_skin = rng.uniform(5.0, 15.0)
        t1 = rng.uniform(295.0, 312.0)
        xland = 1.0 + float(rng.integers(0, 2))
    elif regime == "very_unstable":
        wind = rng.uniform(0.2, 1.0)
        dt_skin = rng.uniform(15.0, 30.0)
        t1 = rng.uniform(295.0, 315.0)
        xland = 1.0 + float(rng.integers(0, 2))
    elif regime == "very_stable":
        wind = rng.uniform(0.3, 1.2)
        dt_skin = -rng.uniform(10.0, 22.0)
        tke0 = rng.uniform(0.0, 0.08)
        xland = 1.0 + float(rng.integers(0, 2))
    elif regime == "cold_saturated":
        t1 = rng.uniform(250.0, 275.0)
        rh = 1.0
        dt_skin = rng.uniform(0.5, 4.0)
        mavail = 1.0
    elif regime == "storm_ocean":
        xland = 2.0
        wind = rng.uniform(22.0, 38.0)
        dt_skin = rng.uniform(-1.0, 3.0)
        mavail = 1.0
        rh = rng.uniform(0.8, 0.98)
    elif regime == "rough_forest":
        z0base = rng.uniform(1.0, 2.653)
        dt_skin = rng.uniform(-3.0, 9.0)
    elif regime == "thin_lowest_layer":
        # ZSL below 2 m puts the 2 m level above the lowest model level, so
        # TH02 leaves the THZ0..THLOW bracket (module_sf_myjsfc.F:964-967).
        dz0 = rng.uniform(1.5, 3.8)
        dt_skin = rng.uniform(-6.0, 10.0)
        xland = 1.0 + float(rng.integers(0, 2))
    else:
        raise ValueError(regime)

    # Layer depths: stretched, never round, thinner near the ground.
    dz = np.empty(nz)
    for k in range(nz):
        dz[k] = min(dz0 * 1.09 ** k, 650.0) * rng.uniform(0.97, 1.03)
    zface = np.concatenate(([0.0], np.cumsum(dz)))
    zmid = 0.5 * (zface[:-1] + zface[1:])

    tmid = t1 - lapse * (zmid - zmid[0])
    tmid = np.maximum(tmid, 190.0)
    tbar = tmid.mean()
    pint = psfc * np.exp(-G * zface / (RD * tbar))
    pmid = psfc * np.exp(-G * zmid / (RD * tbar))
    th = tmid * (1.0e5 / pmid) ** CAPA
    qv = np.minimum(rh, 1.0) * _qsat(tmid, pmid) * np.exp(-zmid / 3000.0)
    qv[0] = rh * _qsat(tmid[0], pmid[0])
    qc = np.zeros(nz)
    qc[0] = qc1
    direction = rng.uniform(0.0, 2.0 * np.pi)
    speed = wind * (1.0 + 0.25 * np.log1p(zmid / 50.0))
    if wind == 0.0:
        speed = np.zeros(nz)
    u = speed * np.cos(direction)
    v = speed * np.sin(direction)
    shape = np.clip(1.0 - zmid / pbl_depth, 0.0, None) ** 1.5
    q2 = tke0 * shape + rng.uniform(0.0, 0.02, nz) * (tke0 > 0.0)
    if regime == "dead_tke":
        q2 = np.zeros(nz)

    tsk = t1 + dt_skin
    return {
        "ht": ht, "dz": dz, "pmid": pmid, "th": th, "t": tmid, "qv": qv,
        "qc": qc, "u": u, "v": v, "q2": q2, "pint": pint, "tsk": tsk,
        "xland": xland, "mavail": mavail, "z0base": z0base,
        "znt": (z0base if xland < 1.5 else 10.0 ** rng.uniform(-4.5, -3.0)),
    }


def _warm_state(rng, col, slot):
    """INOUT state a warm call might inherit; USTAR walks the sea regimes."""
    ust_band = ((0.03, 0.2), (0.25, 0.65), (0.75, 1.6))[slot % 3]
    qsat_sfc = _qsat(col["tsk"], col["pint"][0])
    return {
        "qsfc": qsat_sfc * (0.98 if col["xland"] > 1.5 else rng.uniform(0.4, 1.0)),
        "thz0": col["tsk"] * (1.0e5 / col["pint"][0]) ** CAPA
        + rng.uniform(-0.5, 0.5),
        "qz0": qsat_sfc * rng.uniform(0.6, 1.0),
        "uz0": rng.uniform(-0.3, 0.3) if col["xland"] > 1.5 else 0.0,
        "vz0": rng.uniform(-0.3, 0.3) if col["xland"] > 1.5 else 0.0,
        "ustar": rng.uniform(*ust_band),
        "akhs": rng.uniform(0.002, 0.08),
        "akms": rng.uniform(0.002, 0.08),
    }


def _write(path, cols, states, nz, nsteps, it0):
    ncol = len(cols)
    with open(path, "wb") as out:
        np.asarray([ncol, nz, nsteps, it0], dtype="<i4").tofile(out)

        def put2(values):
            np.asarray(values, dtype="<f4").tofile(out)

        def put3(name, nk):
            block = np.stack([c[name] for c in cols], axis=1)   # (nk, ncol)
            assert block.shape == (nk, ncol)
            np.asarray(block, dtype="<f4").tofile(out)

        put2([c["ht"] for c in cols])
        for name in ("dz", "pmid", "th", "t", "qv", "qc", "u", "v", "q2"):
            put3(name, nz)
        put3("pint", nz + 1)
        for name in ("tsk", "xland", "mavail", "z0base"):
            put2([c[name] for c in cols])
        for name in ("qsfc", "thz0", "qz0", "uz0", "vz0", "ustar"):
            put2([s[name] for s in states])
        put2([c["znt"] for c in cols])
        for name in ("akhs", "akms"):
            put2([s[name] for s in states])


def build(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = (("cold", 96, 40, 3, 1, 20261007),
            ("warm", 96, 40, 3, 8, 20261008),
            ("nz2", 32, 2, 2, 1, 20261009))
    names = []
    for name, ncol, nz, nsteps, it0, seed in sets:
        rng = np.random.default_rng(seed)
        cols, states = [], []
        for i in range(ncol):
            regime = REGIMES[i % len(REGIMES)]
            col = _column(rng, regime, nz)
            cols.append(col)
            if it0 == 1:
                # Cold start: the INOUT words WRF's init leaves behind.
                states.append({"qsfc": 0.0, "thz0": 0.0, "qz0": 0.0,
                               "uz0": 0.0, "vz0": 0.0, "ustar": 0.1,
                               "akhs": 0.0, "akms": 0.0})
            else:
                states.append(_warm_state(rng, col, i))
        _write(out_dir / f"case_{name}.bin", cols, states, nz, nsteps, it0)
        names.append(name)
    (out_dir / "cases.txt").write_text("\n".join(names) + "\n")
    return names


def build_stress(out_dir: Path, ncol: int) -> list[str]:
    """Two large sets, every column's regime drawn at random.

    Not the committed fixture: a measurement that the curated sets above
    are not hiding a regime, rerun by stress.sh.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, it0, seed in (("stress_cold", 1, 7001), ("stress_warm", 5, 7002)):
        rng = np.random.default_rng(seed)
        cols, states = [], []
        for i in range(ncol):
            regime = REGIMES[int(rng.integers(0, len(REGIMES)))]
            col = _column(rng, regime, 50)
            cols.append(col)
            if it0 == 1:
                states.append({"qsfc": 0.0, "thz0": 0.0, "qz0": 0.0,
                               "uz0": 0.0, "vz0": 0.0, "ustar": 0.1,
                               "akhs": 0.0, "akms": 0.0})
            else:
                states.append(_warm_state(rng, col, i))
        _write(out_dir / f"case_{name}.bin", cols, states, 50, 4, it0)
        names.append(name)
    (out_dir / "cases.txt").write_text("\n".join(names) + "\n")
    return names


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--stress":
        print(" ".join(build_stress(Path(sys.argv[1]), int(sys.argv[3]))))
    else:
        print(" ".join(build(Path(sys.argv[1]))))
