"""Synthetic inputs for the two NOAA oracles (test scaffolding, NumPy).

Writes ``case_a.bin`` (the retrieval: temperature 230 to 310 K, pressure 200
to 1000 hPa, reflectivity -10 to 70 dBZ plus both sentinels and every
threshold, 1,048,576 points the Fortran loop visits), ``case_b.bin`` (the
final precipitation analysis: 4,096 columns built to reach every branch) and
``case_b_noctp.bin`` (the same columns with no satellite cloud-top pressure).
Layouts are documented in ``oracle_a.f90`` and ``oracle_b.f90``.

Usage: python make_cases.py <out-dir> [--seed N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NO_ECHO = -99.0
MISSING = -99999.0

#: Case B column categories, in the order they are dealt out.
CATEGORIES = (
    "no_coverage", "no_echo", "no_echo_satellite_clear",
    "no_echo_satellite_cloudy", "cold_surface", "warm_background_above",
    "warm_background_below", "warm_light_precipitation",
    "strongest_echo_first_level", "strongest_echo_top_level", "random",
    "edges",
)

#: Regional rd / cp as NOAA's constants.f90 sets them (rd = 287.04 at :266,
#: cp = 1004.6 at :73); used only to build a theta that gives a chosen
#: temperature.  The oracle recomputes temperature from theta itself.
KAPPA = 287.04 / 1004.6


def case_a(path: Path, rng) -> dict:
    nx = ny = 130
    nz = 66
    shape = (nz, ny, nx)
    t = rng.uniform(230.0, 310.0, shape).astype(np.float32)
    p = rng.uniform(20000.0, 100000.0, shape).astype(np.float32)
    ref = rng.uniform(-10.0, 70.0, shape)
    pick = rng.random(shape)
    ref[pick < 0.04] = NO_ECHO
    ref[(pick >= 0.04) & (pick < 0.08)] = MISSING
    # A third of the echoes are float32-valued, which is what the radar-grid
    # adapter hands the kernel; the rest are arbitrary doubles, which is
    # what NOAA's vertical interpolation produces.
    as_single = (pick >= 0.08) & (pick < 0.40)
    ref[as_single] = ref[as_single].astype(np.float32).astype(np.float64)
    edges = np.array([0.0, 28.0, 55.0, 15.0, 70.0, -10.0, 1.0e-300,
                      np.nextafter(28.0, 0.0), np.nextafter(28.0, 99.0),
                      np.nextafter(55.0, 0.0), np.nextafter(55.0, 99.0),
                      np.nextafter(0.0, -1.0)])
    chosen = (pick >= 0.40) & (pick < 0.43)
    ref[chosen] = rng.choice(edges, int(chosen.sum()))
    # Temperature thresholds: 0 C, 5 C, and tc = -0.1 (the MIN of :142).
    anchors = np.array([273.15, 278.15, 273.05], np.float32)
    near = np.concatenate([
        anchors, np.nextafter(anchors, np.float32(0.0)),
        np.nextafter(anchors, np.float32(1.0e9)),
        anchors + np.float32(1.0e-3), anchors - np.float32(1.0e-3)])
    tpick = rng.random(shape)
    chosen = tpick < 0.03
    t[chosen] = rng.choice(near, int(chosen.sum()))
    with open(path, "wb") as stream:
        stream.write(np.array([nx, ny, nz], np.int32).tobytes())
        stream.write(np.ascontiguousarray(t).tobytes())
        stream.write(np.ascontiguousarray(p).tobytes())
        stream.write(np.ascontiguousarray(ref).tobytes())
    return {"nx": nx, "ny": ny, "nz": nz,
            "visited_points": (nx - 2) * (ny - 2) * (nz - 2),
            "echo_points": int(np.count_nonzero(ref >= 0.0))}


def _background(rng, nsig, scale: str):
    """One column's ges_qr, ges_qnr, ges_qs, ges_qg (float32-valued)."""
    def one(level_scale):
        present = rng.random(nsig) < 0.7
        values = level_scale * rng.lognormal(0.0, 1.0, nsig)
        return np.where(present, values, 0.0)

    if scale == "zero":
        qr = qs = qg = np.zeros(nsig)
    else:
        level_scale = {"low": 2.0e-5, "high": 2.5e-3}[scale]
        qr, qs, qg = one(level_scale), one(level_scale), one(0.5 * level_scale)
    qnr = np.where(qr > 0.0, rng.uniform(1.0e2, 1.0e6, nsig), 0.0)
    # A few inconsistent pairs, as an analysis can leave them.
    flip = rng.random(nsig) < 0.05
    qnr = np.where(flip & (qr > 0.0), 0.0, qnr)
    qnr = np.where(flip & (qr == 0.0), rng.uniform(1.0, 1.0e3, nsig), qnr)
    return tuple(np.asarray(v, np.float32).astype(np.float64)
                 for v in (qr, qnr, qs, qg))


def _echo_profile(rng, nsig, *, peak, k_peak, single: bool):
    width = rng.uniform(2.0, 8.0)
    k = np.arange(nsig)
    ref = peak - 12.0 * ((k - k_peak) / width) ** 2
    floor = rng.uniform(0.0, 5.0)
    below = ref <= floor
    filler = np.where(rng.random(nsig) < 0.6, NO_ECHO, MISSING)
    ref = np.where(below, filler, ref)
    if single:
        ref = ref.astype(np.float32).astype(np.float64)
    return ref


def case_b(path: Path, rng) -> dict:
    lon2 = lat2 = 66
    nsig = 40
    shape = (nsig, lat2, lon2)
    theta = np.empty(shape, np.float32)
    p_hpa = np.empty(shape, np.float32)
    ref = np.full(shape, MISSING)
    ctp = np.full((lat2, lon2), MISSING, np.float32)
    ges = [np.zeros(shape) for _ in range(4)]
    categories = np.empty((lat2, lon2), np.int32)
    k = np.arange(nsig)
    for j in range(lat2):
        for i in range(lon2):
            cat = CATEGORIES[(j * lon2 + i) % len(CATEGORIES)]
            categories[j, i] = CATEGORIES.index(cat)
            psfc = rng.uniform(850.0, 1020.0)
            pcol = psfc * np.exp(-0.048 * k)
            cold = cat == "cold_surface" or (
                cat in ("strongest_echo_first_level", "random", "edges")
                and rng.random() < 0.4)
            # Never within 0.25 K of the 5 C threshold: the kernel tests the
            # float32 temperature, NOAA the double product it came from.
            t1 = rng.uniform(255.0, 277.9) if cold else rng.uniform(278.4, 305.0)
            tcol = np.maximum(t1 - rng.uniform(1.2, 1.9) * k, 205.0)
            theta[:, j, i] = (tcol / (pcol / 1000.0) ** KAPPA).astype(np.float32)
            p_hpa[:, j, i] = pcol.astype(np.float32)
            scale = rng.choice(["zero", "low", "high"])
            single = bool(rng.random() < 0.5)
            column = np.full(nsig, MISSING)
            if cat == "no_coverage":
                pass
            elif cat.startswith("no_echo"):
                column = np.where(rng.random(nsig) < 0.6, NO_ECHO, MISSING)
                column[rng.integers(0, nsig)] = NO_ECHO
                if cat == "no_echo_satellite_clear":
                    ctp[j, i] = rng.choice([1013.0, rng.uniform(1010.0, 1049.9)])
                elif cat == "no_echo_satellite_cloudy":
                    ctp[j, i] = rng.uniform(150.0, 1009.9)
                scale = rng.choice(["low", "high"])
            elif cat == "cold_surface":
                column = _echo_profile(rng, nsig, peak=rng.uniform(8.0, 60.0),
                                       k_peak=rng.integers(1, nsig - 1),
                                       single=single)
            elif cat == "warm_background_above":
                column = _echo_profile(rng, nsig, peak=rng.uniform(8.0, 45.0),
                                       k_peak=rng.integers(1, nsig - 1),
                                       single=single)
                scale = "high"
            elif cat == "warm_background_below":
                column = _echo_profile(rng, nsig, peak=rng.uniform(30.0, 68.0),
                                       k_peak=rng.integers(1, nsig - 1),
                                       single=single)
                scale = rng.choice(["zero", "low"])
            elif cat == "warm_light_precipitation":
                column = _echo_profile(rng, nsig, peak=rng.uniform(15.0, 28.0),
                                       k_peak=rng.integers(1, nsig - 1),
                                       single=single)
            elif cat == "strongest_echo_first_level":
                column = _echo_profile(rng, nsig, peak=rng.uniform(20.0, 60.0),
                                       k_peak=0, single=single)
            elif cat == "strongest_echo_top_level":
                column = _echo_profile(rng, nsig, peak=rng.uniform(20.0, 60.0),
                                       k_peak=nsig - 1, single=single)
            elif cat == "random":
                draw = rng.random(nsig)
                column = np.where(draw < 0.35, rng.uniform(0.1, 65.0, nsig),
                                  np.where(draw < 0.7, NO_ECHO, MISSING))
                ctp[j, i] = rng.choice([MISSING, 1013.0, 600.0])
            else:  # edges
                column = rng.choice(
                    [0.0, -100.0, 15.0, 28.0, NO_ECHO, MISSING, 35.0, 35.0,
                     np.nextafter(15.0, 0.0), np.nextafter(28.0, 99.0),
                     -99.999, -100.001, -999.0, -500.0], nsig)
                if rng.random() < 0.5:
                    # No echo anywhere: the largest value is 0 or -100.
                    column = np.where(column > 0.0, rng.choice([0.0, -100.0]),
                                      column)
                ctp[j, i] = rng.choice([MISSING, 1010.0, 1050.0, 1013.0,
                                        np.float32(1009.99)])
            ref[:, j, i] = column
            qr, qnr, qs, qg = _background(rng, nsig, scale)
            if cat == "edges":
                # Negative values an unbounded analysis can leave: the clamp.
                for values in (qr, qnr, qs, qg):
                    bad = rng.random(nsig) < 0.1
                    values[bad] = -np.float64(np.float32(1.0e-6))
            for store, values in zip(ges, (qr, qnr, qs, qg)):
                store[:, j, i] = values
    for target, cloud_top in ((path, ctp),
                              (path.with_name("case_b_noctp.bin"),
                               np.full_like(ctp, MISSING))):
        with open(target, "wb") as stream:
            stream.write(np.array([lon2, lat2, nsig], np.int32).tobytes())
            for array in (theta, p_hpa, ref, cloud_top, *ges):
                stream.write(np.ascontiguousarray(array).tobytes())
    np.save(path.with_name("case_b_categories.npy"), categories)
    interior = categories[1:-1, 1:-1]
    return {"lon2": lon2, "lat2": lat2, "nsig": nsig,
            "interior_columns": int(interior.size),
            "columns_by_category": {
                name: int(np.count_nonzero(interior == index))
                for index, name in enumerate(CATEGORIES)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary = {"seed": args.seed,
               "case_a": case_a(out / "case_a.bin", rng),
               "case_b": case_b(out / "case_b.bin", rng)}
    (out / "cases.json").write_text(json.dumps(summary, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
