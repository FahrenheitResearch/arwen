"""Inputs for the radar latent heating oracle, and the oracle's answers.

Writes one folder per case holding raw float32 arrays (``ref``, ``t``, ``q``,
``p``, ``h``, C-order ``(nz, ny, nx)``, which is NOAA's Fortran ``(nlon,
nlat, nsig)``), runs NOAA's compiled code on them through
``oracle_ref2tten`` and keeps what it wrote (``pblh``, ``refcone``,
``tten``).  NumPy here builds test inputs only; it computes no product.

    python cases.py --oracle-dir <dir with oracle_ref2tten> --out <dir>

Cases
-----
``branches``
    64 x 64 x 50.  Columns come in 4 x 4 blocks, each block one recipe, so
    every branch of NOAA's routines is reached by construction: both sides
    of 277.15 K and of 28 dBZ, cold echo depth either side of 200 hPa,
    coverage depth either side of 300 hPa, the boundary-layer top below and
    above level 7, the 0.01 K/s cap, no echo, no coverage, each cone-fill
    class and both of its fill rules, the sentinel boundaries, and the
    stratiform reclassing.  ``branches-all-precip`` is the same inputs with
    the convection-only switch off.
``random-<seed>``
    200 x 200 x 50, three seeds.  Regimes are drawn per 5 x 5 block of
    columns and values per point.
``smooth``
    A float64 53 x 37 field with non-zero edges, for ``SMOOTH`` alone.
``vinterp-<levels>``
    ``vinterp_radar_ref`` alone on 50 x 36 x 40 for each of its three
    mosaic level sets (21, 31, 33).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

NO_COVERAGE = np.float32(-99999.0)
NO_ECHO = np.float32(-99.0)


def _background(rng, nz, ny, nx, *, mixed_levels):
    """A plausible column set: pressure falling from about 1000 to 100 hPa,
    a mixed layer ``mixed_levels`` deep under a stable profile, vapour
    falling with height, heights from a standard-atmosphere fit."""
    k = np.arange(nz, dtype=np.float64)[:, None, None]
    psfc = 1000.0 + rng.uniform(-25.0, 15.0, size=(1, ny, nx))
    p = psfc - (psfc - 100.0) * k / (nz - 1)
    p = p + rng.uniform(-0.4, 0.4, size=(nz, ny, nx))
    mixed = np.asarray(mixed_levels, dtype=np.float64)[None, :, :]
    theta = 300.0 + rng.uniform(-3.0, 3.0, size=(1, ny, nx)) \
        + 0.85 * np.maximum(k - mixed + 1.0, 0.0) \
        + np.where(k >= mixed, 1.6, 0.0)
    theta = theta + rng.uniform(-0.02, 0.02, size=(nz, ny, nx))
    q = 0.014 * np.exp(-k / 9.0) * rng.uniform(0.6, 1.2, size=(nz, ny, nx))
    h = 44330.0 * (1.0 - (p / psfc) ** 0.1903)
    h = h + 25.0 + rng.uniform(0.0, 4.0, size=(nz, ny, nx))
    return (p.astype(np.float32), theta.astype(np.float32),
            q.astype(np.float32), h.astype(np.float32))


def _fill(column, lo, hi, value):
    column[lo:hi] = value


def _recipes():
    """Each recipe writes one column of reflectivity (zero-based levels);
    the number beside it is the mixed-layer depth it wants, which moves the
    boundary-layer top either side of level 7."""
    def no_coverage(c, rng):
        c[:] = NO_COVERAGE

    def no_echo(c, rng):
        c[:] = NO_ECHO

    def deep_storm(c, rng):
        c[:] = NO_ECHO
        _fill(c, 2, 36, 45.0 + rng.uniform(-1.0, 1.0))

    def capped(c, rng):
        c[:] = NO_ECHO
        _fill(c, 4, 31, 65.0 + rng.uniform(-1.0, 1.0))

    def warm_weak(c, rng):                     # warmer than 4 C, under 28 dBZ
        c[:] = NO_ECHO
        _fill(c, 6, 16, 20.0 + rng.uniform(-1.0, 1.0))

    def cold_weak(c, rng):                     # heated, never "convective"
        c[:] = NO_ECHO
        _fill(c, 24, 35, 15.0 + rng.uniform(-1.0, 1.0))

    def cold_shallow(c, rng):                  # 28 dBZ, under 200 hPa deep
        c[:] = NO_ECHO
        _fill(c, 26, 34, 31.0 + rng.uniform(-1.0, 1.0))

    def cold_deep(c, rng):                     # 28 dBZ, over 200 hPa deep
        c[:] = NO_ECHO
        _fill(c, 26, 39, 31.0 + rng.uniform(-1.0, 1.0))

    def cold_edge(top):
        def recipe(c, rng):
            c[:] = NO_ECHO
            _fill(c, 26, top, 30.0)
        return recipe

    def coverage(depth):
        def recipe(c, rng):
            c[:] = NO_COVERAGE
            _fill(c, 12, 12 + depth, NO_ECHO)
        return recipe

    def cone(strength, base=11, top=31, low=None):
        def recipe(c, rng):
            c[:] = NO_ECHO
            _fill(c, 0, base, NO_COVERAGE)
            _fill(c, base, top, strength + rng.uniform(-0.5, 0.5))
            if low is not None:
                _fill(c, base, base + 3, low)
        return recipe

    def cone_low(c, rng):                      # fill starts under 750 m
        c[:] = NO_ECHO
        _fill(c, 0, 2, NO_COVERAGE)
        _fill(c, 2, 20, 40.0)

    def cone_no_max(c, rng):                   # echo only above the scan
        c[:] = NO_COVERAGE
        _fill(c, 26, 34, 30.0)

    def below_bottom(c, rng):
        c[:] = NO_ECHO
        _fill(c, 1, 5, 45.0)

    def mixed(c, rng):
        draw = rng.uniform(size=c.size)
        c[:] = np.where(draw < 0.3, NO_COVERAGE,
                        np.where(draw < 0.6, NO_ECHO,
                                 rng.uniform(-5.0, 70.0, size=c.size)))

    def thresholds(c, rng):
        c[:] = NO_ECHO
        values = (28.0, 27.999, 28.001, 0.001, 0.0009, 0.0, -99.999,
                  -100.0, -100.001, -199.99, -200.0, -200.01, 19.0, 19.001,
                  20.0, 19.999, 24.999, 25.0, 49.999, 50.0, 55.0)
        for index, value in enumerate(values):
            c[8 + index] = value

    return [
        (no_coverage, 4), (no_echo, 4), (deep_storm, 4), (capped, 3),
        (warm_weak, 5), (cold_weak, 4), (cold_shallow, 4), (cold_deep, 4),
        (cold_edge(36), 4), (cold_edge(37), 4), (cold_edge(38), 4),
        (coverage(15), 4), (coverage(16), 4), (coverage(17), 4),
        (coverage(18), 4),
        (cone(22.0), 3), (cone(27.0), 4), (cone(33.0), 5), (cone(38.0), 6),
        (cone(43.0), 7), (cone(48.0), 8), (cone(60.0), 9),
        (cone(60.0, low=25.0), 4), (cone(41.0, base=20, top=30), 12),
        (cone_low, 1), (cone_no_max, 4), (below_bottom, 4), (mixed, 2),
        (mixed, 10), (thresholds, 4), (deep_storm, 13), (capped, 9),
    ]


def branches_case(nz=50, ny=64, nx=64, seed=20261003):
    rng = np.random.default_rng(seed)
    recipes = _recipes()
    block = 4
    by, bx = ny // block, nx // block
    kind = (rng.permutation(by * bx) % len(recipes)).reshape(by, bx)
    kind_full = np.repeat(np.repeat(kind, block, axis=0), block, axis=1)
    mixed_levels = np.empty((ny, nx), dtype=np.float64)
    ref = np.empty((nz, ny, nx), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            recipe, depth = recipes[int(kind_full[j, i])]
            mixed_levels[j, i] = depth
            column = np.empty(nz, dtype=np.float32)
            recipe(column, rng)
            ref[:, j, i] = column
    # One 12 x 12 patch of 65 dBZ echo through levels 5-31: the tendency
    # there is far above the 0.01 K/s cap before smoothing, and the patch
    # is wide enough that its middle is still at the cap after the two
    # passes, so the cap is visible in the stored field and not only in
    # the arithmetic behind it.
    ref[4:31, 40:52, 40:52] = np.float32(65.0)
    ref[:4, 40:52, 40:52] = NO_ECHO
    ref[31:, 40:52, 40:52] = NO_ECHO
    mixed_levels[40:52, 40:52] = 3
    p, theta, q, h = _background(rng, nz, ny, nx, mixed_levels=mixed_levels)
    # Put some points' temperature within a few hundredths of 277.15 K at
    # an echo level, on both sides, so the warm/cold split is exercised at
    # its edge and not only far from it.
    exner = (p.astype(np.float64) / 1000.0) ** (287.04 / 1004.6)
    for _ in range(40):
        j, i = int(rng.integers(1, ny - 1)), int(rng.integers(1, nx - 1))
        k = int(rng.integers(12, 30))
        target = 277.15 + rng.choice([-0.02, -0.001, 0.0, 0.001, 0.02])
        theta[k, j, i] = np.float32(target / exner[k, j, i])
        ref[k, j, i] = np.float32(rng.choice([27.0, 28.0, 35.0]))
    return {"ref": ref, "t": theta, "q": q, "p": p, "h": h}


def fixture_case(nz=50, ny=12, nx=12, seed=20261004):
    """A small case for the committed test fixture: every recipe on at
    least three interior columns, one column each."""
    rng = np.random.default_rng(seed)
    recipes = _recipes()
    order = rng.permutation(ny * nx) % len(recipes)
    mixed_levels = np.empty((ny, nx), dtype=np.float64)
    ref = np.empty((nz, ny, nx), dtype=np.float32)
    for n, kind in enumerate(order):
        j, i = divmod(n, nx)
        recipe, depth = recipes[int(kind)]
        mixed_levels[j, i] = depth
        column = np.empty(nz, dtype=np.float32)
        recipe(column, rng)
        ref[:, j, i] = column
    p, theta, q, h = _background(rng, nz, ny, nx, mixed_levels=mixed_levels)
    return {"ref": ref, "t": theta, "q": q, "p": p, "h": h}


def write_fixture(path: Path, oracle_dir: Path, work: Path):
    """Run the oracle on :func:`fixture_case` with the switch on and off and
    store inputs and answers in one compressed file."""
    arrays = fixture_case()
    stored = {f"in_{name}": value for name, value in arrays.items()}
    for flag in (1, 0):
        folder = work / f"fixture-{flag}"
        meta = write_case(folder, arrays, oracle_dir, convection_only=flag)
        shape = (meta["nz"], meta["ny"], meta["nx"])
        for name, s in (("pblh", shape[1:]), ("refcone", shape),
                        ("tten", shape)):
            stored[f"out{flag}_{name}"] = np.fromfile(
                folder / f"{name}.bin", dtype=np.float32).reshape(s)
    np.savez_compressed(path, **stored)


def random_case(seed, nz=50, ny=200, nx=200):
    rng = np.random.default_rng(seed)
    block = 5
    regime = rng.integers(0, 6, size=(ny // block, nx // block))
    regime = np.repeat(np.repeat(regime, block, axis=0), block, axis=1)
    mixed_levels = rng.integers(1, 15, size=(ny, nx)).astype(np.float64)
    p, theta, q, h = _background(rng, nz, ny, nx, mixed_levels=mixed_levels)
    draw = rng.uniform(size=(nz, ny, nx))
    echo = rng.uniform(-5.0, 70.0, size=(nz, ny, nx)).astype(np.float32)
    # regime -> (share with no coverage, share with no echo)
    shares = np.array([[0.9, 0.05], [0.05, 0.9], [0.1, 0.3], [0.4, 0.3],
                       [0.02, 0.1], [0.3, 0.6]])
    none = shares[regime, 0][None, :, :]
    clear = shares[regime, 1][None, :, :]
    ref = np.where(draw < none, NO_COVERAGE,
                   np.where(draw < none + clear, NO_ECHO, echo))
    # Storm cores: coherent strong echo through a deep layer in some blocks.
    levels = np.arange(nz)[:, None, None]
    core = (regime == 4)[None, :, :] & (levels > 5) & (levels < 34)
    ref = np.where(core & (draw > 0.15), 30.0 + 25.0 * draw, ref)
    return {"ref": ref.astype(np.float32), "t": theta, "q": q, "p": p, "h": h}


def vinterp_case(folder: Path, oracle_dir: Path, nlevels: int, seed: int,
                 nz=50, ny=36, nx=40):
    """Inputs for ``vinterp_radar_ref`` alone, and its answer.

    Valid values, no echo (-99), missing (-999, -99999) and out-of-range
    values (|Z| >= 90) mixed per mosaic point; terrain 0-1500 m; some
    columns on flat ground with model heights exactly on mosaic levels, so
    the half-open intervals are exercised at their edges, the top level
    (no coverage at or above it) included.
    """
    from gpuwm.da.radar_tten import MOSAIC_LEVELS_KM

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    zh = rng.uniform(0.0, 1500.0, size=(ny, nx)).astype(np.float32)
    k = np.arange(nz, dtype=np.float64)[:, None, None]
    h = (k + 0.5) * 420.0 + rng.uniform(-50.0, 50.0, size=(nz, ny, nx))
    h = h.astype(np.float32)
    levels_m = np.asarray(MOSAIC_LEVELS_KM[nlevels], np.float32) * 1000.0
    for n in range(12):
        j, i = 2 + n, 3 + n
        zh[j, i] = 0.0
        h[:len(levels_m), j, i] = levels_m[:nz]
    draw = rng.uniform(size=(nlevels, ny, nx))
    mosaic = rng.uniform(-30.0, 80.0, size=(nlevels, ny, nx))
    mosaic = np.where(draw < 0.15, -99.0, mosaic)
    mosaic = np.where((draw >= 0.15) & (draw < 0.25), -999.0, mosaic)
    mosaic = np.where((draw >= 0.25) & (draw < 0.32), -99999.0, mosaic)
    mosaic = np.where((draw >= 0.32) & (draw < 0.35), 95.0, mosaic)
    mosaic = np.where((draw >= 0.35) & (draw < 0.38), -95.0, mosaic)
    mosaic = np.where((draw >= 0.38) & (draw < 0.40), 89.9, mosaic)
    mosaic = np.where((draw >= 0.40) & (draw < 0.42), -98.95, mosaic)
    mosaic = mosaic.astype(np.float32)
    mosaic.tofile(folder / "mosaic.bin")
    h.tofile(folder / "h.bin")
    zh.tofile(folder / "zh.bin")
    subprocess.run([str(oracle_dir / "oracle_vinterp"), str(folder), str(nx),
                    str(ny), str(nz), str(nlevels)], check=True,
                   stdout=subprocess.DEVNULL)
    (folder / "case.json").write_text(json.dumps(
        {"nz": nz, "ny": ny, "nx": nx, "levels": nlevels}, indent=1))


def write_case(folder: Path, arrays, oracle_dir: Path, convection_only=1):
    folder.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = arrays["ref"].shape
    for name, value in arrays.items():
        np.ascontiguousarray(value, dtype=np.float32).tofile(
            folder / f"{name}.bin")
    with open(folder / "oracle.log", "w") as log:
        subprocess.run(
            [str(oracle_dir / "oracle_ref2tten"), str(folder), str(nx),
             str(ny), str(nz), str(convection_only)],
            check=True, stdout=log, stderr=subprocess.STDOUT)
    meta = {"nz": nz, "ny": ny, "nx": nx, "convection_only": convection_only}
    (folder / "case.json").write_text(json.dumps(meta, indent=1))
    return meta


def smooth_case(folder: Path, oracle_dir: Path, ny=53, nx=37, seed=7):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    field = rng.uniform(-0.01, 0.01, size=(ny, nx))
    field.astype(np.float64).tofile(folder / "smooth_in.bin")
    out = {}
    for passes in (1, 2, 5):
        subprocess.run([str(oracle_dir / "oracle_smooth"), str(folder),
                        str(nx), str(ny), str(passes)], check=True)
        (folder / "smooth_out.bin").rename(folder / f"smooth_out_{passes}.bin")
        out[str(passes)] = f"smooth_out_{passes}.bin"
    (folder / "case.json").write_text(json.dumps(
        {"ny": ny, "nx": nx, "passes": out}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=None,
                        help="also write the small committed test fixture "
                             "(tests/data/radar_tten_oracle_fixture.npz)")
    args = parser.parse_args()
    if args.fixture is not None:
        write_fixture(args.fixture, args.oracle_dir, args.out)
    write_case(args.out / "branches", branches_case(), args.oracle_dir)
    write_case(args.out / "branches-all-precip", branches_case(),
               args.oracle_dir, convection_only=0)
    for seed in (1, 2, 3):
        write_case(args.out / f"random-{seed}", random_case(seed),
                   args.oracle_dir)
    smooth_case(args.out / "smooth", args.oracle_dir)
    for seed, levels in enumerate((21, 31, 33), start=1):
        vinterp_case(args.out / f"vinterp-{levels}", args.oracle_dir,
                     levels, seed)
    print("cases written under", args.out)


if __name__ == "__main__":
    main()
