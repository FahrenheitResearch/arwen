#!/usr/bin/env python3
"""Cut a small committed fixture from full column sets: real columns chosen
per process regime, with WRF v4.6.1's own answers for them.

Every regime a repair of the port acts in is represented, so the fixture
check goes red if any of them is undone, plus the regimes the remaining
named differences and the two rounding-decided rates live in, plus ordinary
columns that must stay at rounding.  Columns are chosen from WRF's own
checkpoint streams (``instrument_wrf_rates.py``); nothing about the port
enters the choice.

The fixture holds the raw float32 column inputs (``prepare`` forms the rest
exactly as the adapter does), the pristine WRF driver's outputs, the
instrumented driver's sixty-four process rates as float32, and the few
checkpoint values the fixture check's rounding rules read.
``fixture_check.py`` then needs only a C++ compiler and the Thompson table
set to grade the port against it; no Fortran, no WRF source.

``--mp8`` writes the classic Thompson companion of an existing mp=28
fixture: the same columns, cut by the mp=28 regimes above, with the answers
WRF's same module gives them as the THOMPSON case of the microphysics driver
calls it (``run_columns_classic``, is_aerosol_aware false).  Both fixtures
therefore grade the two schemes on identical inputs.

usage: make_fixture.py WRF_BUILD_DIR OUT.npz COLUMNS.npz [COLUMNS.npz ...]
       make_fixture.py --mp8 WRF_BUILD_DIR OUT_MP8.npz FIXTURE_MP28.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real_column_parity as R  # noqa: E402

R1 = 1.0e-12
R2 = 1.0e-6
f32 = np.float32

#: The raw column fields ``prepare`` reads.
RAW_KEYS = ("th", "p", "geop", "w", "nwfa2d", "nifa2d", *R.SPECIES)
#: Checkpoint values the fixture check's rules read, as float32.
CHECKPOINT_KEYS = (("cp1", "rho"), ("cp1", "qr1d"), ("cp1", "nr1d"),
                   ("cp1", "t1d"), ("cp2", "ssatw"), ("cp2", "prw_vcd"),
                   ("cpx", "t1d"), ("cpx", "ng1d"))


def _stage(cps, cp, var, dt):
    """WRF's working value ``X1d + Xten*DT`` in float32."""
    return R.wrf_stage(cps, cp, var, dt)


def regimes(inp, cps, dt):
    """``{label: (ncol,) bool}``: which columns carry each regime.

    A column qualifies if any of its levels does.  Levels WRF did not reach
    (no_micro columns) are NaN in the checkpoints and never qualify.
    """
    c1, c2 = cps["cp1"], cps["cp2"]
    rho = c1["rho"]
    lev = {}
    with np.errstate(invalid="ignore"):
        # A: ice arriving with mass and no number (:1855-1858).
        lev["A entry ice without number"] = (
            (c1["qi1d"] > R1) & (c1["ni1d"] * rho <= R2))
        # K: an orphan number, n > 0 over q <= R1 (:1871-1872, :1900-1901),
        # for ice where WRF nucleates on the zeroed number (:2627), and for
        # rain.
        lev["K orphan ice number where WRF nucleates"] = (
            (inp["qi"] <= R1) & (inp["ni"] > 0) & (c1["pri_inu"] > 0))
        lev["K orphan rain number"] = (inp["qr"] <= R1) & (inp["nr"] > 0)
        # C/F: graupel or rain left at or below R1 as a concentration after
        # the sources, with mass above zero (:3068-3091, :3118-3160).
        qg1 = _stage(cps, "cp1", "qg", dt)
        qr1 = _stage(cps, "cp1", "qr", dt)
        lev["C/F graupel emptied at the source stage"] = (
            (c1["qg1d"] > R1) & (qg1 * rho <= R1))
        lev["C rain emptied at the source stage"] = (
            (c1["qr1d"] > R1) & (qr1 * rho <= R1))
        # F: graupel melting at warm levels (the number re-balance).
        lev["F melting graupel"] = (
            (c1["qg1d"] > 1.0e-7) & (c1["t1d"] > 275.0))
        # D/H: a mixing ratio above R1 whose concentration is not.
        band = np.zeros_like(rho, dtype=bool)
        for var in ("qi", "qs", "qr"):
            q = _stage(cps, "cp2", var, dt)
            band |= (q > R1) & (q * rho <= R1)
        lev["D/H mass above R1 below it as a concentration"] = band
        # E: ice collected by snow or rain.
        # (NaN compares unequal to zero: the no_micro columns are excluded
        # by name wherever a rate is tested for being non-zero.)
        reached = np.isfinite(rho)
        lev["E ice collection"] = reached & (
            (c1["pni_sci"] != 0) | (c1["pni_rci"] != 0))
        # B: snow riming cloud at cold levels.
        lev["B cold snow riming"] = (c1["prs_scw"] != 0) & (c1["t1d"] < 273.15)
        # O: graupel at or below R1 at entry, which WRF zeroes (:1941) and
        # whose vapour number rate it therefore never forms.
        lev["O graupel at or below R1 at entry"] = reached & (
            (inp["qg"] > 0) & (inp["qg"] <= R1))
        # Rain evaporation just after the adjustment saturated the air.
        eps = float(np.finfo(np.float32).eps)
        lev["rain evaporation at saturation"] = (
            (c2["prv_rev"] != 0) & (c2["prw_vcd"] != 0)
            & (np.abs(c2["ssatw"]) <= 16 * eps))
        # Rain self-collection at the break-up diameter (:2159-2176).
        rr = c1["qr1d"] * rho
        nn = np.maximum(c1["nr1d"] * rho, R2)
        lam = np.cbrt(np.pi * 1000.0 * nn / np.where(rr > 0, rr, 1.0))
        mvd = 3.672 / lam
        lev["rain break-up diameter"] = (
            (rr > R1) & (np.abs(mvd - 1950.0e-6) < 20.0e-6))
        # G: vapour exactly zero somewhere in a column with microphysics.
        lev["G vapour at zero"] = (inp["qv"] == 0) & np.isfinite(rho)
        # Ordinary regimes that must stay at rounding.
        lev["deep graupel"] = c1["qg1d"] > 1.0e-4
        lev["warm rain"] = (c1["qr1d"] > 1.0e-4) & (c1["t1d"] > 290.0)
        lev["supersaturated cloud"] = (c1["qc1d"] > 1.0e-4) & (
            c2["prw_vcd"] > 0)
    out = {label: np.nan_to_num(mask.astype(float), nan=0.0).any(axis=1)
           for label, mask in lev.items()}
    # L: the cloud fallout gate (:3485, :3645).  A column whose post-source
    # cloud the adjustment cleared everywhere, and that still ends the
    # adjustment with cloud in its lowest 500 m.
    qc1 = _stage(cps, "cp1", "qc", dt)
    qc2 = _stage(cps, "cp2", "qc", dt)
    with np.errstate(invalid="ignore"):
        lqc = qc1 > R1
        cleared = (c2["prw_vcd"] != 0) & (qc2 * rho <= R1)
        agl = np.cumsum(inp["dz"], axis=1) - inp["dz"]
        low_cloud = ((qc2 * rho > R1) & (agl <= 500.0)).any(axis=1)
    out["L cloud formed where the gate is shut"] = (
        lqc.any(axis=1) & ~(lqc & ~cleared).any(axis=1) & low_cloud)
    micro = cps["cp1"]["present"]
    all_empty = np.ones(inp["p"].shape[0], bool)
    for var in ("qc", "qr", "qi", "qs", "qg"):
        all_empty &= (inp[var] <= R1).all(axis=1)
    out["clear air"] = all_empty & ~micro
    return out


#: How many columns each regime contributes to the fixture, taken one per
#: source in turn so that the regimes span the saved states.
PER_REGIME = {
    "A entry ice without number": 3,
    "K orphan ice number where WRF nucleates": 3,
    "K orphan rain number": 2,
    "C/F graupel emptied at the source stage": 3,
    "C rain emptied at the source stage": 2,
    "F melting graupel": 3,
    "D/H mass above R1 below it as a concentration": 3,
    "E ice collection": 2,
    "B cold snow riming": 2,
    "O graupel at or below R1 at entry": 2,
    "rain evaporation at saturation": 3,
    "rain break-up diameter": 2,
    "G vapour at zero": 2,
    "L cloud formed where the gate is shut": 3,
    "deep graupel": 2,
    "warm rain": 2,
    "supersaturated cloud": 2,
    "clear air": 1,
}


def _run_wrf(build, inp, dt, stem, mp=28):
    ncol, nz = inp["p"].shape
    run_dir = build / "run"
    in_path = run_dir / f"{stem}-in.bin"
    binary = R.WRF_BINARY[mp]
    R.write_wrf_input(in_path, inp, dt, mp)
    R.run_wrf(build / "pristine" / binary, run_dir, in_path,
              run_dir / f"{stem}-pristine.out")
    R.run_wrf(build / "rates" / binary, run_dir, in_path,
              run_dir / f"{stem}-rates.out")
    a = (run_dir / f"{stem}-pristine.out").read_bytes()
    if a != (run_dir / f"{stem}-rates.out").read_bytes():
        raise SystemExit("instrumented WRF changed its outputs")
    wrf = R.read_wrf_output(run_dir / f"{stem}-pristine.out", ncol, nz, mp)
    cps = R.read_checkpoints(run_dir, ncol, nz)
    for cp in R.SCHEMA:
        (run_dir / f"wrf-{cp}.bin").unlink(missing_ok=True)
    for suffix in ("-in.bin", "-pristine.out", "-rates.out"):
        (run_dir / f"{stem}{suffix}").unlink(missing_ok=True)
    return wrf, cps


def pick(masks_by_source, seed=0):
    """``[(source, column, label)]``: PER_REGIME columns per regime, one per
    source in turn, never the same column twice."""
    rng = np.random.default_rng(seed)
    taken, chosen = set(), []
    names = list(masks_by_source)
    for label, count in PER_REGIME.items():
        pools = {}
        for src in names:
            hits = [int(h) for h in np.flatnonzero(masks_by_source[src][label])]
            rng.shuffle(hits)
            pools[src] = hits
        got = 0
        while got < count and any(pools.values()):
            for src in names:
                if got >= count:
                    break
                while pools[src]:
                    col = pools[src].pop()
                    if (src, col) not in taken:
                        taken.add((src, col))
                        chosen.append((src, col, label))
                        got += 1
                        break
    return chosen


def _answers(build, raw, dt, stem, mp):
    """The fixture arrays for ``raw`` columns: the inputs, WRF's outputs,
    its sixty-four rates and the checkpoint values the rules read."""
    inp = R.prepare(raw, mp)
    wrf, cps = _run_wrf(build, inp, dt, stem, mp)
    arrays = {f"col_{k}": v for k, v in raw.items()}
    arrays.update({f"wrf_{k}": np.asarray(v, f32) for k, v in wrf.items()})
    for name in R.RATES:
        cp = "cp2" if name in R.LATE_RATES else "cp1"
        arrays[f"rate_{name}"] = cps[cp][name].astype(f32)
    for cp, name in CHECKPOINT_KEYS:
        arrays[f"{cp}_{name}"] = cps[cp][name].astype(f32)
    arrays["cp1_present"] = cps["cp1"]["present"]
    arrays["dt"] = np.float32(dt)
    arrays["mp_physics"] = np.int32(mp)
    receipt = (build / "BUILD-RECEIPT.txt").read_text(encoding="utf-8")
    arrays["wrf_build_receipt"] = np.array(receipt)
    return arrays


def main_mp8(argv):
    """``--mp8 BUILD OUT FIXTURE28``: the classic companion fixture."""
    build, out, source = Path(argv[0]), Path(argv[1]), Path(argv[2])
    z = np.load(source)
    raw = {k: z[f"col_{k}"] for k in RAW_KEYS}
    arrays = _answers(build, raw, float(z["dt"]), "fixture-mp8", 8)
    arrays["labels"] = z["labels"]
    arrays["origin"] = z["origin"]
    arrays["columns_from"] = np.array(source.name)
    np.savez_compressed(out, **arrays)
    print(out, raw["p"].shape[0], "columns, mp_physics 8")
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--mp8":
        if len(argv) != 5:
            print(__doc__)
            return 2
        return main_mp8(argv[2:])
    if len(argv) < 4:
        print(__doc__)
        return 2
    build, out, sources = Path(argv[1]), Path(argv[2]), argv[3:]
    loaded, masks, dt = {}, {}, None
    for src in sources:
        cols = R.load_columns(src)
        dt = float(cols["dt"]) if dt is None else dt
        if float(cols["dt"]) != dt:
            raise SystemExit("all sources must share one time step")
        inp = R.prepare(cols)
        _wrf, cps = _run_wrf(build, inp, dt, "fixture-select")
        loaded[src] = cols
        masks[src] = regimes(inp, cps, dt)
        print(Path(src).name, {k: int(v.sum()) for k, v in masks[src].items()})
    chosen = pick(masks)
    parts = [{k: np.asarray(loaded[src][k])[[col]] for k in RAW_KEYS}
             for src, col, _label in chosen]
    labels = [label for _s, _c, label in chosen]
    origin = [f"{Path(src).name}:{col}" for src, col, _l in chosen]
    raw = {k: np.concatenate([p[k] for p in parts]).astype(f32)
           for k in RAW_KEYS}
    inp = R.prepare(raw)
    ncol = inp["p"].shape[0]
    wrf, cps = _run_wrf(build, inp, dt, "fixture")

    arrays = {f"col_{k}": v for k, v in raw.items()}
    arrays.update({f"wrf_{k}": np.asarray(v, f32) for k, v in wrf.items()})
    for name in R.RATES:
        cp = "cp2" if name in R.LATE_RATES else "cp1"
        arrays[f"rate_{name}"] = cps[cp][name].astype(f32)
    for cp, name in CHECKPOINT_KEYS:
        arrays[f"{cp}_{name}"] = cps[cp][name].astype(f32)
    arrays["cp1_present"] = cps["cp1"]["present"]
    arrays["dt"] = np.float32(dt)
    arrays["labels"] = np.array(labels)
    arrays["origin"] = np.array(origin)
    receipt = (build / "BUILD-RECEIPT.txt").read_text(encoding="utf-8")
    arrays["wrf_build_receipt"] = np.array(receipt)
    np.savez_compressed(out, **arrays)
    got = dict(zip(*np.unique(labels, return_counts=True)))
    print(out, ncol, "columns", {str(k): int(v) for k, v in got.items()})
    missing = [k for k in PER_REGIME if k not in got]
    if missing:
        print("regimes no source carried:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
