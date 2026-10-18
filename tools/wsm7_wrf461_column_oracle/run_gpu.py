"""Grade gpuwm's CUDA WSM7 against the WRF v4.6.1 column oracle, word for word.

Reads the SAME ``columns.bin`` / ``columns72.bin`` the WRF Fortran read
(make_columns.py), runs the production launchers
(``gpuwm.core.wsm_family.launch_column`` and ``launch_refl10cm``) step by
step exactly as WRF's wrapper runs (microphysics with its effective radii,
then REFL_10CM from th*pii), and compares every output word.

Effective radii: WRF stores metres; gpuwm's radiation contract is microns,
formed as one float32 multiply ``re * 1.0e6`` -- the multiply WRF's RRTMG
does when it reads re_cloud/re_ice/re_snow.  WRF's words are put through
that multiply before comparison.

Subcommands::

    python -m tools.wsm7_wrf461_column_oracle.run_gpu grade ORACLE_DIR \
        [--arm defined|stock|d1] [--out JSON]
        every word of every step against run_wsm7_columns.F90's output
        (GPU; the box that ran build.sh, or a copy of its outputs)
    python -m ... fortran-pair ORACLE_DIR ARM_A ARM_B [--out JSON]  (CPU)
        two Fortran arms against each other, and the probe counts
    python -m ... make-fixture ORACLE_DIR OUT_DIR                   (CPU)
        the committed fixture: the column sets, a SHA-256 of every defined-
        arm output field at every step, its final step's words, and the
        stock arm's per-column final-step digests
    python -m ... check-fixture FIXTURE_DIR [--tier N] [--out JSON] (GPU)
        grade against the committed fixture (what the pytest gate runs)

GPU work.  On a shared box run only through its GPU mutex.
GPUWM_WRF_EXACT=1 selects the strict build; unset is the default build.  The
WSM7 unit compiles with --fmad=false --ftz=false in BOTH, so both are graded
to the same words; every report states which build ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

LEVEL_IN = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "qh",
            "den", "pii", "p", "dz")
SURFACE_IN = ("rain", "snow", "graupel", "hail")
LEVEL_OUT = ("th", "qv", "qc", "qi", "qr", "qs", "qg", "qh",
             "re_cloud", "re_ice", "re_snow", "refl_10cm", "t")
SURFACE_OUT = ("rain", "rainncv", "snow", "snowncv", "graupel",
               "graupelncv", "hail", "hailncv", "sr")
#: (column set, dt, nsteps): dt 30 s is one minor loop; 60 s one loop at
#: half dtcldcr; 150 s one loop with dtcld > dtcldcr; 300 s nint(2.5) = 3
#: loops of 100 s; the 72-level set runs the deeper WSM7_KMAX tier.
RUNS = (("columns", 30.0, 12), ("columns", 60.0, 8), ("columns", 150.0, 3),
        ("columns", 300.0, 4), ("columns72", 30.0, 4))


def tag_of(stem: str, dt: float, nsteps: int) -> str:
    return f"{stem}-dt{int(dt)}-n{nsteps}"


def read_columns(path: Path):
    with open(path, "rb") as fh:
        ncol, nz = (int(v) for v in np.fromfile(fh, "<i4", 2))
        level = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in LEVEL_IN}
        surf = {n: np.fromfile(fh, "<f4", ncol) for n in SURFACE_IN}
    return ncol, nz, level, surf


def micron(re_metres):
    return (np.asarray(re_metres, np.float32) * np.float32(1.0e6)).astype(np.float32)


def read_wrf(path: Path):
    """run_wsm7_columns.F90's output, effective radii already in microns."""
    with open(path, "rb") as fh:
        ncol, nz, nsteps = (int(v) for v in np.fromfile(fh, "<i4", 3))
        dt = float(np.fromfile(fh, "<f4", 1)[0])
        steps = []
        for _ in range(nsteps):
            lv = {n: np.fromfile(fh, "<f4", nz * ncol).reshape(nz, ncol) for n in LEVEL_OUT}
            for n in ("re_cloud", "re_ice", "re_snow"):
                lv[n] = micron(lv[n])
            sf = {n: np.fromfile(fh, "<f4", ncol) for n in SURFACE_OUT}
            steps.append((lv, sf))
    return dict(ncol=ncol, nz=nz, nsteps=nsteps, dt=dt, steps=steps)


def run_gpu(columns: Path, dt: float, nsteps: int):
    """The production launchers, step by step, as WRF's wrapper runs."""
    import cupy as cp
    from gpuwm.core.wsm_family import launch_column, launch_refl10cm

    ncol, nz, level, surf = read_columns(columns)
    f32 = np.float32
    dev = {n: cp.asarray(v.reshape(nz, 1, ncol), dtype=cp.float32)
           for n, v in level.items()}
    s = {n: cp.zeros((1, ncol), dtype=cp.float32) for n in
         ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv",
          "hailnc", "hailncv", "sr")}
    for src, dst in (("rain", "rainnc"), ("snow", "snownc"),
                     ("graupel", "graupelnc"), ("hail", "hailnc")):
        s[dst][...] = cp.asarray(surf[src].reshape(1, ncol))
    effc = cp.full((nz, 1, ncol), f32(2.49), cp.float32)
    effi = cp.full((nz, 1, ncol), f32(4.99), cp.float32)
    effs = cp.full((nz, 1, ncol), f32(9.99), cp.float32)
    refl = cp.empty((nz, 1, ncol), cp.float32)
    t = cp.empty((nz, 1, ncol), cp.float32)
    species = {n: dev[n] for n in ("qc", "qi", "qr", "qs", "qg", "qh")}
    steps = []
    for _ in range(nsteps):
        launch_column(24, theta=dev["th"], qv=dev["qv"], species=species,
                      rho=dev["den"], pii=dev["pii"], pressure=dev["p"],
                      dz=dev["dz"], surface=s, effc=effc, effi=effi,
                      effs=effs, dt=dt,
                      # The WRF oracle is exact mode (finding WB5 off).
                      conservative_surface=False)
        cp.multiply(dev["th"], dev["pii"], out=t)
        launch_refl10cm(24, qv=dev["qv"], species=species, t=t,
                        pressure=dev["p"], refl=refl)
        cp.cuda.runtime.deviceSynchronize()
        g = lambda a: a.get().reshape(nz, ncol)
        lv = {"th": g(dev["th"]), "qv": g(dev["qv"]), "qc": g(dev["qc"]),
              "qi": g(dev["qi"]), "qr": g(dev["qr"]), "qs": g(dev["qs"]),
              "qg": g(dev["qg"]), "qh": g(dev["qh"]), "re_cloud": g(effc),
              "re_ice": g(effi), "re_snow": g(effs), "refl_10cm": g(refl),
              "t": g(t)}
        sf = {"rain": s["rainnc"], "rainncv": s["rainncv"], "snow": s["snownc"],
              "snowncv": s["snowncv"], "graupel": s["graupelnc"],
              "graupelncv": s["graupelncv"], "hail": s["hailnc"],
              "hailncv": s["hailncv"], "sr": s["sr"]}
        sf = {k: v.get().reshape(ncol) for k, v in sf.items()}
        steps.append((lv, sf))
    return steps


def _ordered(a):
    bits = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)


def ulp(a, b):
    return np.abs(_ordered(a) - _ordered(b))


def digest(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, "<f4").tobytes()).hexdigest()


def _pairs(a_step, b_step):
    (al, as_), (bl, bs) = a_step, b_step
    out = {n: (al[n], bl[n]) for n in LEVEL_OUT}
    out.update({n: (as_[n][None, :], bs[n][None, :]) for n in SURFACE_OUT})
    return out


def _accumulate(fields, istep, pairs, regimes, keep=None):
    for n, (w, g) in pairs.items():
        w = np.asarray(w, np.float32)
        g = np.asarray(g, np.float32)
        names = [r["name"] for r in regimes]
        if keep is not None:
            w, g = w[:, keep], g[:, keep]
            names = [names[i] for i in np.nonzero(keep)[0]]
        diff_bits = w.view(np.uint32) != g.view(np.uint32)
        u = np.where(np.isnan(w) & np.isnan(g), 0, ulp(w, g))
        gap = np.abs(w.astype(np.float64) - g.astype(np.float64))
        gap = np.where(np.isnan(gap), 0.0, gap)
        row = fields.setdefault(n, {"words": 0, "bit_mismatch": 0, "max_ulp": 0,
                                    "max_abs": 0.0, "worst": None, "columns": {},
                                    "first_step_with_a_mismatch": None})
        row["words"] += int(w.size)
        row["bit_mismatch"] += int(diff_bits.sum())
        row["max_abs"] = max(row["max_abs"], float(gap.max()) if gap.size else 0.0)
        if diff_bits.any() and row["first_step_with_a_mismatch"] is None:
            row["first_step_with_a_mismatch"] = istep + 1
        for i in np.nonzero(diff_bits.any(axis=0))[0]:
            row["columns"][names[i]] = row["columns"].get(names[i], 0) + int(diff_bits[:, i].sum())
        m = int(u.max()) if u.size else 0
        if diff_bits.any() and (row["worst"] is None or m > row["max_ulp"]):
            k, i = np.unravel_index(int(np.argmax(u)), u.shape)
            row["max_ulp"] = max(row["max_ulp"], m)
            row["worst"] = {"step": istep + 1, "level": int(k), "column": names[i],
                            "a": float(w[k, i]), "b": float(g[k, i])}


def grade(a, b_steps, regimes):
    fields = {}
    for istep, (sa, sb) in enumerate(zip(a["steps"], b_steps)):
        _accumulate(fields, istep, _pairs(sa, sb), regimes)
    return fields


def _header():
    import cupy as cp
    import gpuwm.wrf_exact as wrf_exact

    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    return {"strict_build": bool(wrf_exact.ENABLED),
            "device": name.decode() if isinstance(name, bytes) else str(name),
            "nvrtc": ".".join(map(str, cp.cuda.nvrtc.getVersion())),
            "cupy": cp.__version__}


def _summary(fields):
    return {n: [r["bit_mismatch"], r["max_ulp"]] for n, r in fields.items()
            if r["bit_mismatch"]}


def _mismatch_columns(fields) -> set:
    out = set()
    for r in fields.values():
        out.update(r["columns"])
    return out


def cmd_grade(oracle_dir: Path, arm: str):
    report = {**_header(), "arm": arm, "runs": {}}
    total = 0
    for stem, dt, nsteps in RUNS:
        regimes = json.loads((oracle_dir / f"{stem}.json").read_text())["columns"]
        tag = tag_of(stem, dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-{arm}-{tag}.bin")
        fields = grade(wrf, run_gpu(oracle_dir / f"{stem}.bin", dt, nsteps), regimes)
        bad = sum(r["bit_mismatch"] for r in fields.values())
        words = sum(r["words"] for r in fields.values())
        report["runs"][tag] = {"words": words, "bit_mismatch": bad,
                               "mismatch_columns": sorted(_mismatch_columns(fields)),
                               "fields": fields}
        total += bad
        print(f"{tag}: {bad} of {words} words differ; {_summary(fields)}", flush=True)
    report["total_bit_mismatch"] = total
    report["total_words"] = sum(r["words"] for r in report["runs"].values())
    return report


def _probe_counts(oracle_dir: Path, tag: str, regimes) -> dict:
    out = {}
    path = oracle_dir / f"probe-{tag}.txt"
    for line in path.read_text().splitlines():
        i, n = (int(v) for v in line.split())
        if n:
            out[regimes[i - 1]["name"]] = n
    return out


def cmd_fortran_pair(oracle_dir: Path, arm_a: str, arm_b: str):
    report = {"arms": [arm_a, arm_b], "runs": {}}
    for stem, dt, nsteps in RUNS:
        regimes = json.loads((oracle_dir / f"{stem}.json").read_text())["columns"]
        tag = tag_of(stem, dt, nsteps)
        a = read_wrf(oracle_dir / f"wrf-{arm_a}-{tag}.bin")
        b = read_wrf(oracle_dir / f"wrf-{arm_b}-{tag}.bin")
        fields = grade(a, b["steps"], regimes)
        report["runs"][tag] = {
            "words": sum(r["words"] for r in fields.values()),
            "bit_mismatch": sum(r["bit_mismatch"] for r in fields.values()),
            "mismatch_columns": sorted(_mismatch_columns(fields)),
            "probe_fallthrough_columns": _probe_counts(oracle_dir, tag, regimes),
            "fields": fields}
        print(f"{tag}: {report['runs'][tag]['bit_mismatch']} words differ; "
              f"{_summary(fields)}; columns {report['runs'][tag]['mismatch_columns']}",
              flush=True)
    return report


FIXTURE_FILES = ("columns.bin", "columns.json", "columns72.bin", "columns72.json",
                 "wsm7-init-words.txt", "libmvec-report.txt", "compiler.txt",
                 "oracle-sha256sums.txt", "wrf-commit.txt")


def cmd_make_fixture(oracle_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name in FIXTURE_FILES:
        shutil.copyfile(oracle_dir / name, out / name)
    digests = {"sets": {}, "runs": {}, "stock_final_column_digests": {}}
    for stem in ("columns", "columns72"):
        digests["sets"][stem] = hashlib.sha256(
            (oracle_dir / f"{stem}.bin").read_bytes()).hexdigest()
    final = {}
    for stem, dt, nsteps in RUNS:
        tag = tag_of(stem, dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-defined-{tag}.bin")
        digests["runs"][tag] = [
            {**{n: digest(lv[n]) for n in LEVEL_OUT},
             **{n: digest(sf[n]) for n in SURFACE_OUT}}
            for lv, sf in wrf["steps"]]
        lv, sf = wrf["steps"][-1]
        for n in LEVEL_OUT:
            final[f"{tag}/{n}"] = lv[n]
        for n in SURFACE_OUT:
            final[f"{tag}/{n}"] = sf[n]
        # stock WRF's final step, one short digest per column and field:
        # lets the gate name the columns the declared divergences move.
        stock = read_wrf(oracle_dir / f"wrf-stock-{tag}.bin")
        slv, ssf = stock["steps"][-1]
        digests["stock_final_column_digests"][tag] = {
            **{n: [column_digest(slv[n][:, i]) for i in range(stock["ncol"])]
               for n in LEVEL_OUT},
            **{n: [column_digest(ssf[n][i:i + 1]) for i in range(stock["ncol"])]
               for n in SURFACE_OUT}}
    (out / "wrf461-step-digests.json").write_text(json.dumps(digests, indent=1) + "\n")
    np.savez_compressed(out / "wrf461-final-step.npz", **final)
    (out / "arm-attribution.json").write_text(
        json.dumps(arm_attribution(oracle_dir), indent=1) + "\n")
    return {"written": sorted(p.name for p in out.iterdir())}


def arm_attribution(oracle_dir: Path) -> dict:
    """Which columns each declared divergence moves, per run (Fortran arms).

    ``d1_vs_stock``: columns where WRF with only the remap-search fix differs
    from stock WRF; ``probe``: columns whose stock find_kt fell through with
    arrival mass above kb (the probe arm); ``defined_vs_d1``: columns D2
    moves (REFL_10CM only, where hail is present).
    """
    out = {}
    for stem, dt, nsteps in RUNS:
        regimes = json.loads((oracle_dir / f"{stem}.json").read_text())["columns"]
        tag = tag_of(stem, dt, nsteps)
        arms = {a: read_wrf(oracle_dir / f"wrf-{a}-{tag}.bin")
                for a in ("stock", "d1", "defined")}
        d1 = grade(arms["stock"], arms["d1"]["steps"], regimes)
        d2 = grade(arms["d1"], arms["defined"]["steps"], regimes)
        d1_final, d2_final = {}, {}
        _accumulate(d1_final, nsteps - 1,
                    _pairs(arms["stock"]["steps"][-1], arms["d1"]["steps"][-1]),
                    regimes)
        _accumulate(d2_final, nsteps - 1,
                    _pairs(arms["d1"]["steps"][-1], arms["defined"]["steps"][-1]),
                    regimes)
        out[tag] = {
            "d1_vs_stock_final": sorted(_mismatch_columns(d1_final)),
            "defined_vs_d1_final": sorted(_mismatch_columns(d2_final)),
            "d1_vs_stock": sorted(_mismatch_columns(d1)),
            "d1_vs_stock_fields": _summary(d1),
            "probe": _probe_counts(oracle_dir, tag, regimes),
            "defined_vs_d1": sorted(_mismatch_columns(d2)),
            "defined_vs_d1_fields": _summary(d2)}
    return out


def column_digest(a) -> str:
    return digest(a)[:16]


def force_tier(tier: int) -> None:
    """Compile the WSM7 kernels at WSM7_KMAX = ``tier`` whatever nz is."""
    import gpuwm.core.wsm_family as fam

    fam.force_level_tier(int(tier))


def check_fixture(fixture: Path, *, runs=RUNS):
    """Grade the CUDA kernels against the committed fixture.

    Every field of every step is checked by digest against WRF with the two
    declared divergences (the defined arm) and the final step word by word.
    The stock comparison reports, per run, the columns whose final step
    differs from unmodified WRF.
    """
    digests = json.loads((fixture / "wrf461-step-digests.json").read_text())
    final = np.load(fixture / "wrf461-final-step.npz")
    report = {**_header(), "runs": {}}
    total = 0
    for stem, dt, nsteps in runs:
        regimes = json.loads((fixture / f"{stem}.json").read_text())["columns"]
        tag = tag_of(stem, dt, nsteps)
        gpu = run_gpu(fixture / f"{stem}.bin", dt, nsteps)
        step_digest_mismatch = []
        computed_step_sha256 = []
        words_checked = 0
        for istep, ((lv, sf), want) in enumerate(zip(gpu, digests["runs"][tag])):
            got = {**{n: digest(lv[n]) for n in LEVEL_OUT},
                   **{n: digest(sf[n]) for n in SURFACE_OUT}}
            computed_step_sha256.append(got)
            words_checked += sum(a.size for a in (*lv.values(), *sf.values()))
            step_digest_mismatch += [f"step {istep + 1} {n}" for n in got
                                     if got[n] != want[n]]
        lv, sf = gpu[-1]
        wrf_last = ({n: final[f"{tag}/{n}"] for n in LEVEL_OUT},
                    {n: final[f"{tag}/{n}"] for n in SURFACE_OUT})
        fields = {}
        _accumulate(fields, nsteps - 1, _pairs(wrf_last, (lv, sf)), regimes)
        stock = digests["stock_final_column_digests"][tag]
        differs_from_stock = sorted({
            regimes[i]["name"]
            for n in LEVEL_OUT for i in range(len(regimes))
            if column_digest(lv[n][:, i]) != stock[n][i]} | {
            regimes[i]["name"]
            for n in SURFACE_OUT for i in range(len(regimes))
            if column_digest(sf[n][i:i + 1]) != stock[n][i]})
        bad = sum(r["bit_mismatch"] for r in fields.values()) + len(step_digest_mismatch)
        total += bad
        report["runs"][tag] = {"final_step": fields,
                               "computed_step_sha256": computed_step_sha256,
                               "words_checked": int(words_checked),
                               "step_digest_mismatch": step_digest_mismatch,
                               "columns_differing_from_stock_wrf": differs_from_stock}
    report["total_mismatch"] = total
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade")
    g.add_argument("oracle_dir", type=Path)
    g.add_argument("--arm", default="defined")
    g.add_argument("--tier", type=int)
    g.add_argument("--out", type=Path)
    f = sub.add_parser("fortran-pair")
    f.add_argument("oracle_dir", type=Path)
    f.add_argument("arm_a")
    f.add_argument("arm_b")
    f.add_argument("--out", type=Path)
    m = sub.add_parser("make-fixture")
    m.add_argument("oracle_dir", type=Path)
    m.add_argument("out_dir", type=Path)
    c = sub.add_parser("check-fixture")
    c.add_argument("fixture_dir", type=Path)
    c.add_argument("--tier", type=int)
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "make-fixture":
        print(json.dumps(cmd_make_fixture(args.oracle_dir, args.out_dir)))
        return 0
    if args.cmd == "fortran-pair":
        report = cmd_fortran_pair(args.oracle_dir, args.arm_a, args.arm_b)
        if args.out:
            args.out.write_text(json.dumps(report, indent=1) + "\n")
        return 0
    if getattr(args, "tier", None):
        force_tier(args.tier)
    if args.cmd == "grade":
        report = cmd_grade(args.oracle_dir, args.arm)
        bad = report["total_bit_mismatch"]
    else:
        report = check_fixture(args.fixture_dir)
        bad = report["total_mismatch"]
    report["tier"] = args.tier
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"total_mismatch": bad, "strict_build": report["strict_build"],
                      "tier": args.tier,
                      "summary": {tag: {
                          "digest_mismatch": len(r.get("step_digest_mismatch", [])),
                          "bit_mismatch": r.get("bit_mismatch"),
                          "final_step": {n: [f["bit_mismatch"], f["max_ulp"]]
                                         for n, f in r.get("final_step", r.get("fields", {})).items()
                                         if f["bit_mismatch"]},
                          "columns_differing_from_stock_wrf":
                              r.get("columns_differing_from_stock_wrf")}
                          for tag, r in report["runs"].items()}}))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
