"""Run gpuwm's CUDA WSM6 over the oracle column set and grade it bit for bit.

Reads the SAME ``columns.bin`` the WRF Fortran read (make_columns.py), runs
the production launchers (``gpuwm.core.wsm6.launch_wsm6`` and
``gpuwm.core.refl.launch_refl10cm_wsm6``) step by step exactly as WRF's
wrapper runs (microphysics, then the radiation effective radii, then
REFL_10CM from th*pii after the step), and compares every output word.

Effective radii: WRF stores metres; gpuwm's radiation contract is microns,
formed as one float32 multiply ``re * 1.0e6`` -- the multiply WRF's RRTMG
does when it reads re_cloud/re_ice/re_snow.  WRF's words are put through
that multiply before comparison.

Three subcommands::

    python run_gpu.py grade ORACLE_DIR [--flavour O0|O2vec] [--out JSON]
        every word of every step against run_wsm6_columns.F90's output
        (needs the Fortran build, i.e. the box that ran build.sh)
    python run_gpu.py make-fixture ORACLE_DIR OUT_DIR       (CPU only)
        the committed fixture: the column set, a SHA-256 of every WRF
        output field at every step, and the final step's words
    python run_gpu.py check-fixture FIXTURE_DIR [--out JSON] [--tier N]
        grade against the committed fixture (what the pytest gate runs)

GPU work.  On a shared box run only through its GPU mutex.
GPUWM_WRF_EXACT=1 selects the strict build (no FMA, no FTZ); unset is the
default build.  Every report states which.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.wsm6_wrf461_column_oracle.common import (
    _header, micron, _ordered, ulp, digest, read_wrf as _read_wrf, read_columns as _read_columns, accumulate)

LEVEL_OUT = ("th", "qv", "qc", "qi", "qr", "qs", "qg",
             "re_cloud", "re_ice", "re_snow", "refl_10cm", "t")
SURFACE_OUT = ("rain", "rainncv", "snow", "snowncv", "graupel",
               "graupelncv", "sr")
#: (hail_opt, dt, nsteps): dt 30 s is one minor loop; 150 s one loop with
#: dtcld > dtcldcr; 300 s nint(2.5) = 3 loops of 100 s.
RUNS = ((0, 30.0, 12), (1, 30.0, 12), (0, 150.0, 3), (1, 150.0, 3),
        (0, 300.0, 4), (1, 300.0, 4))


def tag_of(hail: int, dt: float, nsteps: int) -> str:
    return f"h{hail}-dt{int(dt)}-n{nsteps}"


def read_columns(path: Path):
    return _read_columns(path, ("th", "qv", "qc", "qr", "qi", "qs", "qg", "den", "pii", "p", "dz"),
                         ("rain", "snow", "graupel"))


def read_wrf(path: Path):
    return _read_wrf(path, LEVEL_OUT, SURFACE_OUT, ("hail",))


def run_gpu(columns: Path, hail: int, dt: float, nsteps: int):
    """The production launchers, step by step, as WRF's wrapper runs."""
    import cupy as cp
    from gpuwm.core.refl import launch_refl10cm_wsm6
    from gpuwm.core.wsm6 import launch_wsm6

    ncol, nz, level, surf = read_columns(columns)
    f32 = np.float32
    dev = {n: cp.asarray(v.reshape(nz, 1, ncol), dtype=cp.float32)
           for n, v in level.items()}
    s = {n: cp.zeros((1, ncol), dtype=cp.float32) for n in
         ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")}
    s["rainnc"][...] = cp.asarray(surf["rain"].reshape(1, ncol))
    s["snownc"][...] = cp.asarray(surf["snow"].reshape(1, ncol))
    s["graupelnc"][...] = cp.asarray(surf["graupel"].reshape(1, ncol))
    effc = cp.full((nz, 1, ncol), f32(2.49), cp.float32)
    effi = cp.full((nz, 1, ncol), f32(4.99), cp.float32)
    effs = cp.full((nz, 1, ncol), f32(9.99), cp.float32)
    refl = cp.empty((nz, 1, ncol), cp.float32)
    t = cp.empty((nz, 1, ncol), cp.float32)
    steps = []
    for _ in range(nsteps):
        launch_wsm6(dev["th"], dev["qv"], dev["qc"], dev["qr"], dev["qi"],
                    dev["qs"], dev["qg"], dev["den"], dev["pii"], dev["p"],
                    dev["dz"], s["rainnc"], s["rainncv"], s["snownc"],
                    s["snowncv"], s["graupelnc"], s["graupelncv"], s["sr"],
                    dt, effc=effc, effi=effi, effs=effs, hail_opt=hail)
        cp.multiply(dev["th"], dev["pii"], out=t)
        launch_refl10cm_wsm6(dev["qv"], dev["qr"], dev["qs"], dev["qg"], t,
                             dev["p"], refl, hail_opt=hail, melting=True)
        cp.cuda.runtime.deviceSynchronize()
        g = lambda a: a.get().reshape(nz, ncol)
        lv = {"th": g(dev["th"]), "qv": g(dev["qv"]), "qc": g(dev["qc"]),
              "qi": g(dev["qi"]), "qr": g(dev["qr"]), "qs": g(dev["qs"]),
              "qg": g(dev["qg"]), "re_cloud": g(effc), "re_ice": g(effi),
              "re_snow": g(effs), "refl_10cm": g(refl), "t": g(t)}
        sf = {"rain": s["rainnc"], "rainncv": s["rainncv"], "snow": s["snownc"],
              "snowncv": s["snowncv"], "graupel": s["graupelnc"],
              "graupelncv": s["graupelncv"], "sr": s["sr"]}
        sf = {k: v.get().reshape(ncol) for k, v in sf.items()}
        steps.append((lv, sf))
    return steps


def _pairs(wrf_step, gpu_step):
    (wl, ws), (gl, gs) = wrf_step, gpu_step
    out = {n: (wl[n], gl[n]) for n in LEVEL_OUT}
    out.update({n: (ws[n][None, :], gs[n][None, :]) for n in SURFACE_OUT})
    return out


def _accumulate(fields, istep, pairs, regimes, keep=None):
    return accumulate(fields, istep, pairs, [r["name"] for r in regimes], keep,
                      allow_empty=False, worst_labels=("wrf", "gpu"))


def grade(wrf, gpu, regimes):
    fields = {}
    for istep, (ws, gs) in enumerate(zip(wrf["steps"], gpu)):
        _accumulate(fields, istep, _pairs(ws, gs), regimes)
    return fields


def cmd_grade(oracle_dir: Path, flavour: str):
    regimes = json.loads((oracle_dir / "columns.json").read_text())["columns"]
    report = {**_header(), "flavour": flavour, "runs": {}}
    total = 0
    for hail, dt, nsteps in RUNS:
        tag = tag_of(hail, dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-{flavour}-{tag}.bin")
        fields = grade(wrf, run_gpu(oracle_dir / "columns.bin", hail, dt, nsteps), regimes)
        report["runs"][tag] = fields
        bad = sum(r["bit_mismatch"] for r in fields.values())
        total += bad
        print(f"{tag}: {bad} differing words; " + ", ".join(
            f"{n} {r['bit_mismatch']}/{r['max_ulp']}u" for n, r in fields.items()
            if r["bit_mismatch"]), flush=True)
    report["total_bit_mismatch"] = total
    return report


def cmd_make_fixture(oracle_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("columns.bin", "columns.json", "wsm6-init-f32.txt",
                 "libmvec-report.txt", "compiler.txt", "oracle-sha256sums.txt"):
        shutil.copyfile(oracle_dir / name, out / name)
    digests = {"columns_bin_sha256": hashlib.sha256(
        (oracle_dir / "columns.bin").read_bytes()).hexdigest(), "runs": {}}
    final = {}
    for hail, dt, nsteps in RUNS:
        tag = tag_of(hail, dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-O0-{tag}.bin")
        digests["runs"][tag] = [
            {**{n: digest(lv[n]) for n in LEVEL_OUT},
             **{n: digest(sf[n]) for n in SURFACE_OUT}}
            for lv, sf in wrf["steps"]]
        lv, sf = wrf["steps"][-1]
        for n in LEVEL_OUT:
            final[f"{tag}/{n}"] = lv[n]
        for n in SURFACE_OUT:
            final[f"{tag}/{n}"] = sf[n]
    (out / "wrf461-step-digests.json").write_text(json.dumps(digests, indent=1) + "\n")
    np.savez_compressed(out / "wrf461-final-step.npz", **final)
    return {"written": sorted(p.name for p in out.iterdir())}


def force_tier(tier: int) -> None:
    """Compile the WSM6 kernels at WSM6_KMAX = ``tier`` whatever nz is."""
    import gpuwm.core.wsm6 as wsm6

    wsm6._kernel_capacity = lambda nz, _tier=int(tier): _tier


def check_fixture(fixture: Path, *, exclude_columns=(), runs=RUNS):
    """Grade the CUDA kernels against the committed fixture.

    With no exclusions every field of every step is checked by digest and
    the final step word by word.  ``exclude_columns`` drops named columns
    from the word-by-word final-step check (the digests cover whole fields,
    so they are skipped then).
    """
    regimes = json.loads((fixture / "columns.json").read_text())["columns"]
    digests = json.loads((fixture / "wrf461-step-digests.json").read_text())
    final = np.load(fixture / "wrf461-final-step.npz")
    keep = np.asarray([r["name"] not in exclude_columns for r in regimes])
    report = {**_header(), "excluded_columns": list(exclude_columns), "runs": {}}
    total = 0
    for hail, dt, nsteps in runs:
        tag = tag_of(hail, dt, nsteps)
        gpu = run_gpu(fixture / "columns.bin", hail, dt, nsteps)
        step_digest_mismatch = []
        if not exclude_columns:
            for istep, ((lv, sf), want) in enumerate(zip(gpu, digests["runs"][tag])):
                got = {**{n: digest(lv[n]) for n in LEVEL_OUT},
                       **{n: digest(sf[n]) for n in SURFACE_OUT}}
                step_digest_mismatch += [f"step {istep + 1} {n}" for n in got
                                         if got[n] != want[n]]
        lv, sf = gpu[-1]
        wrf_last = ({n: final[f"{tag}/{n}"] for n in LEVEL_OUT},
                    {n: final[f"{tag}/{n}"] for n in SURFACE_OUT})
        fields = {}
        _accumulate(fields, nsteps - 1, _pairs(wrf_last, (lv, sf)), regimes,
                    keep=None if not exclude_columns else keep)
        bad = sum(r["bit_mismatch"] for r in fields.values()) + len(step_digest_mismatch)
        total += bad
        report["runs"][tag] = {"final_step": fields,
                               "step_digest_mismatch": step_digest_mismatch}
    report["total_mismatch"] = total
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade")
    g.add_argument("oracle_dir", type=Path)
    g.add_argument("--flavour", default="O0")
    g.add_argument("--out", type=Path)
    m = sub.add_parser("make-fixture")
    m.add_argument("oracle_dir", type=Path)
    m.add_argument("out_dir", type=Path)
    c = sub.add_parser("check-fixture")
    c.add_argument("fixture_dir", type=Path)
    c.add_argument("--exclude", action="append", default=[])
    c.add_argument("--tier", type=int)
    c.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "make-fixture":
        print(json.dumps(cmd_make_fixture(args.oracle_dir, args.out_dir)))
        return 0
    if args.cmd == "grade":
        report = cmd_grade(args.oracle_dir, args.flavour)
        bad = report["total_bit_mismatch"]
    else:
        if args.tier:
            force_tier(args.tier)
        report = check_fixture(args.fixture_dir, exclude_columns=tuple(args.exclude))
        report["tier"] = args.tier
        bad = report["total_mismatch"]
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"total_mismatch": bad, "strict_build": report["strict_build"],
                      "summary": {tag: {
                          "digest_mismatch": len(r.get("step_digest_mismatch", [])),
                          "final_step": {n: [f["bit_mismatch"], f["max_ulp"]]
                                         for n, f in r.get("final_step", r).items()
                                         if f["bit_mismatch"]}}
                          for tag, r in report["runs"].items()}}))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
