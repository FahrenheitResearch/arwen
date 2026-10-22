"""Run gpuwm's CUDA WSM5 over the oracle column sets and grade it word for word.

Reads the SAME ``main.bin``/``lev2.bin`` the WRF Fortran read
(make_columns.py), runs the production launchers
(``gpuwm.core.microphysics.launch_column_scheme`` and
``launch_column_refl`` on the mp_physics = 4 table row) step by step exactly
as WRF's wrapper runs (the scheme, then REFL_10CM and the radiation
effective radii from th*pii after the step), and compares every output word.

Effective radii: WRF stores metres; gpuwm's radiation contract is microns,
formed as one float32 multiply ``re * 1.0e6``.  WRF's words are put through
that multiply before comparison.

Subcommands::

    python run_gpu.py grade ORACLE_DIR [--out JSON]
        every word of every step against the defined arm (WOOF's declared
        arithmetic: must be zero differing words) and against the stock
        scalar WRF build (differing columns must be exactly the probe
        columns of the two declared fixes)
    python run_gpu.py make-fixture ORACLE_DIR OUT_DIR       (CPU only)
        the committed fixture: the column sets, a SHA-256 of every output
        field at every step for the defined and the stock arm, both arms'
        final-step words, the probe columns, the transport probe
    python run_gpu.py check-fixture FIXTURE_DIR [--tier N] [--out JSON]
        grade against the committed fixture (what the pytest gate runs)
    python run_gpu.py plant FIXTURE_DIR
        flip the last bit of one float32 literal of kernels/wsm5.cu in this
        process only, and require check-fixture to fail
    python run_gpu.py remap-probe [--out JSON]
        lane 073's transport-only control through the production device
        remap at 2, 50 and 80 levels

GPU work.  On a shared box run only through its card queue.
GPUWM_WRF_EXACT=1 selects the strict build; unset is the default build.
WSM5 compiles unflushed and uncontracted in both (load_module_unflushed),
so the two are expected to agree; every report states which ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.wsm6_wrf461_column_oracle.common import (
    _header)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import oracle_io as io  # type: ignore
else:
    from . import oracle_io as io


def run_gpu(columns: Path, dt: float, nsteps: int):
    """The production launchers, step by step, as WRF's wrapper runs."""
    import cupy as cp
    from gpuwm.core.microphysics import (COLUMN_SCHEMES, launch_column_refl,
                                         launch_column_scheme)

    row = COLUMN_SCHEMES[4]
    ncol, nz, level, surf = io.read_set(columns)
    f32 = np.float32
    dev = {n: cp.asarray(v.reshape(nz, 1, ncol), dtype=cp.float32)
           for n, v in level.items()}
    s = {n: cp.zeros((1, ncol), dtype=cp.float32) for n in row.surface}
    s["mp_rainnc"][...] = cp.asarray(surf["rain"].reshape(1, ncol))
    s["mp_snownc"][...] = cp.asarray(surf["snow"].reshape(1, ncol))
    effc = cp.full((nz, 1, ncol), f32(2.49), cp.float32)
    effi = cp.full((nz, 1, ncol), f32(4.99), cp.float32)
    effs = cp.full((nz, 1, ncol), f32(9.99), cp.float32)
    refl = cp.empty((nz, 1, ncol), cp.float32)
    t = cp.empty((nz, 1, ncol), cp.float32)
    steps = []
    for _ in range(nsteps):
        launch_column_scheme(
            row, dev["th"], [dev[n] for n in row.species], dev["den"], dev["p"],
            dev["pii"], dev["dz"], [s[n] for n in row.surface], effc, effi, effs,
            dt, conservative_surface=False)  # WRF oracle: exact (WB5 off)
        cp.multiply(dev["th"], dev["pii"], out=t)
        launch_column_refl(row, [dev[n] for n in row.refl_species], t, dev["p"], refl)
        cp.cuda.runtime.deviceSynchronize()
        g = lambda a: a.get().reshape(nz, ncol)
        lv = {"th": g(dev["th"]), "qv": g(dev["qv"]), "qc": g(dev["qc"]),
              "qi": g(dev["qi"]), "qr": g(dev["qr"]), "qs": g(dev["qs"]),
              "re_cloud": g(effc), "re_ice": g(effi), "re_snow": g(effs),
              "refl_10cm": g(refl), "t": g(t)}
        sf = {"rain": s["mp_rainnc"], "rainncv": s["mp_rainncv"],
              "snow": s["mp_snownc"], "snowncv": s["mp_snowncv"], "sr": s["mp_sr"]}
        sf = {k: v.get().reshape(ncol) for k, v in sf.items()}
        steps.append((lv, sf))
    return {"steps": steps}


def _probe_columns(run, names):
    remap, psevp = run["probes"][-1]
    return ({names[i] for i in np.nonzero(remap)[0]},
            {names[i] for i in np.nonzero(psevp)[0]})


def cmd_grade(oracle_dir: Path):
    report = {**_header(), "runs": {}}
    total = 0
    for set_name, dt, nsteps in io.RUNS:
        tag = io.tag_of(set_name, dt, nsteps)
        names = io.column_names(oracle_dir, set_name)
        gpu = run_gpu(oracle_dir / f"{set_name}.bin", float(dt), nsteps)
        defined = io.read_wrf(oracle_dir / f"wrf-defined-{tag}.bin")
        scalar = io.read_wrf(oracle_dir / f"wrf-scalar-{tag}.bin")
        vs_defined = io.compare_runs(defined, gpu, names)
        vs_stock = io.compare_runs(scalar, gpu, names)
        remap_cols, psevp_cols = _probe_columns(defined, names)
        stock_cols = io.mismatch_columns(vs_stock)
        bad = sum(r["bit_mismatch"] for r in vs_defined.values())
        mask_ok = stock_cols == (remap_cols | psevp_cols)
        total += bad + (0 if mask_ok else 1)
        report["runs"][tag] = {
            "vs_defined": vs_defined,
            "vs_stock": {"bit_mismatch": sum(r["bit_mismatch"] for r in vs_stock.values()),
                         "columns": sorted(stock_cols),
                         "remap_probe_columns": sorted(remap_cols),
                         "psevp_probe_columns": sorted(psevp_cols),
                         "mask_equals_probe_union": mask_ok,
                         "fields": {n: [r["bit_mismatch"], r["max_ulp"]]
                                    for n, r in vs_stock.items() if r["bit_mismatch"]}}}
        print(f"{tag}: vs defined {bad} differing words"
              + "".join(f"; {n} {r['bit_mismatch']}/{r['max_ulp']}u"
                        for n, r in vs_defined.items() if r["bit_mismatch"])
              + f" | vs stock {report['runs'][tag]['vs_stock']['bit_mismatch']} words in "
              f"{len(stock_cols)} columns, mask==probes {mask_ok}", flush=True)
    report["total_mismatch"] = total
    return report


def cmd_make_fixture(oracle_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("main.bin", "main.json", "lev2.bin", "lev2.json",
                 "wsm5-init-f32.txt", "libmvec-report.txt", "compiler.txt",
                 "compile-flags.txt"):
        shutil.copyfile(oracle_dir / name, out / name)
    digests = {"inputs_sha256": {n: hashlib.sha256((oracle_dir / n).read_bytes()).hexdigest()
                                 for n in ("main.bin", "lev2.bin")},
               "runs": {}}
    final = {}
    probes = {}
    for set_name, dt, nsteps in io.RUNS:
        tag = io.tag_of(set_name, dt, nsteps)
        names = io.column_names(oracle_dir, set_name)
        digests["runs"][tag] = {}
        for arm in ("defined", "scalar"):
            run = io.read_wrf(oracle_dir / f"wrf-{arm}-{tag}.bin")
            digests["runs"][tag][arm] = [
                {**{n: io.digest(lv[n]) for n in io.LEVEL_OUT},
                 **{n: io.digest(sf[n]) for n in io.SURFACE_OUT}}
                for lv, sf in run["steps"]]
            lv, sf = run["steps"][-1]
            for n in io.LEVEL_OUT:
                final[f"{arm}/{tag}/{n}"] = lv[n]
            for n in io.SURFACE_OUT:
                final[f"{arm}/{tag}/{n}"] = sf[n]
            if arm == "defined":
                remap_cols, psevp_cols = _probe_columns(run, names)
                probes[tag] = {"remap": sorted(remap_cols), "psevp": sorted(psevp_cols)}
    (out / "wrf461-step-digests.json").write_text(json.dumps(digests, indent=1) + "\n")
    (out / "probe-columns.json").write_text(json.dumps(probes, indent=1) + "\n")
    np.savez_compressed(out / "wrf461-final-step.npz", **final)
    for name in ("arms.json", "water-budget.json"):
        if (oracle_dir / name).exists():
            text = json.loads((oracle_dir / name).read_text())
            if name == "water-budget.json":
                for row in text.values():
                    for arm in ("scalar", "defined"):
                        row[arm].pop("per_column", None)
            (out / name).write_text(json.dumps(text, indent=1) + "\n")
    return {"written": sorted(p.name for p in out.iterdir())}


def force_tier(tier: int) -> None:
    """Compile the WSM5 kernels at WSM5_KMAX = ``tier`` whatever nz is."""
    from gpuwm.core import microphysics

    row = microphysics.COLUMN_SCHEMES[4]
    import dataclasses
    microphysics.COLUMN_SCHEMES[4] = dataclasses.replace(
        row, level_tier=lambda nz, _tier=int(tier): _tier)


def check_fixture(fixture: Path):
    """Grade the CUDA kernels against the committed fixture.

    Every field of every step against the defined arm by digest, the final
    step word by word; and against the stock arm, the set of differing
    columns at the final step must be exactly the union of the two probes.
    """
    from gpuwm.core.microphysics import COLUMN_SCHEMES

    digests = json.loads((fixture / "wrf461-step-digests.json").read_text())
    probes = json.loads((fixture / "probe-columns.json").read_text())
    final = np.load(fixture / "wrf461-final-step.npz")
    report = {**_header(), "tier": COLUMN_SCHEMES[4].level_tier(45), "runs": {}}
    total = 0
    for set_name, dt, nsteps in io.RUNS:
        tag = io.tag_of(set_name, dt, nsteps)
        names = io.column_names(fixture, set_name)
        gpu = run_gpu(fixture / f"{set_name}.bin", float(dt), nsteps)
        step_digest_mismatch = []
        for istep, ((lv, sf), want) in enumerate(zip(gpu["steps"],
                                                     digests["runs"][tag]["defined"])):
            got = {**{n: io.digest(lv[n]) for n in io.LEVEL_OUT},
                   **{n: io.digest(sf[n]) for n in io.SURFACE_OUT}}
            step_digest_mismatch += [f"step {istep + 1} {n}" for n in got
                                     if got[n] != want[n]]
        last = {"steps": [gpu["steps"][-1]]}
        arm = lambda a: {"steps": [({n: final[f"{a}/{tag}/{n}"] for n in io.LEVEL_OUT},
                                    {n: final[f"{a}/{tag}/{n}"] for n in io.SURFACE_OUT})]}
        vs_defined = io.compare_runs(arm("defined"), last, names)
        vs_stock = io.compare_runs(arm("scalar"), last, names)
        want_cols = set(probes[tag]["remap"]) | set(probes[tag]["psevp"])
        stock_cols = io.mismatch_columns(vs_stock)
        bad = (sum(r["bit_mismatch"] for r in vs_defined.values())
               + len(step_digest_mismatch) + (0 if stock_cols == want_cols else 1))
        total += bad
        report["runs"][tag] = {
            "step_digest_mismatch": step_digest_mismatch,
            "final_step_vs_defined": {n: [r["bit_mismatch"], r["max_ulp"]]
                                      for n, r in vs_defined.items() if r["bit_mismatch"]},
            "stock_mismatch_columns": sorted(stock_cols),
            "probe_columns": sorted(want_cols),
            "mask_equals_probe_union": stock_cols == want_cols}
    report["total_mismatch"] = total
    return report


def plant_literal() -> tuple[str, str]:
    """The planted change: the cloud-radius micron multiplier, one ULP up."""
    old = "1.0e6f"
    bumped = np.nextafter(np.float32(1.0e6), np.float32(float("inf")))
    new = f"{float(bumped):.9e}f"
    assert np.float32(float(new[:-1])) == bumped
    return old, new


def cmd_plant(fixture: Path):
    import gpuwm.core.kernels as kernels

    old, new = plant_literal()
    original = kernels._unit_text

    def planted(name, kernel_dir=kernels._KDIR):
        text = original(name, kernel_dir)
        if name == "wsm5":
            anchor = "effc[id] = re_qc * " + old
            assert text.count(anchor) == 1, text.count(anchor)
            text = text.replace(anchor, "effc[id] = re_qc * " + new)
        return text

    kernels._unit_text = planted
    report = check_fixture(fixture)
    report["planted"] = {"from": old, "to": new}
    return report


PROBE_SOURCE = r"""
extern "C" __global__ void wsm5_remap_probe(
    int km, const float *den, const float *dz, const float *ww, float *rq,
    float *precip, float dt, const float *coef)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
    wsm5_nislfv(km, coef, den, dz, ww, ww, rq, precip, dt, 0, 2, 1);
}
"""


def cmd_remap_probe():
    """Lane 073's control through the production device remap (iter = 0)."""
    import cupy as cp
    from gpuwm import nvrtc_ptx_cache
    from gpuwm.core.kernels import UNFLUSHED_OPTIONS, module_source_int_defines
    from gpuwm.core.wsm6_constants import wsm5_init_words

    source = module_source_int_defines("wsm5", (("WSM5_KMAX", 80),)) + PROBE_SOURCE
    image, _ = nvrtc_ptx_cache.compile_using_nvrtc(
        source, UNFLUSHED_OPTIONS, None, "wsm5_remap_probe.cu")
    module = cp.cuda.function.Module()
    module.load(image.encode() if isinstance(image, str) else image)
    kernel = module.get_function("wsm5_remap_probe")
    coef = cp.asarray(wsm5_init_words())
    rows = []
    for nz in (2, 50, 80):
        for dt in (0.001, 1.0, 20.0):
            den = cp.ones(nz, cp.float32)
            dz = cp.full(nz, 100.0, cp.float32)
            ww = cp.ones(nz, cp.float32)
            rq = cp.full(nz, 0.001, cp.float32)
            precip = cp.zeros(1, cp.float32)
            kernel((1,), (1,), (np.int32(nz), den, dz, ww, rq, precip,
                                np.float32(dt), coef))
            rows.append((nz, float(np.float32(dt)), rq.get(), float(precip.get()[0])))
    if __package__ in (None, ""):
        from compare_arms import budget  # type: ignore
    else:
        from .compare_arms import budget
    out = budget(rows)
    for r in out:
        print(f"nz{r['nz']} dt{r['dt']:g}: lost {r['lost_kg_m2']:.3e} kg m-2, "
              f"relative {r['relative_error']:.2e}", flush=True)
    return {**_header(), "rows": out,
            "words": {f"nz{nz}-dt{dt:g}": [int(w) for w in rq.view(np.uint32)] + [
                int(np.float32(p).view(np.uint32))]
                for nz, dt, rq, p in rows},
            "total_mismatch": sum(not np.isfinite(r["relative_error"]) or r["relative_error"] >= 1.0e-6 for r in out)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade")
    g.add_argument("oracle_dir", type=Path)
    g.add_argument("--out", type=Path)
    m = sub.add_parser("make-fixture")
    m.add_argument("oracle_dir", type=Path)
    m.add_argument("out_dir", type=Path)
    c = sub.add_parser("check-fixture")
    c.add_argument("fixture_dir", type=Path)
    c.add_argument("--tier", type=int)
    c.add_argument("--out", type=Path)
    pl = sub.add_parser("plant")
    pl.add_argument("fixture_dir", type=Path)
    pl.add_argument("--out", type=Path)
    rp = sub.add_parser("remap-probe")
    rp.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "make-fixture":
        print(json.dumps(cmd_make_fixture(args.oracle_dir, args.out_dir)))
        return 0
    if args.cmd == "grade":
        report = cmd_grade(args.oracle_dir)
    elif args.cmd == "check-fixture":
        if args.tier:
            force_tier(args.tier)
        report = check_fixture(args.fixture_dir)
    elif args.cmd == "plant":
        report = cmd_plant(args.fixture_dir)
    else:
        report = cmd_remap_probe()
    bad = report["total_mismatch"]
    text = json.dumps(report, indent=1)
    if getattr(args, "out", None):
        args.out.write_text(text + "\n")
    summary = {"total_mismatch": bad, "strict_build": report["strict_build"]}
    if "runs" in report:
        summary["runs"] = {tag: {k: v for k, v in r.items()
                                 if k in ("step_digest_mismatch", "final_step_vs_defined",
                                          "mask_equals_probe_union")}
                           for tag, r in report["runs"].items()}
    if "planted" in report:
        summary["planted"] = report["planted"]
    print(json.dumps(summary))
    if args.cmd == "plant":
        return 0 if bad > 0 else 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
