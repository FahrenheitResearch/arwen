"""Run gpuwm's CUDA WSM6R over the oracle column set and grade it bit for bit.

Reads the SAME ``columns.bin`` WRF's Fortran read (make_columns.py), runs the
production launcher ``gpuwm.core.wsm6.launch_wsm6r`` step by step exactly as
WRF's wrapper runs, and compares every output word: th, qv, qc, qi, qr, qs,
qg, t (= th*pii), RAIN and RAINNCV.

RAINNCV on multi-loop runs.  WRF's wsm6r reports the LAST minor loop's
fallout in RAINNCV; gpuwm reports the step's (a declared divergence, see
kernels/wsm6r.cu).  On runs whose dt makes more than one minor loop
(nint(dt/120) > 1) RAINNCV is therefore not graded word for word; it is
graded against the step's own RAIN increment instead, to within the
rounding of the float32 accumulation (``rainncv_step_check``).

Subcommands::

    python run_gpu.py grade ORACLE_DIR [--flavour O0|O2vec] [--out JSON]
        every word of every step against run_wsm6r_columns.F90's output
    python run_gpu.py make-fixture ORACLE_DIR OUT_DIR       (CPU only)
        the committed fixture: the column set, a SHA-256 of every WRF
        output field at every step, and the final step's words
    python run_gpu.py check-fixture FIXTURE_DIR [--exclude NAME]... [--out JSON]
        grade against the committed fixture (what the pytest gate runs)
    python run_gpu.py plant FIXTURE_DIR
        the mutation control: a one-bit change planted in a revised WSM6R
        term must be caught

NaN sign and payload are reported as a declared divergence when WRF itself
becomes nonfinite. Raw hashes are retained. The gate canonicalizes only NaN
words; all finite words and nonfinite masks must still match.

GPU work. On a shared box run only through its GPU mutex.
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
    _header, micron, _ordered, ulp, read_wrf as _read_wrf, read_columns as _read_columns)

LEVEL_OUT = ("th", "qv", "qc", "qi", "qr", "qs", "qg", "t",
             "re_cloud", "re_ice", "re_snow")
RADII = ("re_cloud", "re_ice", "re_snow")
SURFACE_OUT = ("rain", "rainncv")
#: (dt, nsteps); build.sh's list.
RUNS = ((30.0, 12), (60.0, 6), (150.0, 3), (240.0, 3), (300.0, 4))


def tag_of(dt: float, nsteps: int) -> str:
    return f"dt{int(dt)}-n{nsteps}"


def minor_loops(dt: float) -> int:
    """wsm62D's ``max(nint(delt/dtcldcr), 1)`` (half away from zero)."""
    x = np.float32(dt) / np.float32(120.0)
    return max(int(np.floor(abs(float(x)) + 0.5)) * (1 if x >= 0 else -1), 1)


def read_columns(path: Path):
    return _read_columns(path, ("th", "qv", "qc", "qr", "qi", "qs", "qg", "den", "pii", "p", "dz"),
                         ("rain",))


def read_wrf(path: Path):
    return _read_wrf(path, LEVEL_OUT, SURFACE_OUT)


def run_gpu(columns: Path, dt: float, nsteps: int):
    """The production launcher, step by step, as WRF's wrapper runs."""
    import cupy as cp
    from gpuwm.core.wsm6 import launch_wsm6r, launch_wsm6r_effective_radii

    ncol, nz, level, surf = read_columns(columns)
    dev = {n: cp.asarray(v.reshape(nz, 1, ncol), dtype=cp.float32)
           for n, v in level.items()}
    rain = cp.asarray(surf["rain"].reshape(1, ncol), dtype=cp.float32)
    rainncv = cp.zeros((1, ncol), cp.float32)
    work = cp.zeros((3, nz, 1, ncol), cp.float32)
    t = cp.empty((nz, 1, ncol), cp.float32)
    eff = {n: cp.empty((nz, 1, ncol), cp.float32) for n in RADII}
    steps = []
    for _ in range(nsteps):
        launch_wsm6r(dev["th"], dev["qv"], dev["qc"], dev["qr"], dev["qi"],
                     dev["qs"], dev["qg"], dev["den"], dev["pii"], dev["p"],
                     dev["dz"], rain, rainncv, dt, work=work)
        launch_wsm6r_effective_radii(
            dev["th"], dev["pii"], dev["qc"], dev["qi"], dev["qs"], dev["den"],
            eff["re_cloud"], eff["re_ice"], eff["re_snow"])
        cp.multiply(dev["th"], dev["pii"], out=t)
        cp.cuda.runtime.deviceSynchronize()
        g = lambda a: a.get().reshape(nz, ncol)
        lv = {"th": g(dev["th"]), "qv": g(dev["qv"]), "qc": g(dev["qc"]),
              "qi": g(dev["qi"]), "qr": g(dev["qr"]), "qs": g(dev["qs"]),
              "qg": g(dev["qg"]), "t": g(t),
              **{n: g(eff[n]) for n in RADII}}
        sf = {"rain": rain.get().reshape(ncol), "rainncv": rainncv.get().reshape(ncol)}
        steps.append((lv, sf))
    return steps


def digest(a, *, canonical_nan=False) -> str:
    words = np.ascontiguousarray(a, "<f4").view(np.uint32).copy()
    if canonical_nan:
        words[np.isnan(words.view(np.float32))] = np.uint32(0x7FC00000)
    return hashlib.sha256(words.tobytes()).hexdigest()


def graded_fields(dt: float):
    """Fields graded word for word on a run with time step ``dt``."""
    if minor_loops(dt) > 1:
        return LEVEL_OUT + ("rain",)
    return LEVEL_OUT + SURFACE_OUT


def _pairs(wrf_step, gpu_step, fields):
    (wl, ws), (gl, gs) = wrf_step, gpu_step
    out = {}
    for n in fields:
        if n in LEVEL_OUT:
            out[n] = (wl[n], gl[n])
        else:
            out[n] = (ws[n][None, :], gs[n][None, :])
    return out


def _accumulate(fields, istep, pairs, regimes, keep=None):
    for n, (w, g) in pairs.items():
        w = np.asarray(w, np.float32)
        g = np.asarray(g, np.float32)
        if keep is not None:
            w, g = w[:, keep], g[:, keep]
            names = [regimes[i]["name"] for i in np.nonzero(keep)[0]]
        else:
            names = [r["name"] for r in regimes]
        diff_bits = w.view(np.uint32) != g.view(np.uint32)
        nan_payload = diff_bits & np.isnan(w) & np.isnan(g)
        failed_bits = diff_bits & ~nan_payload
        u = np.where(np.isnan(w) & np.isnan(g), 0, ulp(w, g))
        gap = np.abs(w.astype(np.float64) - g.astype(np.float64))
        gap = np.where(np.isnan(gap), 0.0, gap)
        rel = gap / np.maximum(np.abs(w.astype(np.float64)), 1.0e-30)
        row = fields.setdefault(n, {"words": 0, "bit_mismatch": 0, "max_ulp": 0,
                                    "max_abs": 0.0, "max_rel": 0.0, "worst": None,
                                    "nan_payload_mismatch": 0, "failed_words": 0,
                                    "columns": {}, "first_step_with_a_mismatch": None})
        row["words"] += int(w.size)
        row["bit_mismatch"] += int(diff_bits.sum())
        row["nan_payload_mismatch"] += int(nan_payload.sum())
        row["failed_words"] += int(failed_bits.sum())
        row["max_abs"] = max(row["max_abs"], float(gap.max()))
        row["max_rel"] = max(row["max_rel"], float(np.where(diff_bits, rel, 0.0).max()))
        if diff_bits.any() and row["first_step_with_a_mismatch"] is None:
            row["first_step_with_a_mismatch"] = istep + 1
        for i in np.nonzero(diff_bits.any(axis=0))[0]:
            row["columns"][names[i]] = row["columns"].get(names[i], 0) + int(diff_bits[:, i].sum())
        m = int(u.max())
        if diff_bits.any() and (row["worst"] is None or m > row["max_ulp"]):
            k, i = np.unravel_index(int(np.argmax(u)), u.shape)
            row["max_ulp"] = max(row["max_ulp"], m)
            row["worst"] = {"step": istep + 1, "level": int(k), "column": names[i],
                            "wrf": float(w[k, i]), "gpu": float(g[k, i])}


def rainncv_step_check(rain_before, gpu_step, keep=None) -> dict:
    """gpuwm's per-step RAINNCV against the step's own RAIN increment.

    RAIN accumulates each loop's fallout word in float32, RAINNCV sums the
    same words from zero, so the two may differ by the rounding of the
    running RAIN total: at most one ULP of RAIN per minor loop.
    """
    _, sf = gpu_step
    rain = np.asarray(sf["rain"], np.float64)
    inc = rain - np.asarray(rain_before, np.float64)
    rcv = np.asarray(sf["rainncv"], np.float64)
    if keep is not None:
        rain, inc, rcv = rain[keep], inc[keep], rcv[keep]
    tol = 8.0 * np.spacing(np.maximum(np.abs(rain), 1.0e-30).astype(np.float32)).astype(np.float64)
    gap = np.abs(rcv - inc)
    return {"max_abs": float(gap.max()), "violations": int((gap > tol).sum()),
            "negative": int((rcv < 0).sum())}


def grade(wrf, gpu, regimes, dt, rain0):
    fields = {}
    checks = []
    before = rain0
    for istep, (ws, gs) in enumerate(zip(wrf["steps"], gpu)):
        _accumulate(fields, istep, _pairs(ws, gs, graded_fields(dt)), regimes)
        if minor_loops(dt) > 1:
            checks.append(rainncv_step_check(before, gs))
        before = gs[1]["rain"]
    out = {"fields": fields}
    if checks:
        out["rainncv_step_check"] = {
            "max_abs": max(c["max_abs"] for c in checks),
            "violations": sum(c["violations"] for c in checks),
            "negative": sum(c["negative"] for c in checks)}
    return out


def _bad(result) -> int:
    n = sum(r["failed_words"] for r in result["fields"].values())
    chk = result.get("rainncv_step_check")
    if chk:
        n += chk["violations"] + chk["negative"]
    return n


def cmd_grade(oracle_dir: Path, flavour: str):
    regimes = json.loads((oracle_dir / "columns.json").read_text())["columns"]
    _, _, _, surf = read_columns(oracle_dir / "columns.bin")
    report = {**_header(), "flavour": flavour, "runs": {}}
    total = 0
    for dt, nsteps in RUNS:
        tag = tag_of(dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-{flavour}-{tag}.bin")
        result = grade(wrf, run_gpu(oracle_dir / "columns.bin", dt, nsteps),
                       regimes, dt, surf["rain"])
        report["runs"][tag] = result
        bad = _bad(result)
        total += bad
        print(f"{tag}: {bad} differing words; " + ", ".join(
            f"{n} {r['bit_mismatch']}/{r['max_ulp']}u" for n, r in result["fields"].items()
            if r["bit_mismatch"]), flush=True)
    report["total_bit_mismatch"] = total
    return report


def cmd_make_fixture(oracle_dir: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name in ("columns.bin", "columns.json", "wsm6r-init-f32.txt",
                 "libmvec-report.txt", "compiler.txt", "oracle-sha256sums.txt"):
        shutil.copyfile(oracle_dir / name, out / name)
    digests = {"columns_bin_sha256": hashlib.sha256(
        (oracle_dir / "columns.bin").read_bytes()).hexdigest(), "runs": {}}
    final = {}
    for dt, nsteps in RUNS:
        tag = tag_of(dt, nsteps)
        wrf = read_wrf(oracle_dir / f"wrf-O0-{tag}.bin")
        digests["runs"][tag] = [
            {**{n: digest(lv[n]) for n in LEVEL_OUT},
             **{n: digest(sf[n]) for n in SURFACE_OUT}}
            for lv, sf in wrf["steps"]]
        digests.setdefault("canonical_nan_runs", {})[tag] = [
            {**{n: digest(lv[n], canonical_nan=True) for n in LEVEL_OUT},
             **{n: digest(sf[n], canonical_nan=True) for n in SURFACE_OUT}}
            for lv, sf in wrf["steps"]]
        lv, sf = wrf["steps"][-1]
        for n in LEVEL_OUT:
            final[f"{tag}/{n}"] = lv[n]
        for n in SURFACE_OUT:
            final[f"{tag}/{n}"] = sf[n]
            # the step before the last, for the RAINNCV increment check
            if len(wrf["steps"]) > 1:
                final[f"{tag}/{n}_prev"] = wrf["steps"][-2][1][n]
    (out / "wrf461-step-digests.json").write_text(json.dumps(digests, indent=1) + "\n")
    np.savez_compressed(out / "wrf461-final-step.npz", **final)
    return {"written": sorted(p.name for p in out.iterdir())}


def check_fixture(fixture: Path, *, exclude_columns=(), runs=RUNS):
    """Grade the CUDA kernel against the committed fixture.

    With no exclusions every graded field of every step is checked by
    digest and the final step word by word.  ``exclude_columns`` drops
    named columns from the word-by-word final-step check (the digests cover
    whole fields, so they are skipped then).
    """
    regimes = json.loads((fixture / "columns.json").read_text())["columns"]
    digests = json.loads((fixture / "wrf461-step-digests.json").read_text())
    final = np.load(fixture / "wrf461-final-step.npz")
    keep = np.asarray([r["name"] not in exclude_columns for r in regimes])
    report = {**_header(), "excluded_columns": list(exclude_columns), "runs": {}}
    total = 0
    for dt, nsteps in runs:
        tag = tag_of(dt, nsteps)
        graded = graded_fields(dt)
        gpu = run_gpu(fixture / "columns.bin", dt, nsteps)
        step_digest_mismatch = []
        raw_step_digest_mismatch = []
        if not exclude_columns:
            for istep, ((lv, sf), want) in enumerate(zip(gpu, digests["runs"][tag])):
                got = {**{n: digest(lv[n]) for n in LEVEL_OUT},
                       **{n: digest(sf[n]) for n in SURFACE_OUT}}
                raw_step_digest_mismatch += [
                    f"step {istep + 1} {n}" for n in graded if got[n] != want[n]]
                canon_want = digests["canonical_nan_runs"][tag][istep]
                canon_got = {
                    **{n: digest(lv[n], canonical_nan=True) for n in LEVEL_OUT},
                    **{n: digest(sf[n], canonical_nan=True) for n in SURFACE_OUT}}
                step_digest_mismatch += [
                    f"step {istep + 1} {n}" for n in graded
                    if canon_got[n] != canon_want[n]]
        lv, sf = gpu[-1]
        wrf_last = ({n: final[f"{tag}/{n}"] for n in LEVEL_OUT},
                    {n: final[f"{tag}/{n}"] for n in SURFACE_OUT})
        result = {"fields": {}}
        _accumulate(result["fields"], nsteps - 1, _pairs(wrf_last, (lv, sf), graded),
                    regimes, keep=None if not exclude_columns else keep)
        if minor_loops(dt) > 1 and len(gpu) > 1:
            result["rainncv_step_check"] = rainncv_step_check(
                gpu[-2][1]["rain"], gpu[-1],
                keep=None if not exclude_columns else keep)
        bad = _bad(result) + len(step_digest_mismatch)
        total += bad
        report["runs"][tag] = {"final_step": result["fields"],
                               "rainncv_step_check": result.get("rainncv_step_check"),
                               "step_digest_mismatch": step_digest_mismatch,
                               "raw_step_digest_mismatch": raw_step_digest_mismatch}
    report["total_mismatch"] = total
    report["declared_divergences"] = {
        "rainncv": "per-step fallout instead of WRF last minor loop",
        "nan_payload": "NaN sign and payload after WRF thin-layer instability; finite words and nonfinite masks remain graded"}
    report["nan_payload_mismatch"] = sum(
        f["nan_payload_mismatch"] for r in report["runs"].values()
        for f in r["final_step"].values())
    return report


#: The planted mutation: one bit of one revised WSM6R term.  pgmlt_b is the
#: graupel-melting ventilation word of wsm6rinit (module_mp_wsm6r.F:403),
#: a term WSM6R revised away from WSM6's precg2 form; its lowest mantissa
#: bit is flipped in the device coefficient table.
PLANT = ("pgmlt_b", 1)


def cmd_plant(fixture: Path):
    """A one-bit change in a revised term must turn the gate red."""
    import cupy as cp
    import gpuwm.core.wsm6 as wsm6
    from gpuwm.core.wsm6_constants import WSM6R_INIT_F32

    baseline = check_fixture(fixture)
    if baseline["total_mismatch"] != 0:
        raise RuntimeError("one-bit control requires a passing clean gate; "
                           "pre-existing mismatches would make detection vacuous")
    names = [n for n, _ in WSM6R_INIT_F32]
    index = names.index(PLANT[0])
    clean = wsm6._wsm6r_coefficients()
    words = clean.get().view(np.uint32).copy()
    words[index] ^= np.uint32(PLANT[1])
    planted = cp.asarray(words.view(np.float32))
    original = wsm6._wsm6r_coefficients
    wsm6._wsm6r_coefficients = lambda: planted
    try:
        report = check_fixture(fixture)
    finally:
        wsm6._wsm6r_coefficients = original
    report["baseline_mismatch"] = baseline["total_mismatch"]
    report["planted"] = {"word": PLANT[0], "xor": PLANT[1],
                         "clean_bits": f"{int(clean.get().view(np.uint32)[index]):08X}",
                         "planted_bits": f"{int(words[index]):08X}"}
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
    c.add_argument("--out", type=Path)
    pl = sub.add_parser("plant")
    pl.add_argument("fixture_dir", type=Path)
    pl.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "make-fixture":
        print(json.dumps(cmd_make_fixture(args.oracle_dir, args.out_dir)))
        return 0
    if args.cmd == "grade":
        report = cmd_grade(args.oracle_dir, args.flavour)
        bad = report["total_bit_mismatch"]
    elif args.cmd == "plant":
        report = cmd_plant(args.fixture_dir)
        bad = report["total_mismatch"]
    else:
        report = check_fixture(args.fixture_dir, exclude_columns=tuple(args.exclude))
        bad = report["total_mismatch"]
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"total_mismatch": bad, "strict_build": report["strict_build"],
                      "summary": {tag: {
                          "digest_mismatch": len(r.get("step_digest_mismatch", []) or []),
                          "final_step": {n: [f["bit_mismatch"], f["max_ulp"]]
                                         for n, f in (r.get("final_step") or r.get("fields")).items()
                                         if f["bit_mismatch"]}}
                          for tag, r in report["runs"].items()}}))
    if args.cmd == "plant":
        # The control PASSES when the planted bit is caught.
        return 0 if bad > 0 else 1
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
